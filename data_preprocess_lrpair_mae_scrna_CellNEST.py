# ============================================================================
# data_preprocess_lrpair_mae_scrna_CellNEST.py
# Written by: Fatema Tuz Zohora (参照 CellNEST 框架扩展)
#
# 目标
# ----
# 为“无先验 Tokenization + Masked Autoencoder”流程准备输入矩阵：
#   · Token = 配受体对（LR pair）
#   · 每个 Token 的初始特征 = 该配受体对在所有细胞类型对上的通讯强度向量
#   · 输出矩阵 X_lr：shape = (N_lr, M_ctpair)
#
# 设计说明（对应问题陈述）
# ========================
# 1. N_lr：筛选后的配受体对数量（默认取总活跃度最高的 4000）
# 2. M_ctpair：细胞类型对组合数（有序对 typeA → typeB）
# 3. X_lr[i, j]：配受体对 i 在细胞类型对 j 上的平均通讯分数
# 4. 不引入任何先验标签、位置编码或生物学注释
#
# 输出文件
# --------
# input_graph/<data_name>/<data_name>_lrpair_mae_tokens
#   内容（pickle + gzip）：
#     [X_lr, lr_id_to_pair, cp_id_to_pair, lr_total_score]
#     - X_lr            : ndarray (N_lr, M_ctpair)
#     - lr_id_to_pair   : dict {new_lr_id -> (ligand, receptor)}
#     - cp_id_to_pair   : dict {cp_id -> (typeA, typeB)}
#     - lr_total_score  : ndarray (N_lr,), 用于记录每个 LR 对的全局活跃度
#
# metadata/<data_name>/<data_name>_lrpair_mae_metadata.txt
#   记录 M、N 及筛选阈值等摘要信息
# ============================================================================

print('Loading dependencies... / 正在加载依赖包...')
import numpy as np
import pickle
from scipy import sparse
import qnorm
from collections import defaultdict
import pandas as pd
import gzip
import argparse
import os
import scanpy as sc
print('Dependencies loaded. / 依赖包加载完毕')


def _make_cell_type_pairs(cell_types, block_same_type):
    """生成有序细胞类型对列表。"""
    cp_list = []
    for sender in cell_types:
        for receiver in cell_types:
            if block_same_type == 1 and sender == receiver:
                continue
            cp_list.append((sender, receiver))
    cp_to_id = {cp: i for i, cp in enumerate(cp_list)}
    cp_id_to_pair = {i: cp for cp, i in cp_to_id.items()}
    return cp_to_id, cp_id_to_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'CellNEST scRNA-seq LR 对 MAE 预处理 —— '
            '构建 LR 对 Token 矩阵 (N_lr × M_ctpair)'
        )
    )

    # =================== 必填参数 =============================================
    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称')
    parser.add_argument('--data_from', type=str, required=True,
                        help='scRNA-seq 数据路径（.h5ad 格式）')

    # =================== 可选参数（已设默认值） ================================
    parser.add_argument('--cell_type_col', type=str, default='cell_type',
                        help='adata.obs 中细胞类型注释列名，默认为 "cell_type"')
    parser.add_argument('--data_to', type=str, default='input_graph/',
                        help='输出矩阵保存路径')
    parser.add_argument('--metadata_to', type=str, default='metadata/',
                        help='元数据保存路径')
    parser.add_argument('--filter_min_cell', type=int, default=1,
                        help='基因过滤：基因至少在多少个细胞中表达（默认 1）')
    parser.add_argument('--threshold_gene_exp', type=float, default=98,
                        help='基因表达活跃百分位阈值（默认 98）')
    parser.add_argument('--block_autocrine', type=int, default=0,
                        help='设为 1 则忽略自分泌信号（同一细胞同时为发送方和接收方）')
    parser.add_argument('--block_same_type', type=int, default=0,
                        help='设为 1 则忽略同细胞类型内部通讯（如 Tcell→Tcell）')
    parser.add_argument('--database_path', type=str,
                        default='database/CellNEST_database.csv',
                        help='配受体数据库路径')
    parser.add_argument('--top_lr_pairs', type=int, default=4000,
                        help='保留活跃度最高的前 N 个配受体对（0 表示保留全部）')
    parser.add_argument('--log1p', type=int, default=1,
                        help='设为 1 对通讯分数做 log1p 压缩（默认 1）')
    args = parser.parse_args()

    # =================== 路径初始化 ===========================================
    if args.data_to == 'input_graph/':
        args.data_to = args.data_to + args.data_name + '/'
    if not os.path.exists(args.data_to):
        os.makedirs(args.data_to)

    if args.metadata_to == 'metadata/':
        args.metadata_to = args.metadata_to + args.data_name + '/'
    if not os.path.exists(args.metadata_to):
        os.makedirs(args.metadata_to)

    # =================== 读取 scRNA-seq 数据 ==================================
    print('Reading scRNA-seq data... / 正在读取 scRNA-seq 数据...')
    adata = sc.read_h5ad(args.data_from)
    print('Data loaded. / 数据读取完毕')

    gene_count_before = len(list(adata.var_names))
    sc.pp.filter_genes(adata, min_cells=args.filter_min_cell)
    gene_count_after = len(list(adata.var_names))
    print(
        f'Gene filtering done: {gene_count_before} -> {gene_count_after} / '
        f'基因过滤完毕：基因数从 {gene_count_before} 减少至 {gene_count_after}'
    )

    gene_ids = list(adata.var_names)
    n_cells = adata.obs_names.shape[0]
    print(f'Total cells: {n_cells} / 细胞总数：{n_cells}')

    # =================== 细胞类型注释 =========================================
    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(
            'Cell-type column \"%s\" not found in adata.obs.\n'
            'Please set --cell_type_col to the correct column name.\n'
            'Available columns: %s'
            % (args.cell_type_col, list(adata.obs.columns))
        )

    cell_type_array = np.array(adata.obs[args.cell_type_col].astype(str))
    unique_cell_types = sorted(list(set(cell_type_array)))
    print(
        f'Found {len(unique_cell_types)} cell types: {unique_cell_types} / '
        f'发现 {len(unique_cell_types)} 种细胞类型：{unique_cell_types}'
    )

    # =================== 量化归一化 ==========================================
    print('Running quantile normalization... / 正在进行 quantile normalization...')
    temp = qnorm.quantile_normalize(
        np.transpose(sparse.csr_matrix.toarray(adata.X)))
    cell_vs_gene = np.transpose(temp)
    print(
        f'Quantile normalization done. Matrix shape: {cell_vs_gene.shape} / '
        f'量化归一化完毕，表达矩阵维度：{cell_vs_gene.shape}'
    )

    gene_index = {gene: i for i, gene in enumerate(gene_ids)}

    # =================== 读取配受体数据库 =====================================
    print('Loading LR database... / 正在读取配受体数据库...')
    gene_info = {gene: '' for gene in gene_ids}
    df_lr = pd.read_csv(args.database_path, sep=',')
    print('LR database loaded. / 配受体数据库读取完毕')

    ligand_dict_dataset = defaultdict(list)
    for i in range(df_lr['Ligand'].shape[0]):
        ligand = df_lr['Ligand'][i]
        if ligand not in gene_info:
            continue
        receptor = df_lr['Receptor'][i]
        if receptor not in gene_info:
            continue
        ligand_dict_dataset[ligand].append(receptor)
        gene_info[ligand] = 'included'
        gene_info[receptor] = 'included'

    # 为每个配受体对分配唯一 ID（lr_pair_id）
    l_r_pair = dict()
    lr_id = 0
    for gene in list(ligand_dict_dataset.keys()):
        ligand_dict_dataset[gene] = list(set(ligand_dict_dataset[gene]))
        l_r_pair[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            l_r_pair[gene][receptor_gene] = lr_id
            lr_id += 1

    num_lr_pairs = lr_id
    lr_id_to_pair = {}
    for lig in l_r_pair:
        for rec in l_r_pair[lig]:
            lr_id_to_pair[l_r_pair[lig][rec]] = (lig, rec)

    print(f'Total LR pairs in dataset: {num_lr_pairs} / 数据集中配受体对总数：{num_lr_pairs}')

    # =================== 基因表达活跃百分位阈值 ================================
    print(
        f'Computing per-cell expression thresholds ({args.threshold_gene_exp:.1f}%) ... / '
        f'正在计算各细胞基因表达百分位阈值（阈值：{args.threshold_gene_exp:.1f}%）...'
    )
    cell_percentile = []
    for i in range(n_cells):
        y = sorted(cell_vs_gene[i])
        cutoff = np.percentile(y, args.threshold_gene_exp)
        if cutoff == min(cell_vs_gene[i]):
            times = 1
            while cutoff == min(cell_vs_gene[i]):
                new_th = args.threshold_gene_exp + 5 * times
                if new_th >= 100:
                    cutoff = max(cell_vs_gene[i])
                    if cutoff == min(cell_vs_gene[i]):
                        cutoff = max(cell_vs_gene[i]) + 1
                    break
                cutoff = np.percentile(y, new_th)
                times += 1
        cell_percentile.append(cutoff)
    print('Expression thresholds computed. / 基因表达阈值计算完毕')

    # =================== 全枚举：计算每个（细胞类型对, 配受体对）的通讯分数 ======
    print('Enumerating active LR communications... / 正在枚举所有跨细胞类型活跃配受体通讯（全枚举模式）...')

    ct_pair_lr_score_sum = defaultdict(float)
    ct_pair_lr_count = defaultdict(int)
    lr_total_score_sum = defaultdict(float)

    ligand_list = list(ligand_dict_dataset.keys())
    total_active = 0

    for g_idx, gene in enumerate(ligand_list):
        gene_col = gene_index[gene]
        sender_cells = [i for i in range(n_cells)
                        if cell_vs_gene[i][gene_col] >= cell_percentile[i]]
        if len(sender_cells) == 0:
            print(
                f'Processed ligands {g_idx + 1}/{len(ligand_list)} / '
                f'{g_idx + 1}/{len(ligand_list)} 配体基因已处理',
                end='\r'
            )
            continue

        for gene_rec in ligand_dict_dataset[gene]:
            rec_col = gene_index[gene_rec]
            relation_id = l_r_pair[gene][gene_rec]

            receiver_cells = [j for j in range(n_cells)
                              if cell_vs_gene[j][rec_col] >= cell_percentile[j]]
            if len(receiver_cells) == 0:
                continue

            for i in sender_cells:
                type_i = cell_type_array[i]
                score_i = cell_vs_gene[i][gene_col]

                for j in receiver_cells:
                    if args.block_autocrine == 1 and i == j:
                        continue
                    type_j = cell_type_array[j]
                    if args.block_same_type == 1 and type_i == type_j:
                        continue

                    communication_score = score_i * cell_vs_gene[j][rec_col]
                    if communication_score <= 0:
                        continue

                    ct_pair_key = (type_i, type_j)
                    key = (ct_pair_key, relation_id)
                    ct_pair_lr_score_sum[key] += communication_score
                    ct_pair_lr_count[key] += 1
                    lr_total_score_sum[relation_id] += communication_score
                    total_active += 1

        print(
            f'Processed ligands {g_idx + 1}/{len(ligand_list)} / '
            f'{g_idx + 1}/{len(ligand_list)} 配体基因已处理',
            end='\r'
        )

    print('')
    print(
        f'Active communications: {total_active} / '
        f'共发现 {total_active} 条活跃通讯记录（按细胞对×配受体对计数）'
    )

    # =================== 构建细胞类型对列表 ===================================
    cp_to_id, cp_id_to_pair = _make_cell_type_pairs(
        unique_cell_types, args.block_same_type)
    num_cp_pairs = len(cp_to_id)
    print(f'Cell-type pairs (M): {num_cp_pairs} / 细胞类型对组合数（M）：{num_cp_pairs}')

    # =================== 选择活跃度最高的 LR 对 ================================
    active_lr_ids = list(lr_total_score_sum.keys())
    if len(active_lr_ids) == 0:
        raise RuntimeError(
            'No active LR pairs detected. Consider lowering '
            '--threshold_gene_exp (current: %.1f) or checking the LR database '
            'and input expression matrix.' % args.threshold_gene_exp
        )

    sorted_lr_ids = sorted(
        active_lr_ids,
        key=lambda lr_k: lr_total_score_sum[lr_k],
        reverse=True
    )

    if args.top_lr_pairs > 0 and len(sorted_lr_ids) > args.top_lr_pairs:
        selected_lr_ids = sorted_lr_ids[:args.top_lr_pairs]
    else:
        selected_lr_ids = sorted_lr_ids

    selected_lr_ids = sorted(selected_lr_ids)
    old_to_new_lr = {old_id: new_id for new_id, old_id in enumerate(selected_lr_ids)}

    print(f'Selected LR pairs (N): {len(selected_lr_ids)} / 筛选后 LR 对数量（N）：{len(selected_lr_ids)}')

    # =================== 构建 LR Token 矩阵 ====================================
    X_lr = np.zeros((len(selected_lr_ids), num_cp_pairs), dtype=np.float32)
    for (ct_pk, lr_k), score_sum in ct_pair_lr_score_sum.items():
        if lr_k not in old_to_new_lr:
            continue
        count = ct_pair_lr_count[(ct_pk, lr_k)]
        mean_score = score_sum / count
        cp_id = cp_to_id[ct_pk]
        new_lr_id = old_to_new_lr[lr_k]
        X_lr[new_lr_id, cp_id] = float(mean_score)

    if args.log1p == 1:
        X_lr = np.log1p(X_lr)
        print('Applied log1p to scores. / 已对通讯分数执行 log1p 压缩')

    lr_total_score = np.array(
        [lr_total_score_sum[old_id] for old_id in selected_lr_ids],
        dtype=np.float32
    )
    lr_id_to_pair_selected = {
        old_to_new_lr[old_id]: lr_id_to_pair[old_id]
        for old_id in selected_lr_ids
    }

    # =================== 保存输出 =============================================
    output_path = args.data_to + args.data_name + '_lrpair_mae_tokens'
    print(f'Saving LR token matrix: {output_path} / 正在保存 LR Token 矩阵至：{output_path}')
    with gzip.open(output_path, 'wb') as fp:
        pickle.dump([X_lr, lr_id_to_pair_selected, cp_id_to_pair, lr_total_score], fp)

    meta_path = args.metadata_to + args.data_name + '_lrpair_mae_metadata.txt'
    with open(meta_path, 'w') as f:
        f.writelines([
            f'细胞类型对组合数（M）：{num_cp_pairs}\n',
            f'配受体对总数（数据库匹配）：{num_lr_pairs}\n',
            f'筛选后 LR 对数量（N）：{len(selected_lr_ids)}\n',
            f'top_lr_pairs 参数：{args.top_lr_pairs}\n',
            f'log1p 压缩：{"yes" if args.log1p == 1 else "no"}\n',
        ])

    print(f'Metadata saved: {meta_path} / 元数据保存至：{meta_path}')
    print('Preprocessing complete. / 预处理完毕。')
    print('Next: run run_CellNEST_lrpair_mae_scrna.py for MAE training. / '
          '下一步：运行 run_CellNEST_lrpair_mae_scrna.py 进行 MAE 训练。')
