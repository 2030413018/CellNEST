# ============================================================================
# data_preprocess_celltype_multiedge_scrna_CellNEST.py
#
# 目标
# ----
# 针对 scRNA-seq 数据，以 **细胞类型** 为节点、**活跃配受体对** 为平行有向边
# 构建有向多重图 G_s = (V_s, E_s, X_s)，每个 sampleID 对应一张独立的图。
#
# 图设计（对照问题描述）
# ======================
#
# 节点（Nodes, V_s）
#   · 该样本内 adata.obs[cell_type_col] 的 unique 值（每种细胞亚群一个节点）。
#   · 若样本有 15 种细胞亚群，则图有 15 个节点。
#
# 节点特征（Node Features, X_s）
#   · 对每个细胞类型 c，取该类型内所有细胞在"所有配体 + 受体基因集合"上的
#     平均表达向量：x_c[g] = mean_{i ∈ C_c}(expr_i(g))
#   · 特征维度 d = 数据集中出现的唯一 LR 基因总数（配体 ∪ 受体）。
#   · 建图前进行行 L2 归一化。
#
# 有向多重边（Edges, E_s）
#   · 对每个有序细胞类型对 (u, v) 及每个 LR 对 p = (l, r)：
#     若 p 在 (u, v) 上"活跃"（沿用 CellNEST 的个体细胞百分位阈值判定），
#     则添加一条平行有向边 u → v，边特征 = 1 维标量：
#         w = mean_expr_u(l) × mean_expr_v(r)
#   · 同一 (u, v) 对可有多条平行边（对应不同活跃 LR 对）。
#   · 平行边条数 m_{uv} = 该对上活跃 LR 对的总数。
#
# 活跃判定规则（与 CellNEST 一致）
#   · 对样本中每个细胞独立计算 threshold_gene_exp 百分位数作为活跃阈值。
#   · LR 对 (l, r) 在细胞类型对 (u, v) 上活跃，当且仅当：
#       - 细胞类型 u 中至少有一个细胞其配体 l 的表达量 >= 该细胞的阈值
#       - 细胞类型 v 中至少有一个细胞其受体 r 的表达量 >= 该细胞的阈值
#
# 输出文件（pickle, gzip 压缩）
# --------
#   input_graph/<data_name>/<data_name>_celltype_multiedge_adjacency_records
#   内容（列表）：
#     [row_col, edge_weight, num_nodes, X_feature, ct_id_to_name, lr_gene_list]
#     · row_col       : list of [src_ct_id, dst_ct_id]（有向多重边）
#     · edge_weight   : list of [w]（通讯强度标量，1 维）
#     · num_nodes     : 节点数 |V_s|
#     · X_feature     : ndarray (|V_s|, d)，L2 归一化后的节点特征矩阵
#     · ct_id_to_name : dict {node_id -> cell_type_name}
#     · lr_gene_list  : list，X_feature 列对应的基因名称列表（长度 d）
# ============================================================================

print('正在加载依赖包...')
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
print('依赖包加载完毕')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'CellNEST scRNA-seq 细胞类型多重边图预处理 —— '
            '节点=细胞类型，边=活跃配受体对（多重有向边）'
        )
    )

    # =================== 必填参数 =============================================
    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称（建议包含 sampleID，如 mydata_S1）')
    parser.add_argument('--data_from', type=str, required=True,
                        help='scRNA-seq 数据路径（.h5ad 格式，含原始计数）')

    # =================== 可选参数 ============================================
    parser.add_argument('--sample_col', type=str, default='sampleID',
                        help='adata.obs 中样本列名（默认 "sampleID"）')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='要处理的样本 ID；若为 None 则使用全部细胞')
    parser.add_argument('--cell_type_col', type=str, default='subCluster',
                        help='adata.obs 中细胞类型注释列名（默认 "subCluster"）')
    parser.add_argument('--data_to', type=str, default='input_graph/',
                        help='图文件保存根路径（默认 "input_graph/"）')
    parser.add_argument('--metadata_to', type=str, default='metadata/',
                        help='元数据保存根路径（默认 "metadata/"）')
    parser.add_argument('--filter_min_cell', type=int, default=1,
                        help='基因过滤：基因至少在多少个细胞中表达（默认 1）')
    parser.add_argument('--threshold_gene_exp', type=float, default=98,
                        help='基因表达活跃百分位阈值（默认 98）')
    parser.add_argument('--block_autocrine', type=int, default=0,
                        help='设为 1 则忽略自分泌信号（同细胞类型 u=v）')
    parser.add_argument('--database_path', type=str,
                        default='database/CellNEST_database.csv',
                        help='配受体数据库路径（默认 "database/CellNEST_database.csv"）')
    args = parser.parse_args()

    # =================== 路径初始化 ==========================================
    if args.data_to == 'input_graph/':
        args.data_to = args.data_to + args.data_name + '/'
    if not os.path.exists(args.data_to):
        os.makedirs(args.data_to)

    if args.metadata_to == 'metadata/':
        args.metadata_to = args.metadata_to + args.data_name + '/'
    if not os.path.exists(args.metadata_to):
        os.makedirs(args.metadata_to)

    # =================== 读取 scRNA-seq 数据 =================================
    print('正在读取 scRNA-seq 数据：%s' % args.data_from)
    adata_all = sc.read_h5ad(args.data_from)

    # 按 sampleID 过滤
    if args.sample_id is not None:
        if args.sample_col not in adata_all.obs.columns:
            raise ValueError(
                '在 adata.obs 中未找到样本列 "%s"。\n'
                '可用列：%s' % (args.sample_col, list(adata_all.obs.columns)))
        mask = adata_all.obs[args.sample_col].astype(str) == str(args.sample_id)
        adata = adata_all[mask].copy()
        print('样本 "%s" 过滤后细胞数：%d' % (args.sample_id, adata.n_obs))
    else:
        adata = adata_all.copy()
        print('未指定 --sample_id，使用全部细胞，共 %d 个' % adata.n_obs)

    if adata.n_obs == 0:
        raise ValueError('样本 "%s" 过滤后细胞数为 0，请检查 --sample_col 和 --sample_id。'
                         % args.sample_id)

    # 基因过滤
    gene_count_before = len(list(adata.var_names))
    sc.pp.filter_genes(adata, min_cells=args.filter_min_cell)
    gene_count_after = len(list(adata.var_names))
    print('基因过滤完毕：%d → %d' % (gene_count_before, gene_count_after))

    gene_ids = list(adata.var_names)
    n_cells = adata.n_obs
    print('细胞总数：%d，基因数：%d' % (n_cells, len(gene_ids)))

    # =================== 细胞类型注释 ========================================
    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(
            '在 adata.obs 中未找到细胞类型列 "%s"。\n'
            '当前可用列：%s' % (args.cell_type_col, list(adata.obs.columns)))

    cell_type_array = np.array(adata.obs[args.cell_type_col].astype(str))
    unique_cell_types = sorted(list(set(cell_type_array)))
    print('发现 %d 种细胞类型：%s' % (len(unique_cell_types), unique_cell_types))

    if len(unique_cell_types) < 2:
        raise ValueError(
            '样本中细胞类型数量不足（仅 %d 种），无法构建有意义的细胞间通讯图。'
            % len(unique_cell_types))

    # =================== 量化归一化 ==========================================
    # 与 CellNEST 其他 scRNA-seq 模块保持一致
    print('正在进行 quantile normalization...')
    X_mat = adata.X
    if sparse.issparse(X_mat):
        X_mat = X_mat.toarray()
    temp = qnorm.quantile_normalize(np.transpose(X_mat))
    cell_vs_gene = np.transpose(temp)  # shape: (n_cells, n_genes)
    print('量化归一化完毕，表达矩阵维度：', cell_vs_gene.shape)

    gene_index = {gene: i for i, gene in enumerate(gene_ids)}

    # =================== 构建细胞类型索引 =====================================
    cells_of_type = defaultdict(list)
    for idx, ct in enumerate(cell_type_array):
        cells_of_type[ct].append(idx)
    for ct in unique_cell_types:
        print('  %s：%d 个细胞' % (ct, len(cells_of_type[ct])))

    # 细胞类型节点 ID 映射
    ct_to_id = {ct: i for i, ct in enumerate(unique_cell_types)}
    ct_id_to_name = {i: ct for ct, i in ct_to_id.items()}
    num_ct_nodes = len(unique_cell_types)
    print('节点数（细胞类型数）：%d' % num_ct_nodes)

    # =================== 读取配受体数据库 =====================================
    print('正在读取配受体数据库：%s' % args.database_path)
    gene_info = {gene: '' for gene in gene_ids}
    df_lr = pd.read_csv(args.database_path, sep=',')
    print('数据库读取完毕，共 %d 行' % len(df_lr))

    ligand_dict_dataset = defaultdict(list)  # ligand -> [receptor, ...]
    for i in range(df_lr["Ligand"].shape[0]):
        ligand = df_lr["Ligand"][i]
        if ligand not in gene_info:
            continue
        receptor = df_lr["Receptor"][i]
        if receptor not in gene_info:
            continue
        ligand_dict_dataset[ligand].append(receptor)
        gene_info[ligand] = 'included'
        gene_info[receptor] = 'included'

    # 去重配体对应的受体列表
    for gene in list(ligand_dict_dataset.keys()):
        ligand_dict_dataset[gene] = list(set(ligand_dict_dataset[gene]))

    num_lr_pairs_total = sum(len(v) for v in ligand_dict_dataset.values())
    print('数据集中可用配受体对总数：%d' % num_lr_pairs_total)

    if num_lr_pairs_total == 0:
        raise ValueError('数据集中没有与数据库匹配的配受体对，请检查数据库路径和基因名称格式。')

    # =================== 收集所有 LR 基因（用于节点特征）======================
    # 节点特征 = 细胞类型在所有配体 + 受体基因上的平均表达
    lr_genes_set = set()
    for lig, recs in ligand_dict_dataset.items():
        lr_genes_set.add(lig)
        for rec in recs:
            lr_genes_set.add(rec)
    lr_gene_list = sorted(list(lr_genes_set))  # 固定顺序，保证可复现
    d = len(lr_gene_list)
    lr_gene_to_feat_idx = {g: i for i, g in enumerate(lr_gene_list)}
    print('LR 基因集合大小（节点特征维度 d）：%d' % d)

    # =================== 计算每种细胞类型的平均表达（LR 基因子集）==============
    # ct_mean_expr[ct][g_idx] = 细胞类型 ct 在基因 g（gene_ids索引）上的平均表达
    print('正在计算各细胞类型的平均基因表达...')
    ct_mean_expr = {}
    for ct in unique_cell_types:
        idxs = cells_of_type[ct]
        # 对该细胞类型所有细胞取平均（shape: n_genes）
        mean_all = cell_vs_gene[idxs, :].mean(axis=0)
        ct_mean_expr[ct] = mean_all  # 全基因均值向量，按需查询

    # 构建节点特征矩阵 X（|V| × d）
    # X[ct_id, feat_idx] = 细胞类型 ct 中 LR 基因 lr_gene_list[feat_idx] 的平均表达
    X_raw = np.zeros((num_ct_nodes, d), dtype=np.float32)
    for ct in unique_cell_types:
        ct_id = ct_to_id[ct]
        for g in lr_gene_list:
            g_col = gene_index[g]       # 在全基因矩阵中的列索引
            feat_idx = lr_gene_to_feat_idx[g]
            X_raw[ct_id, feat_idx] = float(ct_mean_expr[ct][g_col])

    print('节点特征矩阵构建完毕，维度：%d × %d' % (num_ct_nodes, d))

    # L2 行归一化
    row_norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X_normalized = (X_raw / row_norms).astype(np.float32)
    print('L2 行归一化完毕')

    # =================== 计算每细胞的活跃阈值（与 CellNEST 一致）==============
    # 对每个细胞独立计算第 threshold_gene_exp 百分位数，作为"高表达"判定门槛
    print('正在计算各细胞基因表达百分位阈值（%.1f%%）...' % args.threshold_gene_exp)
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
    print('阈值计算完毕')

    # =================== 预计算每种细胞类型在每个基因上的活跃性================
    # 对每个基因 g 和每种细胞类型 ct，判断是否有 >= 1 个细胞高表达 g
    # active_sender[ct][g_col] = True/False
    # 同时缓存每种细胞类型中高表达该基因的细胞数（用于快速查询活跃性）
    print('正在预计算各细胞类型的基因活跃性...')
    # 为节省内存，只计算 LR 基因
    ct_lig_active = {}   # ct -> set of ligand gene names that are active in ct
    ct_rec_active = {}   # ct -> set of receptor gene names that are active in ct

    all_lig_genes = set(ligand_dict_dataset.keys())
    all_rec_genes = set()
    for recs in ligand_dict_dataset.values():
        all_rec_genes.update(recs)
    all_lr_genes_for_active = all_lig_genes | all_rec_genes

    for ct in unique_cell_types:
        idxs = cells_of_type[ct]
        active_genes = set()
        for g in all_lr_genes_for_active:
            g_col = gene_index[g]
            for ci in idxs:
                if cell_vs_gene[ci][g_col] >= cell_percentile[ci]:
                    active_genes.add(g)
                    break  # 只需一个细胞满足即可
        ct_lig_active[ct] = active_genes & all_lig_genes
        ct_rec_active[ct] = active_genes & all_rec_genes

    print('活跃性预计算完毕')

    # =================== 构建多重有向边列表 ====================================
    #
    # 边定义：
    #   对每个有序细胞类型对 (u, v) 和每个 LR 对 (l, r)：
    #     若 l 在细胞类型 u 中活跃（至少一个细胞高表达）
    #       AND r 在细胞类型 v 中活跃
    #     则添加有向边 u → v，边权重 = mean_expr_u(l) × mean_expr_v(r)
    #
    # 结果为有向多重图：同一 (u,v) 对可有多条平行边（对应不同活跃 LR 对）
    # -------------------------------------------------------------------------
    print('正在构建多重有向边列表...')

    row_col = []      # [src_ct_id, dst_ct_id]
    edge_weight = []  # [w]（通讯强度标量）

    total_edges = 0
    ligand_list = list(ligand_dict_dataset.keys())

    for u in unique_cell_types:
        u_id = ct_to_id[u]
        for v in unique_cell_types:
            if args.block_autocrine == 1 and u == v:
                continue
            v_id = ct_to_id[v]

            for lig in ligand_list:
                # 配体活跃性检查：u 中是否有细胞高表达配体 lig
                if lig not in ct_lig_active[u]:
                    continue
                lig_col = gene_index[lig]
                mean_lig_u = float(ct_mean_expr[u][lig_col])

                for rec in ligand_dict_dataset[lig]:
                    # 受体活跃性检查：v 中是否有细胞高表达受体 rec
                    if rec not in ct_rec_active[v]:
                        continue
                    rec_col = gene_index[rec]
                    mean_rec_v = float(ct_mean_expr[v][rec_col])

                    w = mean_lig_u * mean_rec_v
                    if w <= 0:
                        continue

                    row_col.append([u_id, v_id])
                    edge_weight.append([w])
                    total_edges += 1

    print('多重有向边构建完毕：%d 个节点，%d 条有向边（多重有向图）'
          % (num_ct_nodes, total_edges))
    if num_ct_nodes > 0:
        print('平均每节点出度：%.1f' % (total_edges / num_ct_nodes))

    if total_edges == 0:
        print('警告：未发现任何活跃配受体通讯，请检查数据质量或降低 --threshold_gene_exp。')

    # 统计每对细胞类型的平行边数
    ct_pair_edge_count = defaultdict(int)
    for rc in row_col:
        ct_pair_edge_count[(rc[0], rc[1])] += 1
    print('细胞类型对通讯统计（活跃 LR 对数）：')
    for (u_id, v_id), cnt in sorted(ct_pair_edge_count.items(),
                                    key=lambda x: -x[1])[:10]:
        print('  %s → %s：%d 条平行边'
              % (ct_id_to_name[u_id], ct_id_to_name[v_id], cnt))

    # =================== 保存图文件 ==========================================
    output_path = (args.data_to + args.data_name
                   + '_celltype_multiedge_adjacency_records')
    print('正在保存图文件至：%s' % output_path)
    with gzip.open(output_path, 'wb') as fp:
        pickle.dump([row_col, edge_weight, num_ct_nodes,
                     X_normalized, ct_id_to_name, lr_gene_list], fp)
    print('图文件保存完毕')

    # =================== 保存元数据 ==========================================
    meta_path = (args.metadata_to + args.data_name
                 + '_celltype_multiedge_metadata.txt')
    with open(meta_path, 'w') as f:
        f.write('样本 ID：%s\n' % str(args.sample_id))
        f.write('细胞总数：%d\n' % n_cells)
        f.write('细胞类型节点数（|V|）：%d\n' % num_ct_nodes)
        f.write('节点特征维度（d = LR 基因数）：%d\n' % d)
        f.write('有向多重边总数：%d\n' % total_edges)
        f.write('\n细胞类型节点列表：\n')
        for nid in sorted(ct_id_to_name.keys()):
            f.write('  节点 %d：%s\n' % (nid, ct_id_to_name[nid]))
        f.write('\n每对细胞类型的活跃 LR 对数（平行边数）：\n')
        for (u_id, v_id), cnt in sorted(ct_pair_edge_count.items(),
                                        key=lambda x: -x[1]):
            f.write('  %s → %s：%d\n'
                    % (ct_id_to_name[u_id], ct_id_to_name[v_id], cnt))

    # 保存细胞类型节点 ID 映射 CSV
    ct_id_df = pd.DataFrame(
        [(v, k) for k, v in sorted(ct_to_id.items(), key=lambda x: x[1])],
        columns=['node_id', 'cell_type']
    )
    ct_id_df.to_csv(
        args.metadata_to + 'ct_node_ids_' + args.data_name + '.csv',
        index=False
    )

    print('元数据保存至：%s' % args.metadata_to)
    print('')
    print('预处理完毕。下一步运行训练：')
    print('  python run_CellNEST_celltype_multiedge_scrna.py \\')
    print('      --data_name %s \\' % args.data_name)
    print('      --model_name CellNEST_ct_multiedge_%s \\' % args.data_name)
    print('      --run_id 0')
