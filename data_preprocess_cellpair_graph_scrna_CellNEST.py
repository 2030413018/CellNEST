# ============================================================================
# data_preprocess_cellpair_graph_scrna_CellNEST.py
# Written by: Fatema Tuz Zohora（参照 CellNEST 框架）
#
# 目标
# ----
# 对 scRNA-seq 数据，构建一个"细胞对图（Cell-Pair Graph）"：
#   · 节点  = 有序细胞类型对（typeA → typeB），每对代表一条潜在通讯通道
#   · 节点特征 = 该细胞类型对在所有活跃配受体对上的通讯分数向量（length = M）
#   · 边    = 配受体对：若两个细胞类型对通过同一配受体对相连，则画一条边
#   · 边特征 = [源节点该LR对通讯分数, 目标节点该LR对通讯分数, 配受体对ID]
#
# 核心设计
# ========
#
# 1. 节点（Cell-Type Pair Nodes）
#    · 对每一对有序细胞类型 (typeA, typeB)，若 typeA 中的某细胞高表达某配体、
#      且 typeB 中的某细胞高表达对应受体，则认为存在该配受体通路的通讯潜力。
#    · 节点特征向量 X[i] ∈ R^M：
#        X[i, k] = 细胞类型对 i 在配受体对 k 上的平均通讯分数（0 = 不活跃）
#    · 这与 CellNEST 双分图中细胞类型方向节点的特征含义完全一致。
#
# 2. 边（Ligand-Receptor Pair Edges）
#    · 对每个活跃配受体对 k：
#        - 找出所有在配受体对 k 上有非零通讯分数的细胞类型对集合 P_k
#        - 对 P_k 中所有节点对 (i, j)（i ≠ j），添加一条有向边 i → j，
#          边特征 = [X[i, k], X[j, k], float(k)]
#    · 这样每条边直接对应一个具体的配受体对，体现"边 = 配受体对"的设计。
#    · 结果为多重有向图（multigraph）：同一对节点间可有多条边（对应不同LR对）。
#
# 3. 与其他 CellNEST 设计的区别
#    · 双分图（bipartite）：LR对节点 + 细胞类型对节点，异类节点间连边
#      → 适合发现配受体通路的共激活模块
#    · LR对图（lr_graph）：LR对为节点，余弦相似度为边
#      → 适合发现相似通讯模式的LR对集群
#    · 本脚本（cell-pair graph）：细胞类型对为节点，配受体对为边
#      → 适合发现具有相似通讯模式的细胞类型对群落（Community）
#        嵌入相近的细胞类型对倾向于使用相同的分子信号通路
#
# 输出文件
# --------
#   input_graph/<data_name>/<data_name>_cellpair_graph_scrna_adjacency_records
#   内容（pickle 列表）：
#     [row_col, edge_weight, lig_rec, num_nodes, X_feature, cp_id_to_pair]
#     - row_col       : list of [src_cp_id, dst_cp_id]（有向边）
#     - edge_weight   : list of [score_src, score_dst, lr_pair_id]（边特征）
#     - lig_rec       : list of [ligand_name, receptor_name]（对应该边的配受体对）
#     - num_nodes     : 活跃细胞类型对节点数 N
#     - X_feature     : numpy ndarray (N, M)，已 L2 归一化的节点特征矩阵
#     - cp_id_to_pair : dict {cp_id -> (typeA, typeB)}，节点ID到细胞类型对的映射
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
            'CellNEST scRNA-seq 细胞对图预处理 —— '
            '以细胞类型对为节点、配受体对为边构建图（用于 GAT+DGI 训练）'
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
                        help='图文件保存路径')
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

    # =================== 读取 scRNA-seq 数据 ==================================
    print('正在读取 scRNA-seq 数据...')
    adata = sc.read_h5ad(args.data_from)
    print('数据读取完毕')

    gene_count_before = len(list(adata.var_names))
    sc.pp.filter_genes(adata, min_cells=args.filter_min_cell)
    gene_count_after = len(list(adata.var_names))
    print('基因过滤完毕：基因数从 %d 减少至 %d' % (gene_count_before, gene_count_after))

    gene_ids = list(adata.var_names)
    cell_barcode = np.array(adata.obs_names)
    n_cells = cell_barcode.shape[0]
    print('细胞总数：%d' % n_cells)

    # =================== 细胞类型注释 =========================================
    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(
            '在 adata.obs 中未找到细胞类型列 "%s"。\n'
            '请使用 --cell_type_col 指定正确的列名。\n'
            '当前可用列：%s' % (args.cell_type_col, list(adata.obs.columns)))

    cell_type_array = np.array(adata.obs[args.cell_type_col].astype(str))
    unique_cell_types = sorted(list(set(cell_type_array)))
    print('发现 %d 种细胞类型：%s' % (len(unique_cell_types), unique_cell_types))

    # =================== 量化归一化 ==========================================
    # 与 CellNEST 其他 scRNA-seq 模块保持一致，使用 quantile normalization 消除
    # 测序深度差异，使不同细胞之间的基因表达量可比。
    print('正在进行 quantile normalization...')
    temp = qnorm.quantile_normalize(
        np.transpose(sparse.csr_matrix.toarray(adata.X)))
    cell_vs_gene = np.transpose(temp)
    print('量化归一化完毕，表达矩阵维度：', cell_vs_gene.shape)

    # =================== 构建细胞类型索引 =====================================
    # cells_of_type[cell_type] = 该类型所有细胞的行索引列表
    print('正在构建细胞类型索引...')
    cells_of_type = defaultdict(list)
    for idx, ct in enumerate(cell_type_array):
        cells_of_type[ct].append(idx)
    for ct in unique_cell_types:
        print('  %s：%d 个细胞' % (ct, len(cells_of_type[ct])))

    gene_index = {gene: i for i, gene in enumerate(gene_ids)}

    # =================== 读取配受体数据库 =====================================
    print('正在读取配受体数据库...')
    gene_info = {gene: '' for gene in gene_ids}
    df_lr = pd.read_csv(args.database_path, sep=",")
    print('配受体数据库读取完毕')

    ligand_dict_dataset = defaultdict(list)
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
    # 构建 lr_id → (ligand, receptor) 反向映射
    lr_id_to_pair = {}
    for lig in l_r_pair:
        for rec in l_r_pair[lig]:
            lr_id_to_pair[l_r_pair[lig][rec]] = (lig, rec)

    print('数据集中配受体对总数：%d' % num_lr_pairs)

    # =================== 基因表达活跃百分位阈值 ================================
    # 与 CellNEST bipartite scrna 完全相同：每个细胞独立计算其第 threshold 百分位数。
    # 某细胞某基因的表达值超过该阈值时，认为该基因在该细胞中"活跃表达"。
    print('正在计算各细胞基因表达百分位阈值（阈值：%.1f%%）...'
          % args.threshold_gene_exp)
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
    print('基因表达阈值计算完毕')

    # =================== 全枚举：计算每个（细胞类型对, 配受体对）的通讯分数 ======
    # -------------------------------------------------------------------------
    # 策略（与 bipartite scrna 完全一致）：
    #   对每个配体 gene 和受体 gene_rec：
    #     · 找出所有高表达 gene 的细胞（发送方候选）
    #     · 找出所有高表达 gene_rec 的细胞（接收方候选）
    #     · 对每对（发送方 i, 接收方 j），以细胞类型对 (typeA, typeB) 为单位，
    #       累加通讯分数（= 配体表达量 × 受体表达量），并记录计数
    #   最终平均分数 mean_score = score_sum / count
    # -------------------------------------------------------------------------
    print('正在枚举所有跨细胞类型活跃配受体通讯（全枚举模式）...')

    ct_pair_lr_score_sum = defaultdict(float)
    ct_pair_lr_count = defaultdict(int)

    ligand_list = list(ligand_dict_dataset.keys())
    total_active = 0

    for g_idx, gene in enumerate(ligand_list):
        gene_col = gene_index[gene]
        sender_cells = [i for i in range(n_cells)
                        if cell_vs_gene[i][gene_col] >= cell_percentile[i]]
        if len(sender_cells) == 0:
            print('%d/%d 配体基因已处理' % (g_idx + 1, len(ligand_list)), end='\r')
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
                    total_active += 1

        print('%d/%d 配体基因已处理' % (g_idx + 1, len(ligand_list)), end='\r')

    print('')
    print('共发现 %d 条活跃通讯记录（按细胞对×配受体对计数）' % total_active)

    # =================== 构建细胞类型对节点 ===================================
    # Step 1：枚举所有出现过活跃通讯的细胞类型对，分配节点 ID
    cp_to_id = {}
    cp_id_counter = 0
    active_ct_pairs = set(ct_pk for (ct_pk, _) in ct_pair_lr_score_sum.keys())
    for ct_pk in sorted(active_ct_pairs):
        if ct_pk not in cp_to_id:
            cp_to_id[ct_pk] = cp_id_counter
            cp_id_counter += 1

    num_cp_nodes = cp_id_counter   # N：细胞类型对节点数
    cp_id_to_pair = {v: k for k, v in cp_to_id.items()}  # 供下游分析使用

    print('细胞类型对节点数（N）：%d' % num_cp_nodes)

    # Step 2：构建节点特征矩阵 X（N × M）
    # X[i, k] = 细胞类型对 i 在配受体对 k 上的平均通讯分数（0 = 不活跃）
    # M = num_lr_pairs（配受体对总数）
    # ------------------------------------------------------------------
    # 设计解释：
    #   · X[i] 编码了细胞类型对 i 的"通讯指纹"——哪些配受体通路在该对中活跃、
    #     以及活跃程度（均值通讯分数）。
    #   · L2 归一化后，余弦相似度可直接由点积计算，消除不同细胞类型对
    #     因细胞数量差异导致的分数量级不同。
    # ------------------------------------------------------------------
    M = num_lr_pairs
    X_raw = np.zeros((num_cp_nodes, M), dtype=np.float32)
    for (ct_pk, lr_k), score_sum in ct_pair_lr_score_sum.items():
        count = ct_pair_lr_count[(ct_pk, lr_k)]
        mean_score = score_sum / count
        cp_id = cp_to_id[ct_pk]
        X_raw[cp_id, lr_k] = float(mean_score)

    print('节点特征矩阵维度（N×M）：%d × %d' % (num_cp_nodes, M))

    # L2 行归一化（与 LR-pair graph 预处理保持一致）
    row_norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X_normalized = (X_raw / row_norms).astype(np.float32)
    print('L2 行归一化完毕')

    # =================== 构建边列表（边 = 配受体对） ===========================
    # -------------------------------------------------------------------------
    # 边的定义：
    #   对每个配受体对 k，找出所有在配受体对 k 上有非零分数的细胞类型对集合 P_k。
    #   对 P_k 中所有有序节点对 (i, j)（i ≠ j），添加一条有向边 i → j，
    #   边特征 = [X_raw[i, k],   # 源节点对配受体对 k 的通讯分数
    #             X_raw[j, k],   # 目标节点对配受体对 k 的通讯分数
    #             float(k)]      # 配受体对 ID（标记哪个LR对连接了这两个细胞类型对）
    #
    # 生物学意义：
    #   · i → j 的边表示"细胞类型对 i 和细胞类型对 j 都通过配受体对 k 进行通讯"。
    #   · 在 GAT 中，节点 i 的更新会聚合所有通过同一 LR 对与其相连的邻居节点的信息，
    #     注意力权重反映了"与我通过同一LR对通讯的其他细胞类型对有多重要"。
    #
    # 多重有向图（multigraph）：
    #   同一对节点 (i, j) 可能通过多个不同的 LR 对相连，对应多条平行边。
    #   GATv2 的消息传递机制会分别处理每条边，对每条边独立计算注意力权重，
    #   这正是我们所期望的——不同 LR 对连接同一对细胞类型对的权重可以不同。
    # -------------------------------------------------------------------------
    print('正在构建边列表（边 = 配受体对）...')

    row_col = []      # [源节点id, 目标节点id]
    edge_weight = []  # [score_src, score_dst, lr_pair_id]
    lig_rec = []      # [ligand_name, receptor_name]

    # 对每个配受体对，找出所有活跃的细胞类型对节点
    # 先按 lr_k 分组
    lr_to_active_cps = defaultdict(list)  # lr_k → [cp_id_1, cp_id_2, ...]
    for (ct_pk, lr_k) in ct_pair_lr_score_sum.keys():
        cp_id = cp_to_id[ct_pk]
        lr_to_active_cps[lr_k].append(cp_id)

    total_edges = 0
    for lr_k, cp_ids in lr_to_active_cps.items():
        if len(cp_ids) < 2:
            # 只有一个细胞类型对使用该LR对，无法建边，跳过
            continue

        lig_name, rec_name = lr_id_to_pair[lr_k]
        score_col = X_raw[:, lr_k]  # 所有节点对该LR对的通讯分数

        # 对该 LR 对的所有活跃细胞类型对，两两连边（全连接，有向）
        for i_idx in range(len(cp_ids)):
            src_id = cp_ids[i_idx]
            score_src = float(score_col[src_id])
            for j_idx in range(len(cp_ids)):
                if i_idx == j_idx:
                    continue
                dst_id = cp_ids[j_idx]
                score_dst = float(score_col[dst_id])

                row_col.append([src_id, dst_id])
                edge_weight.append([score_src, score_dst, float(lr_k)])
                lig_rec.append([lig_name, rec_name])
                total_edges += 1

    print('细胞对图：%d 个节点，%d 条有向边（多重有向图）' % (num_cp_nodes, total_edges))
    print('平均每节点度数：%.1f' % (total_edges / num_cp_nodes if num_cp_nodes > 0 else 0))

    # =================== 保存图文件 ===========================================
    output_path = args.data_to + args.data_name + '_cellpair_graph_scrna_adjacency_records'
    print('正在保存图文件至：%s' % output_path)
    with gzip.open(output_path, 'wb') as fp:
        pickle.dump([row_col, edge_weight, lig_rec,
                     num_cp_nodes, X_normalized, cp_id_to_pair], fp)

    # =================== 保存元数据 ===========================================
    meta_path = args.metadata_to + args.data_name + '_cellpair_graph_metadata.txt'
    with open(meta_path, 'w') as f:
        f.write('细胞类型对节点数（N）：%d\n' % num_cp_nodes)
        f.write('配受体对特征维度（M）：%d\n' % M)
        f.write('有向边总数：%d\n' % total_edges)
        f.write('\n细胞类型对节点列表：\n')
        for cp_id in sorted(cp_id_to_pair.keys()):
            typeA, typeB = cp_id_to_pair[cp_id]
            f.write('  节点 %d：%s → %s\n' % (cp_id, typeA, typeB))

    print('元数据保存至：%s' % meta_path)
    print('预处理完毕。')
    print('')
    print('下一步：运行 run_CellNEST_cellpair_graph_scrna.py 进行 GAT+DGI 训练。')
