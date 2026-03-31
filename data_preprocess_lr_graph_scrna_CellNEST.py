# ============================================================================
# data_preprocess_lr_graph_scrna_CellNEST.py
# Written by: Fatema Tuz Zohora (参照 CellNEST 框架扩展)
#
# 目标
# ----
# 对 scRNA-seq 数据，构建一个"配受体对图（LR-pair Graph）"：
#   · 节点  = 数据集中出现的每一个配受体对（如 CCL2-CCR2）
#   · 节点特征 = 该配受体对在所有细胞类型通讯对中的活跃度向量
#   · 边    = 两个配受体对在相似的细胞类型对中共同活跃（用余弦相似度量化）
#
# 设计思路（对应问题陈述）
# ========================
#
# 1. 节点定义与初始特征 X（N × M 矩阵）
#    · N = 活跃配受体对数量（节点数）
#    · M = 活跃细胞类型通讯对数量（特征维度）
#    · X[i, j] = 配受体对 i 在细胞类型对 j 中的平均通讯分数（0 表示不活跃）
#    · 构建完毕后进行 L2 行归一化，使不同量级的配受体可比。
#
# 2. 边定义与权重（余弦相似度 + KNN 稀疏化）
#    · 由于 X 已 L2 归一化，余弦相似度 = X @ X.T（矩阵内积）
#    · 使用 K 近邻（K-NN）建图，只保留相似度最高的 K 个邻居（默认 K=10）
#      或设定阈值（默认相似度 > 0.5 才连边），两者可同时使用。
#    · 边特征 = 余弦相似度值（标量，shape: [num_edges, 1]）
#    · 图为无向图（双向边）。
#
# 与 CellNEST 双分图（bipartite）设计的区别
# =========================================
#   · 双分图：同时保留配受体节点和细胞类型对节点，两类节点之间连边。
#     目的是学习"哪些 LR 通路出现在相同的细胞对语境中"（通路串扰）。
#   · 本脚本（LR 对图）：只有配受体节点，边直接编码两个 LR 对之间
#     的相似性，让 GAT + DGI 直接在配受体空间中学习共激活模块。
#
# 输出文件
# --------
#   · input_graph/<data_name>/<data_name>_lr_graph_scrna_adjacency_records
#     内容（pickle 列表）：
#       [row_col, edge_weight, lig_rec, num_lr_nodes, X_feature, lr_id_to_pair]
#       - row_col       : list of [src_lr_id, dst_lr_id]（有向边，双向各一）
#       - edge_weight   : list of [cosine_similarity]（边特征）
#       - lig_rec       : list of [ligand_name, receptor_name]（源节点对应）
#       - num_lr_nodes  : 节点数 N
#       - X_feature     : numpy ndarray (N, M)，已 L2 归一化的节点特征矩阵
#       - lr_id_to_pair : dict {lr_id -> (ligand, receptor)}
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
from sklearn.neighbors import NearestNeighbors
print('依赖包加载完毕')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'CellNEST scRNA-seq LR 对图预处理 —— '
            '以配受体对为节点、余弦相似度为边构建图（用于 GAT+DGI 训练）'
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
    parser.add_argument('--knn_k', type=int, default=10,
                        help='KNN 建图时每个节点保留的最近邻数量（默认 10）')
    parser.add_argument('--cosine_threshold', type=float, default=0.0,
                        help='余弦相似度阈值，只有相似度超过此值的边才被保留'
                             '（默认 0.0，即不过滤；建议设 0.3-0.5）')
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

    # =================== 量化归一化 ===========================================
    # 与 bipartite scrna 保持一致，使用 quantile normalization 消除测序深度差异。
    print('正在进行 quantile normalization...')
    temp = qnorm.quantile_normalize(
        np.transpose(sparse.csr_matrix.toarray(adata.X)))
    cell_vs_gene = np.transpose(temp)
    print('量化归一化完毕，表达矩阵维度：', cell_vs_gene.shape)
    gene_index = {gene: i for i, gene in enumerate(gene_ids)}

    # =================== 构建细胞类型索引 =====================================
    print('正在构建细胞类型索引...')
    cells_of_type = defaultdict(list)
    for idx, ct in enumerate(cell_type_array):
        cells_of_type[ct].append(idx)
    for ct in unique_cell_types:
        print('  %s：%d 个细胞' % (ct, len(cells_of_type[ct])))

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

    # 为每个配受体对分配 ID（节点索引）
    l_r_pair = dict()
    lr_id = 0
    for gene in list(ligand_dict_dataset.keys()):
        ligand_dict_dataset[gene] = list(set(ligand_dict_dataset[gene]))
        l_r_pair[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            l_r_pair[gene][receptor_gene] = lr_id
            lr_id += 1

    num_lr_nodes = lr_id
    print('数据集中配受体对节点数（N）：%d' % num_lr_nodes)

    # 构建 lr_id → (ligand, receptor) 反向映射
    lr_id_to_pair = {}
    for lig in l_r_pair:
        for rec in l_r_pair[lig]:
            lr_id_to_pair[l_r_pair[lig][rec]] = (lig, rec)

    # =================== 基因表达活跃百分位阈值 ================================
    print('正在计算各细胞基因表达百分位阈值...')
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

    # =================== 遍历所有跨细胞类型活跃通讯（全枚举策略） ==============
    # 与 bipartite scrna 预处理完全一致：
    #   ct_pair_lr_score_sum[(ct_pair_key, lr_id)] = 累计通讯分数
    #   ct_pair_lr_count[(ct_pair_key, lr_id)]     = 计数
    # 最终 mean_score = score_sum / count。
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

                    ct_pair_key = (type_i, type_j)
                    communication_score = score_i * cell_vs_gene[j][rec_col]
                    if communication_score <= 0:
                        continue

                    key = (ct_pair_key, relation_id)
                    ct_pair_lr_score_sum[key] += communication_score
                    ct_pair_lr_count[key] += 1
                    total_active += 1

        print('%d/%d 配体基因已处理' % (g_idx + 1, len(ligand_list)), end='\r')

    print('')
    print('共发现 %d 条活跃通讯记录（按细胞对×配受体对计数）' % total_active)

    # =================== 构建节点特征矩阵 X（N × M） ==========================
    #
    # 思路：
    #   · 行 = 配受体对（节点），共 N = num_lr_nodes 行
    #   · 列 = 活跃细胞类型通讯对，共 M 列
    #   · X[i, j] = 配受体对 i 在细胞类型对 j 中的平均通讯分数
    #
    # 这个矩阵的含义：
    #   · 如果两个配受体对在相似的细胞类型对中都有高活跃度（高通讯分数），
    #     则它们的特征向量余弦相似度高 → 在图中相连 → GAT 会将它们聚类到一起。
    #   · 这正是"通路串扰（Pathway Crosstalk）"的数学化表达。
    print('正在构建细胞类型对 ID 映射...')
    ct_pair_to_id = {}
    ct_pair_id_counter = 0
    active_ct_pairs = set(ct_pk for (ct_pk, _) in ct_pair_lr_score_sum.keys())
    for ct_pk in sorted(active_ct_pairs):
        if ct_pk not in ct_pair_to_id:
            ct_pair_to_id[ct_pk] = ct_pair_id_counter
            ct_pair_id_counter += 1

    M = ct_pair_id_counter  # 特征维度 = 活跃细胞类型对数
    print('活跃细胞类型通讯对数（M，即特征维度）：%d' % M)
    print('配受体对节点数（N）：%d' % num_lr_nodes)
    print('节点特征矩阵维度：N×M = %d×%d' % (num_lr_nodes, M))

    # 填充 X[i, j]
    X_raw = np.zeros((num_lr_nodes, M), dtype=np.float32)
    for (ct_pk, relation_id), score_sum in ct_pair_lr_score_sum.items():
        count = ct_pair_lr_count[(ct_pk, relation_id)]
        mean_score = score_sum / count
        j = ct_pair_to_id[ct_pk]
        X_raw[relation_id, j] = float(mean_score)

    # 只保留至少有一个非零条目的配受体对节点（过滤全零行）
    # 注意：lr_id 与索引是一一对应的，全零行表示该配受体对在当前数据中没有任何
    # 活跃的细胞类型对通讯，可以安全过滤。
    active_lr_mask = (X_raw.sum(axis=1) > 0)
    active_lr_ids_old = np.where(active_lr_mask)[0]
    active_lr_count = int(active_lr_mask.sum())
    print('活跃配受体对节点数（至少在一个细胞类型对中有通讯分数）：%d / %d'
          % (active_lr_count, num_lr_nodes))

    # 重新映射 ID（old_id → new_id）
    old_to_new_id = {old: new for new, old in enumerate(active_lr_ids_old)}
    X_active = X_raw[active_lr_mask]        # shape: (active_lr_count, M)
    num_nodes_final = active_lr_count       # 最终节点数 N'

    # 更新 lr_id_to_pair 为最终 ID 映射
    lr_id_to_pair_final = {}
    for old_id in active_lr_ids_old:
        lr_id_to_pair_final[old_to_new_id[old_id]] = lr_id_to_pair[old_id]

    # =================== L2 行归一化 ==========================================
    # 归一化后，余弦相似度 = 向量点积，可通过矩阵乘法高效计算。
    row_norms = np.linalg.norm(X_active, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X_normalized = (X_active / row_norms).astype(np.float32)
    print('L2 归一化完毕，节点特征矩阵维度：', X_normalized.shape)

    # =================== 余弦相似度计算 + KNN 建图 ============================
    #
    # 核心算法：
    #   Step 1：X_normalized @ X_normalized.T 得到 N'×N' 余弦相似度矩阵。
    #           由于 X 已 L2 归一化，点积即余弦相似度。
    #   Step 2：对每个节点，取相似度最高的 K 个邻居（KNN）。
    #   Step 3：可选：用 cosine_threshold 进一步过滤相似度低的边。
    #   Step 4：构建双向边列表（无向图用双向有向边表示）。
    #
    # 为什么用 KNN 而不是全连接图？
    #   · 全连接密集图的边数为 N'(N'-1)，计算量大，且会引入大量噪声（低相似度边）。
    #   · KNN 稀疏图保留最相关的邻居，更有利于 GAT 学习局部结构。
    print('正在计算余弦相似度矩阵并构建 KNN 图（K=%d，相似度阈值=%.2f）...'
          % (args.knn_k, args.cosine_threshold))

    # 使用 sklearn NearestNeighbors（cosine 距离 = 1 - 余弦相似度）
    # 注：NearestNeighbors 返回距离，需转换：cosine_sim = 1 - cosine_dist
    k_actual = min(args.knn_k + 1, num_nodes_final)  # +1 因为包含自身
    nbrs = NearestNeighbors(n_neighbors=k_actual, metric='cosine',
                            algorithm='brute', n_jobs=-1)
    nbrs.fit(X_normalized)
    distances, indices = nbrs.kneighbors(X_normalized)
    # distances[i, 0] 是节点 i 到自身的距离（= 0），跳过第一列
    # cosine_similarity = 1 - cosine_distance

    row_col = []
    edge_weight = []
    lig_rec = []

    for src in range(num_nodes_final):
        src_lig, src_rec = lr_id_to_pair_final[src]
        for col_idx in range(1, k_actual):    # 跳过第 0 列（自身）
            dst = int(indices[src, col_idx])
            cosine_dist = float(distances[src, col_idx])
            cosine_sim = 1.0 - cosine_dist    # 转换为相似度

            # 应用余弦相似度阈值过滤
            if cosine_sim < args.cosine_threshold:
                continue

            # 无向图：添加双向边
            # 方向 1：src → dst
            row_col.append([src, dst])
            edge_weight.append([cosine_sim])
            lig_rec.append([src_lig, src_rec])

            # 方向 2：dst → src
            dst_lig, dst_rec = lr_id_to_pair_final[dst]
            row_col.append([dst, src])
            edge_weight.append([cosine_sim])
            lig_rec.append([dst_lig, dst_rec])

    # 去重：同一对节点可能从两侧各被添加一次
    edge_set = {}
    row_col_dedup = []
    edge_weight_dedup = []
    lig_rec_dedup = []
    for k in range(len(row_col)):
        key = (row_col[k][0], row_col[k][1])
        if key not in edge_set:
            edge_set[key] = True
            row_col_dedup.append(row_col[k])
            edge_weight_dedup.append(edge_weight[k])
            lig_rec_dedup.append(lig_rec[k])

    print('LR 对图：节点数 N=%d，有向边总数（双向）=%d'
          % (num_nodes_final, len(row_col_dedup)))
    print('平均每个节点的邻居数：%.2f'
          % (len(row_col_dedup) / max(num_nodes_final, 1)))

    # =================== 保存图数据 ===========================================
    # 保存内容：
    #   [row_col, edge_weight, lig_rec, num_nodes_final, X_normalized, lr_id_to_pair_final]
    #
    # row_col          : list of [src_id, dst_id]
    # edge_weight      : list of [cosine_similarity]（边特征，维度 1）
    # lig_rec          : list of [ligand, receptor]（源节点对应的配受体名称）
    # num_nodes_final  : 最终活跃节点数 N'
    # X_normalized     : ndarray (N', M)，L2 归一化后的节点特征矩阵
    # lr_id_to_pair_final : dict {new_lr_id -> (ligand, receptor)}
    output_path = (args.data_to + args.data_name
                   + '_lr_graph_scrna_adjacency_records')
    with gzip.open(output_path, 'wb') as fp:
        pickle.dump([row_col_dedup, edge_weight_dedup, lig_rec_dedup,
                     num_nodes_final, X_normalized, lr_id_to_pair_final], fp)

    print('图数据已保存至：%s' % output_path)

    # =================== 保存元数据 ===========================================
    # 保存细胞 barcode（无物理坐标，用 0 占位）
    barcode_info = [[cell_barcode[i], 0.0, 0.0, 0]
                    for i in range(n_cells)]
    with gzip.open(args.metadata_to + args.data_name + '_barcode_info', 'wb') as fp:
        pickle.dump(barcode_info, fp)

    # 保存基因列表
    df_out = pd.DataFrame(gene_ids)
    df_out.to_csv(args.metadata_to + 'gene_ids_' + args.data_name + '.csv',
                  index=False, header=False)

    # 保存细胞类型对 ID 映射（供下游分析使用）
    ct_pair_id_df = pd.DataFrame(
        [(str(k[0]), str(k[1]), v) for k, v in sorted(ct_pair_to_id.items(),
                                                        key=lambda x: x[1])],
        columns=['sender_type', 'receiver_type', 'ct_pair_id']
    )
    ct_pair_id_df.to_csv(
        args.metadata_to + 'ct_pair_to_id_' + args.data_name + '.csv',
        index=False
    )

    # 保存配受体对 ID 映射（new_id → ligand, receptor）
    lr_pair_df = pd.DataFrame(
        [(v[0], v[1], k) for k, v in sorted(lr_id_to_pair_final.items())],
        columns=['ligand', 'receptor', 'lr_node_id']
    )
    lr_pair_df.to_csv(
        args.metadata_to + 'lr_pair_ids_' + args.data_name + '.csv',
        index=False
    )

    print('元数据已保存至：%s' % args.metadata_to)
    print('预处理完毕。')
    print('')
    print('下一步：运行以下命令开始训练 GAT 模型：')
    print('  python run_CellNEST_lr_graph_scrna.py \\')
    print('      --data_name %s \\' % args.data_name)
    print('      --model_name CellNEST_lr_%s \\' % args.data_name)
    print('      --run_id 0')
