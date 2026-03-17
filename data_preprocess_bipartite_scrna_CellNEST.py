# Written By
# Fatema Tuz Zohora
#
# scRNA-seq 专用双分图构建脚本——用于通路串扰（Pathway Crosstalk）检测。
#
# ============================================================================
# 与空间转录组版本的核心区别
# ============================================================================
# 1. 仅支持 scRNA-seq 数据（.h5ad 格式），无需空间坐标。
# 2. 邻域关系由 PCA 表达空间的 KNN 确定（而非物理距离），自动禁用 juxtacrine 过滤。
# 3. Node Type 2（细胞方向节点）使用**细胞类型对**（如"巨噬细胞→T细胞"），
#    而非单个细胞对（细胞 i→细胞 j）。
#    这样做的好处：
#      · 大幅减少节点数量，提高计算效率；
#      · 结果更具可解释性（直接回答"哪些L-R通路在某类细胞间协同激活"）；
#      · 通过对同类型细胞对的通讯分数取均值，降低个体细胞噪音。
# ============================================================================
#
# 图结构设计
# ==========
#   节点类型 1 —— 信号节点（Signal nodes，索引 0 … num_lr_nodes-1）
#       每个节点代表数据集中出现的一个唯一配受体对（如 CCL2-CCR2）。
#
#   节点类型 2 —— 细胞类型方向节点（CellType-pair nodes，
#                  索引 num_lr_nodes … total_nodes-1）
#       每个节点代表一个唯一的有序细胞类型对（发送方细胞类型 → 接收方细胞类型）。
#       例如："巨噬细胞→T细胞"、"成纤维细胞→内皮细胞"。
#
#   连边规则
#       节点间只有**跨类型**连边，同类型节点之间不连线。
#       当配受体对 L 在某细胞类型对 (typeA → typeB) 之间有活跃通讯时，
#       L 对应的信号节点与该细胞类型对节点之间画一条边。
#       边特征：[平均距离权重, 平均L-R共表达分数, lr_pair_id]
#
# GAT 目标
# ========
#   找出哪些配受体通路是协同工作的（Pathway Crosstalk）。
#   训练后，嵌入向量相近的信号节点倾向于在相同的细胞类型间协同激活，
#   即属于同一"共信号模块"（如炎症模块、增殖模块）。

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
import gc
from sklearn.neighbors import NearestNeighbors
print('用户参数读取中...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CellNEST scRNA-seq 双分图预处理 —— 通路串扰检测')

    # =================== 必填参数 ===========================================
    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称')
    parser.add_argument('--data_from', type=str, required=True,
                        help='scRNA-seq 数据路径（.h5ad 格式）')

    # =================== 可选参数（已设默认值） ==============================
    parser.add_argument('--cell_type_col', type=str, default='cell_type',
                        help='adata.obs 中细胞类型注释列名，默认为 "cell_type"。'
                             '若数据集中列名不同，请在此指定。')
    parser.add_argument('--data_to', type=str, default='input_graph/',
                        help='双分图文件保存路径（供 GAT 训练使用）')
    parser.add_argument('--metadata_to', type=str, default='metadata/',
                        help='元数据保存路径')
    parser.add_argument('--filter_min_cell', type=int, default=1,
                        help='基因过滤：基因至少在多少个细胞中表达（默认 1）')
    parser.add_argument('--threshold_gene_exp', type=float, default=98,
                        help='基因表达活跃阈值百分位数（默认 98）。'
                             '高于该百分位数的基因被认为在该细胞中活跃表达。')
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='KNN 邻域近邻数（默认 50）。'
                             '自动裁剪为不超过细胞总数 - 1。'
                             '若指定 --k 则以该值为准（兼容旧版参数名）。')
    parser.add_argument('--n_pcs', type=int, default=50,
                        help='PCA 降维维度（默认 50）。'
                             '自动裁剪为不超过 min(细胞数-1, 基因数-1)。')
    parser.add_argument('--block_autocrine', type=int, default=0,
                        help='设为 1 则忽略自分泌信号（同一细胞类型自身发出并接收）')
    parser.add_argument('--database_path', type=str,
                        default='database/CellNEST_database.csv',
                        help='配受体数据库路径，默认为 CellNEST 内置数据库 '
                             '（CellChat + NicheNet 合并版）')
    args = parser.parse_args()

    def _clamp_positive_int(value, max_allowed, param_name):
        """将正整数参数裁剪到 [1, max_allowed] 范围，超出时打印警告。"""
        clamped = min(value, max_allowed)
        if clamped < value:
            print('警告：%s=%d 超过上限 %d，已自动调整为 %d'
                  % (param_name, value, max_allowed, clamped))
        return clamped

    # =================== 路径初始化 ========================================
    if args.data_to == 'input_graph/':
        args.data_to = args.data_to + args.data_name + '/'
    if not os.path.exists(args.data_to):
        os.makedirs(args.data_to)

    if args.metadata_to == 'metadata/':
        args.metadata_to = args.metadata_to + args.data_name + '/'
    if not os.path.exists(args.metadata_to):
        os.makedirs(args.metadata_to)

    # =================== 读取 scRNA-seq 数据 ================================
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

    # =================== 细胞类型注释 ======================================
    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(
            '在 adata.obs 中未找到细胞类型列 "%s"。\n'
            '请使用 --cell_type_col 指定正确的列名。\n'
            '当前可用列：%s' % (args.cell_type_col, list(adata.obs.columns)))

    cell_type_array = np.array(adata.obs[args.cell_type_col].astype(str))
    unique_cell_types = sorted(list(set(cell_type_array)))
    print('发现 %d 种细胞类型：%s' % (len(unique_cell_types), unique_cell_types))

    # =================== 量化归一化 =========================================
    print('正在进行 quantile normalization...')
    temp = qnorm.quantile_normalize(
        np.transpose(sparse.csr_matrix.toarray(adata.X)))
    cell_vs_gene = np.transpose(temp)
    print('量化归一化完毕，表达矩阵维度：', cell_vs_gene.shape)

    # =================== PCA + KNN 邻域构建 =================================
    print('正在计算 PCA 嵌入以构建 KNN 邻域...')
    adata_embed = adata.copy()
    sc.pp.normalize_total(adata_embed, target_sum=1e4)
    sc.pp.log1p(adata_embed)

    n_pcs = _clamp_positive_int(args.n_pcs, min(n_cells - 1, len(gene_ids) - 1), '--n_pcs')
    sc.tl.pca(adata_embed, n_comps=n_pcs)
    pca_coords = np.array(adata_embed.obsm['X_pca'], dtype=np.float64)
    del adata_embed
    gc.collect()
    print('PCA 嵌入计算完毕，维度：%d' % n_pcs)

    k_actual = _clamp_positive_int(args.n_neighbors, n_cells - 1, '--n_neighbors')
    print('正在构建 KNN 邻域（k=%d）...' % k_actual)
    nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm='kd_tree', n_jobs=-1)
    nbrs.fit(pca_coords)
    distances, indices = nbrs.kneighbors(pca_coords)
    print('KNN 邻域构建完毕')

    # 基于距离计算归一化权重（距离越近权重越大）
    print('正在计算邻域距离权重...')
    weightdict_i_to_j = defaultdict(dict)
    for cell_idx in range(n_cells):
        max_val = np.max(distances[cell_idx, :])
        min_val = np.min(distances[cell_idx, :])
        denom = max_val - min_val if (max_val - min_val) > 0 else 1.0
        for neigh_idx in range(indices.shape[1]):
            neigh_cell_idx = indices[cell_idx][neigh_idx]
            d = distances[cell_idx][neigh_idx]
            flipped = 1.0 - (d - min_val) / denom
            # i = neigh_cell_idx 发送信号给 j = cell_idx
            weightdict_i_to_j[neigh_cell_idx][cell_idx] = flipped

    # =================== 构建基因辅助信息 ====================================
    gene_info = {gene: '' for gene in gene_ids}
    gene_index = {gene: i for i, gene in enumerate(gene_ids)}

    # =================== 读取配受体数据库 ====================================
    print('正在读取配受体数据库...')
    df_lr = pd.read_csv(args.database_path, sep=",")
    print('配受体数据库读取完毕')

    ligand_dict_dataset = defaultdict(list)
    # scRNA-seq 模式强制禁用 juxtacrine 过滤（无物理接触概念）
    cell_cell_contact = dict()
    count_pair = 0
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
        count_pair += 1
        # scRNA-seq：不记录 cell_cell_contact，等价于 block_juxtacrine=1

    included_gene = [g for g in gene_info if gene_info[g] == 'included']
    print('数据集中配体数量：%d' % len(ligand_dict_dataset.keys()))
    print('数据集总基因数：%d，其中作为配体或受体的基因数：%d'
          % (len(gene_ids), len(included_gene)))

    # 为每个配受体对分配 ID（信号节点索引）
    l_r_pair = dict()
    lr_id = 0
    for gene in list(ligand_dict_dataset.keys()):
        ligand_dict_dataset[gene] = list(set(ligand_dict_dataset[gene]))
        l_r_pair[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            l_r_pair[gene][receptor_gene] = lr_id
            lr_id += 1

    num_lr_nodes = lr_id
    print('数据集中配受体对节点数（信号节点）：%d' % num_lr_nodes)

    # =================== 基因表达活跃阈值 ====================================
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

    # =================== 遍历所有活跃通讯 ====================================
    # 对每个 KNN 连接的细胞对 (i→j) 及每个活跃 L-R 对，
    # 以"细胞类型对 (typeA→typeB)"为单位汇总通讯分数。
    #
    # 数据结构：
    #   ct_pair_lr_score_sum[(ct_pair_key, lr_id)] = 总分
    #   ct_pair_lr_count[(ct_pair_key, lr_id)]     = 计数
    #   ct_pair_dist_sum[ct_pair_key]               = 距离权重之和
    #   ct_pair_dist_count[ct_pair_key]             = 计数
    # ========================================================================
    print('正在枚举所有活跃配受体通讯...')

    ct_pair_lr_score_sum = defaultdict(float)
    ct_pair_lr_count = defaultdict(int)
    ct_pair_dist_sum = defaultdict(float)
    ct_pair_dist_count = defaultdict(int)

    ligand_list = list(ligand_dict_dataset.keys())
    total_active = 0

    for g_idx, gene in enumerate(ligand_list):
        for i in weightdict_i_to_j:
            if cell_vs_gene[i][gene_index[gene]] < cell_percentile[i]:
                continue
            type_i = cell_type_array[i]
            for j in weightdict_i_to_j[i]:
                if args.block_autocrine == 1 and i == j:
                    continue
                type_j = cell_type_array[j]
                ct_pair_key = (type_i, type_j)
                dist_w = weightdict_i_to_j[i][j]

                for gene_rec in ligand_dict_dataset[gene]:
                    if cell_vs_gene[j][gene_index[gene_rec]] < cell_percentile[j]:
                        continue

                    communication_score = (cell_vs_gene[i][gene_index[gene]]
                                           * cell_vs_gene[j][gene_index[gene_rec]])
                    if communication_score <= 0:
                        continue

                    relation_id = l_r_pair[gene][gene_rec]
                    key = (ct_pair_key, relation_id)

                    ct_pair_lr_score_sum[key] += communication_score
                    ct_pair_lr_count[key] += 1
                    ct_pair_dist_sum[ct_pair_key] += dist_w
                    ct_pair_dist_count[ct_pair_key] += 1
                    total_active += 1

        print('%d/%d 配体基因已处理' % (g_idx + 1, len(ligand_list)), end='\r')

    print('')
    print('共发现 %d 条活跃通讯记录（按细胞对×配受体对计数）' % total_active)

    # =================== 构建双分图 ==========================================
    # -------------------------------------------------------------------------
    # 节点类型 1 —— 信号节点（配受体对节点）：
    #     索引范围：0 … num_lr_nodes-1
    #
    # 节点类型 2 —— 细胞类型方向节点：
    #     索引范围：num_lr_nodes … num_lr_nodes + num_ct_pair_nodes - 1
    #     每个节点代表一个唯一的（发送方细胞类型, 接收方细胞类型）有序对。
    #     ct_pair_to_id[(typeA, typeB)] → 局部 id（0起）
    #     全局节点 id = num_lr_nodes + 局部 id
    #
    # 连边规则：
    #     对每个 (细胞类型对, 配受体对) 的活跃通讯组合，
    #     在信号节点与细胞类型方向节点之间各添加一条双向边（无向图用双向表示）。
    #     边特征：[平均距离权重, 平均L-R分数, lr_pair_id]
    # -------------------------------------------------------------------------

    # Step 1：枚举所有细胞类型对并分配 ID
    ct_pair_to_id = {}
    ct_pair_id_counter = 0

    # 收集所有出现过活跃通讯的细胞类型对
    active_ct_pairs = set(ct_pk for (ct_pk, _) in ct_pair_lr_score_sum.keys())
    for ct_pk in sorted(active_ct_pairs):
        if ct_pk not in ct_pair_to_id:
            ct_pair_to_id[ct_pk] = ct_pair_id_counter
            ct_pair_id_counter += 1

    num_ct_pair_nodes = ct_pair_id_counter
    total_num_nodes = num_lr_nodes + num_ct_pair_nodes

    print('双分图：%d 个信号节点（L-R对） + %d 个细胞类型方向节点 = %d 个节点'
          % (num_lr_nodes, num_ct_pair_nodes, total_num_nodes))

    # Step 2：构建边列表
    row_col = []      # [源节点id, 目标节点id]
    edge_weight = []  # [平均距离权重, 平均L-R分数, lr_pair_id]
    lig_rec = []      # [配体名称, 受体名称]

    # 构建 lr_id → (ligand, receptor) 的反向映射
    lr_id_to_pair = {}
    for lig in l_r_pair:
        for rec in l_r_pair[lig]:
            lr_id_to_pair[l_r_pair[lig][rec]] = (lig, rec)

    for (ct_pk, relation_id), score_sum in ct_pair_lr_score_sum.items():
        count = ct_pair_lr_count[(ct_pk, relation_id)]
        mean_score = score_sum / count

        # 计算该细胞类型对的平均距离权重
        d_sum = ct_pair_dist_sum[ct_pk]
        d_cnt = ct_pair_dist_count[ct_pk]
        mean_dist = d_sum / d_cnt if d_cnt > 0 else 0.0

        lr_node_id = relation_id
        cp_node_id = num_lr_nodes + ct_pair_to_id[ct_pk]
        lig_name, rec_name = lr_id_to_pair[relation_id]

        edge_feat = [mean_dist, mean_score, float(relation_id)]

        # 信号节点 → 细胞类型方向节点
        row_col.append([lr_node_id, cp_node_id])
        edge_weight.append(edge_feat)
        lig_rec.append([lig_name, rec_name])

        # 细胞类型方向节点 → 信号节点（无向图用双向边表示）
        row_col.append([cp_node_id, lr_node_id])
        edge_weight.append(edge_feat)
        lig_rec.append([lig_name, rec_name])

    print('双分图中节点总数：%d（信号节点：%d，细胞类型方向节点：%d），'
          '有向边总数：%d'
          % (total_num_nodes, num_lr_nodes, num_ct_pair_nodes, len(row_col)))
    print('预处理完毕。')
    print('正在写入数据...')

    # =================== 保存双分图 ==========================================
    with gzip.open(args.data_to + args.data_name + '_bipartite_scrna_adjacency_records',
                   'wb') as fp:
        pickle.dump([row_col, edge_weight, lig_rec,
                     num_lr_nodes, num_ct_pair_nodes, ct_pair_to_id], fp)

    # =================== 保存元数据 ==========================================
    # 细胞 barcode 信息（scRNA-seq 无物理坐标，坐标列用 0 填充）
    barcode_info = [[cell_barcode[i], 0.0, 0.0, 0]
                    for i in range(n_cells)]
    with gzip.open(args.metadata_to + args.data_name + '_barcode_info', 'wb') as fp:
        pickle.dump(barcode_info, fp)

    # 基因列表
    df_out = pd.DataFrame(gene_ids)
    df_out.to_csv(args.metadata_to + 'gene_ids_' + args.data_name + '.csv',
                  index=False, header=False)
    df_out = pd.DataFrame(cell_barcode)
    df_out.to_csv(args.metadata_to + 'cell_barcode_' + args.data_name + '.csv',
                  index=False, header=False)

    # 配受体对 ID 映射（供下游分析使用）
    lr_pair_to_id = {}
    for lig in l_r_pair:
        for rec in l_r_pair[lig]:
            lr_pair_to_id[(lig, rec)] = l_r_pair[lig][rec]

    with gzip.open(args.metadata_to + args.data_name + '_lr_pair_to_id', 'wb') as fp:
        pickle.dump(lr_pair_to_id, fp)

    # 细胞类型对 ID 映射
    with gzip.open(args.metadata_to + args.data_name + '_ct_pair_to_id', 'wb') as fp:
        pickle.dump(ct_pair_to_id, fp)

    # 保存细胞类型注释（供后续可视化使用）
    df_ct = pd.DataFrame({'barcode': cell_barcode,
                          'cell_type': cell_type_array})
    df_ct.to_csv(args.metadata_to + 'cell_type_' + args.data_name + '.csv',
                 index=False)

    print('数据写入完毕')
    print('')
    print('═' * 60)
    print('双分图摘要（scRNA-seq 模式）')
    print('═' * 60)
    print('  信号节点（配受体对）数量 : %d' % num_lr_nodes)
    print('  细胞类型方向节点数量     : %d' % num_ct_pair_nodes)
    print('  活跃细胞类型对           : %d' % len(ct_pair_to_id))
    print('  有向边总数               : %d' % len(row_col))
    print('  细胞类型对列表:')
    for ct_pk, idx in sorted(ct_pair_to_id.items(), key=lambda x: x[1]):
        print('    节点%d: %s → %s' % (num_lr_nodes + idx, ct_pk[0], ct_pk[1]))
    print('═' * 60)
