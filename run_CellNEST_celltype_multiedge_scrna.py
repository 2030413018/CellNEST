# ============================================================================
# run_CellNEST_celltype_multiedge_scrna.py
#
# 用途
# ----
# 细胞类型多重边图（Cell-Type Multigraph）模型的训练入口脚本。
# 在"细胞类型多重边图"上运行 GATv2 + DGI，学习细胞类型节点的嵌入表示。
#
# 图结构回顾
# ----------
#   · 节点  = 细胞类型（按 adata.obs[cell_type_col] 的 unique 值划分）
#   · 边    = 有向多重边；同一 (u, v) 对对应多条平行边，每条对应一个活跃 LR 对
#   · 边特征 = 1 维标量：通讯强度 w = mean_expr_u(l) × mean_expr_v(r)
#   · 节点特征 = LR 基因集合上的平均表达向量（已 L2 归一化）
#
# 使用方法
# --------
#   # 第一步：预处理（每个 sampleID 单独运行一次）
#   python data_preprocess_celltype_multiedge_scrna_CellNEST.py \
#       --data_name  mydata_S1 \
#       --data_from  /abs/path/data.h5ad \
#       --sample_col sampleID \
#       --sample_id  S1 \
#       --cell_type_col subCluster
#
#   # 第二步：训练（每个 sampleID 训练一个模型；建议多 run_id 取集成结果）
#   python run_CellNEST_celltype_multiedge_scrna.py \
#       --data_name  mydata_S1 \
#       --model_name CellNEST_ct_multiedge_S1 \
#       --run_id     0
#
# 多样本批量训练示例（Shell）
# -------------------------
#   for sid in S1 S2 S3; do
#     python data_preprocess_celltype_multiedge_scrna_CellNEST.py \
#         --data_name mydata_${sid} \
#         --data_from /abs/path/data.h5ad \
#         --sample_col sampleID --sample_id ${sid} \
#         --cell_type_col subCluster
#
#     python run_CellNEST_celltype_multiedge_scrna.py \
#         --data_name mydata_${sid} \
#         --model_name CellNEST_ct_multiedge_${sid} \
#         --run_id 0
#   done
#
# 输出文件（每个 sampleID 独立目录）
# ------------------------------------
#   · embedding_data/<data_name>/<model_name>_r<run_id>_celltype_multiedge_Embed_X
#       节点嵌入向量（numpy array, shape: |V| × hidden）
#       · |V| = 该样本的细胞类型数
#       · 行顺序对应 ct_id_to_name（保存在预处理的元数据中）
#       · 用于下游聚类或跨样本比较（按细胞类型名称对齐后比较）
#
#   · embedding_data/<data_name>/<model_name>_r<run_id>_celltype_multiedge_attention
#       注意力分数（list，格式同 CellNEST 其他模块），用于分析哪些细胞类型对
#       之间的注意力权重最高（哪条 LR 通路最被模型关注）。
#
#   · model/<data_name>/DGI_celltype_multiedge_<model_name>_r<run_id>.pth.tar
#       最优模型检查点（损失最低的 epoch）。
# ============================================================================

import os
import numpy as np
from datetime import datetime
import random
import argparse
import torch
from torch_geometric.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'CellNEST scRNA-seq 细胞类型多重边图 GAT+DGI 训练 —— '
            '节点=细胞类型，边=活跃配受体对（多重有向边，边特征=通讯强度标量）'
        )
    )

    # =================== 必填参数 =============================================
    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称（与预处理步骤保持一致，建议含 sampleID）')
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型名称（用于输出文件命名）')
    parser.add_argument('--run_id', type=int, required=True,
                        help='运行编号（如 0, 1, 2, ...），建议至少运行 5 次取集成结果')

    # =================== 可选参数（已设默认值） ================================
    parser.add_argument('--num_epoch', type=int, default=60000,
                        help='训练迭代次数（默认 60000）')
    parser.add_argument('--model_path', type=str, default='model/',
                        help='模型权重保存路径（默认 "model/"）')
    parser.add_argument('--embedding_path', type=str, default='embedding_data/',
                        help='节点嵌入向量和注意力分数保存路径（默认 "embedding_data/"）')
    parser.add_argument('--hidden', type=int, default=512,
                        help='隐藏层（嵌入向量）维度（默认 512）')
    parser.add_argument('--training_data', type=str, default='input_graph/',
                        help='图文件所在路径（预处理输出路径，默认 "input_graph/"）')
    parser.add_argument('--heads', type=int, default=1,
                        help='GAT 注意力头数（默认 1）')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout 比率（默认 0）')
    parser.add_argument('--lr_rate', type=float, default=0.00001,
                        help='Adam 优化器学习率（默认 0.00001）')
    parser.add_argument('--manual_seed', type=str, default='no',
                        help='是否手动设置随机种子（"yes" 或 "no"，默认 "no"）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子值（仅在 --manual_seed=yes 时生效，默认 42）')
    parser.add_argument('--load', type=int, default=0,
                        help='设为 1 则从已保存的检查点继续训练（默认 0）')

    args = parser.parse_args()

    # =================== 路径拼接 =============================================
    args.training_data = (args.training_data + args.data_name + '/'
                          + args.data_name
                          + '_celltype_multiedge_adjacency_records')
    args.embedding_path = args.embedding_path + args.data_name + '/'
    args.model_path = args.model_path + args.data_name + '/'
    args.model_name = args.model_name + '_r' + str(args.run_id)

    # =================== 随机种子 =============================================
    if args.manual_seed == 'yes':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        print('已设置随机种子：%d' % args.seed)

    # =================== 创建输出目录 =========================================
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print('============================================================')
    print('参数摘要')
    print('  数据集：%s' % args.data_name)
    print('  训练数据：%s' % args.training_data)
    print('  模型名称：%s' % args.model_name)
    print('  嵌入维度：%d' % args.hidden)
    print('  注意力头数：%d' % args.heads)
    print('  学习率：%g' % args.lr_rate)
    print('  训练轮次：%d' % args.num_epoch)
    print('============================================================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备：%s' % device)

    from CCC_gat_celltype_multiedge import (get_celltype_multigraph,
                                             train_CellNEST_celltype_multiedge)

    # =================== 加载细胞类型多重边图 ==================================
    print('正在加载细胞类型多重边图...')
    data_loader, num_feature, ct_id_to_name, lr_gene_list = get_celltype_multigraph(
        args.training_data)
    num_nodes = len(ct_id_to_name)
    print('节点数（细胞类型数）：%d' % num_nodes)
    print('节点特征维度（LR 基因数 d）：%d' % num_feature)
    print('细胞类型节点列表：')
    for nid in sorted(ct_id_to_name.keys()):
        print('  %d: %s' % (nid, ct_id_to_name[nid]))

    # =================== 训练 GAT+DGI 模型 ====================================
    print('')
    print('开始训练 GAT+DGI 模型...')
    DGI_model = train_CellNEST_celltype_multiedge(
        args, data_loader=data_loader, in_channels=num_feature)

    print('')
    print('============================================================')
    print('训练完毕。输出文件位于：')
    print('  嵌入向量：%s%s_celltype_multiedge_Embed_X'
          % (args.embedding_path, args.model_name))
    print('  注意力分数：%s%s_celltype_multiedge_attention'
          % (args.embedding_path, args.model_name))
    print('  模型权重：%sDGI_celltype_multiedge_%s.pth.tar'
          % (args.model_path, args.model_name))
    print('')
    print('嵌入向量使用方法：')
    print('  import gzip, pickle')
    print('  with gzip.open("<embed_path>", "rb") as f:')
    print('      embed = pickle.load(f)   # shape: (%d, %d)' % (num_nodes, args.hidden))
    print('  # embed[i] 对应 ct_id_to_name[i] = "%s"'
          % (ct_id_to_name.get(0, 'cell_type_0')))
    print('')
    print('下一步建议：')
    print('  · 对 embed 做 K-Means 或层次聚类，发现具有相似通讯角色的细胞类型群落')
    print('  · 跨样本比较时，按 ct_id_to_name 对齐不同样本的嵌入，')
    print('    用余弦相似度或 UMAP 可视化通讯模式的样本间差异')
    print('  · 分析注意力分数，确定哪些细胞类型对之间的通讯最被模型关注')
    print('============================================================')
