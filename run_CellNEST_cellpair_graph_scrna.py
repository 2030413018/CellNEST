# ============================================================================
# run_CellNEST_cellpair_graph_scrna.py
# Written by: Fatema Tuz Zohora（参照 CellNEST 框架）
#
# 用途
# ----
# 细胞对图（Cell-Pair Graph）模型的训练入口脚本。
# 在"细胞对图"上运行 GATv2 + DGI，学习细胞类型通讯对的嵌入表示。
#
# 使用方法
# --------
#   # 第一步：预处理，构建细胞对图
#   python data_preprocess_cellpair_graph_scrna_CellNEST.py \
#       --data_name  my_scrna \
#       --data_from  path/to/my_data.h5ad \
#       --cell_type_col cell_type
#
#   # 第二步：训练 GAT+DGI 模型（建议运行多次取集成结果）
#   python run_CellNEST_cellpair_graph_scrna.py \
#       --data_name  my_scrna \
#       --model_name CellNEST_cellpair_my_dataset \
#       --run_id     0
#
# 输出文件
# --------
#   embedding_data/<data_name>/<model_name>_r<run_id>_cellpair_Embed_X
#       节点嵌入向量（numpy array，shape: N × hidden）
#       N = 活跃细胞类型对节点数，hidden = --hidden 参数（默认 512）
#       用于下游聚类（K-Means、Louvain 等）发现具有相似通讯模式的细胞类型对群落。
#
#   embedding_data/<data_name>/<model_name>_r<run_id>_cellpair_attention
#       注意力分数（list），用于分析哪些配受体对边对节点嵌入的贡献最大。
#
#   model/<data_name>/DGI_cellpair_<model_name>_r<run_id>.pth.tar
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
            'CellNEST scRNA-seq 细胞对图 GAT+DGI 训练 —— '
            '细胞类型通讯对嵌入学习'
        )
    )

    # =================== 必填参数 =============================================
    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称（与预处理步骤保持一致）')
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型名称（用于输出文件命名）')
    parser.add_argument('--run_id', type=int, required=True,
                        help='运行编号（如 0, 1, 2, ...），建议至少运行 5 次取集成结果')

    # =================== 可选参数（已设默认值） ================================
    parser.add_argument('--num_epoch', type=int, default=60000,
                        help='训练迭代次数（默认 60000）')
    parser.add_argument('--model_path', type=str, default='model/',
                        help='模型权重保存路径')
    parser.add_argument('--embedding_path', type=str, default='embedding_data/',
                        help='节点嵌入向量和注意力分数保存路径')
    parser.add_argument('--hidden', type=int, default=512,
                        help='隐藏层（嵌入向量）维度，默认 512')
    parser.add_argument('--training_data', type=str, default='input_graph/',
                        help='细胞对图文件所在路径（预处理输出路径）')
    parser.add_argument('--heads', type=int, default=1,
                        help='注意力头数，默认 1')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout 比率，默认 0')
    parser.add_argument('--lr_rate', type=float, default=0.00001,
                        help='Adam 优化器学习率，默认 0.00001')
    parser.add_argument('--manual_seed', type=str, default='no',
                        help='是否手动设置随机种子（"yes" 或 "no"）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子值（仅在 --manual_seed=yes 时生效）')
    parser.add_argument('--load', type=int, default=0,
                        help='设为 1 则从已保存的检查点继续训练')

    args = parser.parse_args()

    # =================== 路径拼接 =============================================
    args.training_data = (args.training_data + args.data_name + '/'
                          + args.data_name
                          + '_cellpair_graph_scrna_adjacency_records')
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

    print('========================= 参数摘要 ==========================')
    print('数据集：%s' % args.data_name)
    print('训练数据：%s' % args.training_data)
    print('模型名称：%s' % args.model_name)
    print('嵌入维度：%d' % args.hidden)
    print('注意力头数：%d' % args.heads)
    print('学习率：%g' % args.lr_rate)
    print('训练轮次：%d' % args.num_epoch)
    print('=============================================================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备：%s' % device)

    from CCC_gat_cellpair_graph import get_cellpair_graph, train_CellNEST_cellpair

    # =================== 加载细胞对图 =========================================
    print('正在加载细胞对图...')
    data_loader, num_feature, cp_id_to_pair = get_cellpair_graph(
        args.training_data)
    print('节点特征维度（配受体对总数 M）：%d' % num_feature)
    print('节点数（活跃细胞类型对 N）：%d' % len(cp_id_to_pair))

    # =================== 训练 GAT+DGI 模型 ====================================
    print('开始训练 GAT+DGI 模型...')
    DGI_model = train_CellNEST_cellpair(
        args, data_loader=data_loader, in_channels=num_feature)
    print('训练完毕。')
    print('')
    print('嵌入向量已保存至：%s%s_cellpair_Embed_X'
          % (args.embedding_path, args.model_name))
    print('')
    print('下一步建议：')
    print('  · 加载 Embed_X（numpy array，shape N×%d）' % args.hidden)
    print('  · 使用 K-Means 或 Louvain 聚类，发现具有相似通讯模式的细胞类型对群落')
    print('  · 通过 cp_id_to_pair 映射，将每个簇的节点还原为细胞类型对名称')
    print('  · 分析注意力分数，确定哪些配受体对边的权重最高（最关键的LR通路）')
