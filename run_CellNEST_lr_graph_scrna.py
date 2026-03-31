# ============================================================================
# run_CellNEST_lr_graph_scrna.py
# Written by: Fatema Tuz Zohora (参照 CellNEST 框架扩展)
#
# 用途
# ----
# LR-pair Graph 模型的训练入口脚本。
# 在"配受体对图"上运行 GATv2 + DGI，学习配受体共激活模块的嵌入表示。
#
# 使用方法
# --------
#   # 第一步：预处理，构建 LR-pair Graph
#   python data_preprocess_lr_graph_scrna_CellNEST.py \
#       --data_name  my_scrna \
#       --data_from  path/to/my_data.h5ad \
#       --cell_type_col cell_type \
#       --knn_k 10 \
#       --cosine_threshold 0.3
#
#   # 第二步：训练 GAT+DGI 模型（建议运行多次取集成结果）
#   python run_CellNEST_lr_graph_scrna.py \
#       --data_name  my_scrna \
#       --model_name CellNEST_lr_my_dataset \
#       --run_id     0
#
# 前置条件
# --------
# 先运行 data_preprocess_lr_graph_scrna_CellNEST.py，
# 确保 input_graph/<data_name>/<data_name>_lr_graph_scrna_adjacency_records 已生成。
#
# 输出文件
# --------
# · embedding_data/<data_name>/<model_name>_r<run_id>_lr_graph_Embed_X
#       节点嵌入向量（numpy array, shape: N' × hidden）
#       N' = 活跃配受体对数量，hidden = --hidden 参数（默认 512）
#       用于下游聚类（K-Means、Louvain 等）以发现共信号模块。
#
# · embedding_data/<data_name>/<model_name>_r<run_id>_lr_graph_attention
#       注意力分数（list），用于分析哪些配受体对之间互相关注程度最高。
#
# · model/<data_name>/DGI_lr_graph_<model_name>_r<run_id>.pth.tar
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
            'CellNEST scRNA-seq LR-pair Graph GAT+DGI 训练 —— '
            '配受体共激活模块发现'
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
                        help='LR-pair Graph 文件所在路径（预处理输出路径）')
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
                          + args.data_name + '_lr_graph_scrna_adjacency_records')
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

    from CCC_gat_lr_graph import get_lr_graph, train_CellNEST_lr_graph

    # =================== 加载 LR-pair Graph ==================================
    print('正在加载 LR-pair Graph...')
    data_loader, num_feature, lr_id_to_pair = get_lr_graph(args.training_data)
    print('特征维度（活跃细胞类型对数 M）：%d' % num_feature)

    # =================== 训练 GAT+DGI 模型 ====================================
    print('开始训练 GAT+DGI 模型...')
    DGI_model = train_CellNEST_lr_graph(
        args, data_loader=data_loader, in_channels=num_feature)
    print('训练完毕。')
    print('')
    print('嵌入向量已保存至：%s%s_lr_graph_Embed_X'
          % (args.embedding_path, args.model_name))
    print('')
    print('下一步建议：')
    print('  · 加载 Embed_X（numpy array，shape N×%d）' % args.hidden)
    print('  · 使用 K-Means 或 Louvain 聚类，发现配受体共激活模块')
    print('  · 通过 lr_id_to_pair 映射，将每个簇的节点还原为配受体对名称')
