# Written By
# Fatema Tuz Zohora
#
# scRNA-seq 双分图 GAT 训练入口脚本。
#
# 使用方法示例
# ------------
#   # 第一步：预处理，构建双分图
#   python data_preprocess_bipartite_scrna_CellNEST.py \
#       --data_name  my_scrna \
#       --data_from  path/to/my_data.h5ad \
#       --cell_type_col cell_type
#
#   # 第二步：训练 GAT 模型（可重复运行多次，取 run_id 0,1,2,...）
#   python run_CellNEST_bipartite_scrna.py \
#       --data_name  my_scrna \
#       --model_name CellNEST_scrna_my_dataset \
#       --run_id     0
#
# 前置条件
# --------
# 先运行 data_preprocess_bipartite_scrna_CellNEST.py，
# 确保 input_graph/<data_name>/<data_name>_bipartite_scrna_adjacency_records 已生成。

import os
import numpy as np
from datetime import datetime
import random
import argparse
import torch
from torch_geometric.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CellNEST scRNA-seq 双分图 GAT 训练 —— 通路串扰检测')

    # =================== 必填参数 ===========================================
    parser.add_argument('--data_name', type=str,
                        help='数据集名称（与预处理步骤保持一致）')
    parser.add_argument('--model_name', type=str,
                        help='模型名称')
    parser.add_argument('--run_id', type=int,
                        help='运行编号（如 0, 1, 2, ...），建议至少运行 5 次取集成结果')

    # =================== 可选参数（已设默认值） ==============================
    parser.add_argument('--num_epoch', type=int, default=60000,
                        help='训练迭代次数（默认 60000）')
    parser.add_argument('--model_path', type=str, default='model/',
                        help='模型权重保存路径')
    parser.add_argument('--embedding_path', type=str, default='embedding_data/',
                        help='节点嵌入向量和注意力分数保存路径')
    parser.add_argument('--hidden', type=int, default=512,
                        help='隐藏层维度（节点嵌入向量维度，默认 512）')
    parser.add_argument('--training_data', type=str, default='input_graph/',
                        help='双分图文件所在路径（预处理输出路径）')
    parser.add_argument('--heads', type=int, default=1,
                        help='注意力头数（默认 1）')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout 比率（默认 0）')
    parser.add_argument('--lr_rate', type=float, default=0.00001,
                        help='学习率（默认 0.00001）')
    parser.add_argument('--manual_seed', type=str, default='no',
                        help='是否手动设置随机种子，设为 "yes" 并提供 --seed')
    parser.add_argument('--seed', type=int,
                        help='随机种子值（仅在 --manual_seed=yes 时生效）')
    parser.add_argument('--metadata_to', type=str, default='metadata/',
                        help='元数据保存路径')
    parser.add_argument('--load', type=int, default=0,
                        help='设为 1 则从已保存的检查点继续训练')
    parser.add_argument('--load_model_name', type=str, default='None',
                        help='要恢复的模型名称（仅在 --load=1 时生效）')

    args = parser.parse_args()

    # 构建双分图文件的完整路径
    args.training_data = (args.training_data + args.data_name + '/'
                          + args.data_name + '_bipartite_scrna_adjacency_records')

    args.embedding_path = args.embedding_path + args.data_name + '/'
    args.model_path = args.model_path + args.data_name + '/'
    args.model_name = args.model_name + '_r' + str(args.run_id)

    print('数据集：%s，注意力头数：%d，训练数据：%s，隐藏维度：%d'
          % (args.data_name, args.heads, args.training_data, args.hidden))

    if args.manual_seed == 'yes':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print('------------------------模型与训练参数--------------------------')
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备：%s' % device)

    from CCC_gat_bipartite import get_bipartite_graph, train_CellNEST_bipartite

    # 加载双分图（scRNA-seq 版本，细胞类型对节点）
    data_loader, num_feature = get_bipartite_graph(args.training_data)

    # 训练 GAT 模型
    print('开始训练 GAT 模型...')
    DGI_model = train_CellNEST_bipartite(args, data_loader=data_loader,
                                         in_channels=num_feature)
    print('训练完毕。')
