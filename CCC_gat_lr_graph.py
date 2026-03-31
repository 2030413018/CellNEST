# ============================================================================
# CCC_gat_lr_graph.py
# Written by: Fatema Tuz Zohora (参照 CellNEST 框架扩展)
#
# 目标
# ----
# 在"配受体对图（LR-pair Graph）"上进行 GAT + DGI 训练：
#   · 节点 = 配受体对，特征 = 在各细胞类型通讯对中的活跃度向量
#   · 边   = 余弦相似度 KNN 图（由 data_preprocess_lr_graph_scrna_CellNEST.py 生成）
#   · 编码器 = 两层 GATv2（与 CellNEST 双分图版本结构相同）
#   · 训练目标 = DGI（Deep Graph Infomax）：
#       最大化节点嵌入与全图摘要向量之间的互信息（Mutual Information）
#
# DGI 训练原理
# ============
#   DGI 不需要标签，是一种自监督学习方法。
#   正样本：原始图 → 编码器 → 节点嵌入 z_pos（保留图结构信息）
#   负样本：打乱节点特征的图 → 编码器 → z_neg（破坏图结构信息）
#   全图摘要：s = sigmoid(mean(z_pos))（全局信息汇聚）
#   损失函数：最大化 (z_pos, s) 的互信息，最小化 (z_neg, s) 的互信息
#   → 训练后，嵌入 z_pos 同时包含节点自身特征和图邻域结构信息。
#
# GAT + DGI 对本任务的意义
# ========================
#   · 配受体对节点通过注意力机制聚合邻居（相似 LR 对）的信息。
#   · DGI 驱使嵌入向量捕获"哪些 LR 对共同活跃"的全局模式。
#   · 训练收敛后：
#       - 嵌入距离近的 LR 对 → 在相似细胞类型对中协同激活 → 属于同一通路模块
#       - 可通过聚类（K-Means、Louvain 等）发现共信号模块（炎症/增殖/迁移等）
# ============================================================================

from scipy import sparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.data import Data, DataLoader
import gzip

from GATv2Conv_CellNEST import GATv2Conv


def get_lr_graph(training_data):
    """加载配受体对图（LR-pair Graph），并构建 PyG Data 对象。

    参数
    ----
    training_data : str
        由 data_preprocess_lr_graph_scrna_CellNEST.py 生成的图文件路径。
        文件内容（pickle 列表）：
            [row_col, edge_weight, lig_rec, num_nodes, X_feature, lr_id_to_pair]

    返回
    ----
    data_loader : DataLoader
        封装单张 LR-pair Graph 的 DataLoader。
    num_feature : int
        节点特征维度（= 活跃细胞类型对数 M）。
    lr_id_to_pair : dict
        {节点 id -> (配体名称, 受体名称)}，供下游分析使用。
    """
    with gzip.open(training_data, 'rb') as f:
        payload = pickle.load(f)

    row_col      = payload[0]   # list of [src_id, dst_id]
    edge_weight  = payload[1]   # list of [cosine_similarity]
    lig_rec      = payload[2]   # list of [ligand, receptor]
    num_nodes    = payload[3]   # N'：最终活跃节点数
    X_feature    = payload[4]   # ndarray (N', M)，已 L2 归一化
    lr_id_to_pair = payload[5]  # {new_lr_id -> (ligand, receptor)}

    num_feature = X_feature.shape[1]  # M：特征维度

    print('LR-pair Graph：%d 个节点（配受体对），特征维度 M=%d'
          % (num_nodes, num_feature))
    print('有向边总数：%d' % len(row_col))

    # -----------------------------------------------------------------------
    # 构建 PyG Data 对象
    # -----------------------------------------------------------------------
    # edge_index: shape (2, num_edges)，dtype=long
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T

    # edge_attr: shape (num_edges, 1)，余弦相似度作为边特征
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    # x: shape (N', M)，节点特征矩阵
    x = torch.tensor(X_feature, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_loader = DataLoader([graph], batch_size=1)

    print('LR-pair Graph 加载完毕')
    return data_loader, num_feature, lr_id_to_pair


class LRGraphEncoder(nn.Module):
    """两层 GATv2 编码器，用于 LR-pair Graph。

    结构
    ----
    · 第一层 GATv2：in_channels → hidden_channels（利用边特征 edge_dim=1）
    · 第二层 GATv2：hidden_channels → hidden_channels
    · PReLU 激活

    与双分图版本（BipartiteEncoder）的区别
    ------------------------------------
    · add_self_loops=True（同类型节点图可以有自环，有助于节点保留自身信息）
    · edge_dim=1（边特征只有余弦相似度一个标量，而双分图版本是 3 维）

    注意力机制的作用
    ----------------
    在聚合邻居信息时，注意力权重会倾向于更高余弦相似度的邻居。
    但注意力不仅依赖边特征，还依赖两端节点的特征，因此能学习
    更精细的"哪些相似 LR 对更应该互相影响"的关系。
    """

    def __init__(self, in_channels, hidden_channels, heads, dropout):
        super(LRGraphEncoder, self).__init__()
        print('LRGraphEncoder: in_channels=%d, hidden_channels=%d, heads=%d'
              % (in_channels, hidden_channels, heads))

        # 第一层 GATv2：in_channels → hidden_channels
        # add_self_loops=True：允许节点给自身传递信息（unipartite 图标准做法）
        # edge_dim=1：边特征维度 = 余弦相似度（1个标量）
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels,
            edge_dim=1, heads=heads, concat=False,
            add_self_loops=True,
        )

        # 第二层 GATv2：hidden_channels → hidden_channels
        self.conv2 = GATv2Conv(
            hidden_channels, hidden_channels,
            edge_dim=1, heads=heads, concat=False,
            add_self_loops=True,
        )

        self.prelu = nn.PReLU(hidden_channels)

        # 存储注意力权重，供训练后分析使用
        self.attention_scores_mine = None
        self.attention_scores_mine_unnormalized = None
        self.attention_scores_mine_l1 = None
        self.attention_scores_mine_unnormalized_l1 = None

    def forward(self, data):
        """前向传播。

        Parameters
        ----------
        data : torch_geometric.data.Data
            包含 x, edge_index, edge_attr 的图数据对象。

        Returns
        -------
        x : Tensor, shape (N', hidden_channels)
            所有节点的嵌入向量（训练后用于聚类分析）。
        """
        # ---------- 第一层 GATv2 ----------
        x, attention_scores_l1, attn_unnorm_l1 = self.conv1(
            data.x, data.edge_index,
            edge_attr=data.edge_attr,
            return_attention_weights=True,
        )
        # 保存第一层注意力分数（格式与 BipartiteEncoder 保持一致）
        self.attention_scores_mine_l1 = attention_scores_l1
        self.attention_scores_mine_unnormalized_l1 = attn_unnorm_l1

        # ---------- 第二层 GATv2 ----------
        x, attention_scores_l2, attn_unnorm_l2 = self.conv2(
            x, data.edge_index,
            edge_attr=data.edge_attr,
            return_attention_weights=True,
        )
        # 保存第二层注意力分数
        self.attention_scores_mine = attention_scores_l2
        self.attention_scores_mine_unnormalized = attn_unnorm_l2

        # PReLU 激活（可学习的参数化 ReLU，允许负值通过，避免梯度消失）
        x = self.prelu(x)
        return x


class CorruptedGraphData:
    """DGI 负样本容器：打乱节点特征的图（边结构不变）。"""

    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data):
    """DGI 腐蚀函数：随机打乱节点特征行顺序。

    作用
    ----
    · 打乱 x 的行顺序，破坏节点特征与图结构之间的对应关系。
    · 边结构不变，但节点特征不再与其邻居对齐。
    · 编码器对这种"损坏图"的嵌入 z_neg 应与全图摘要 s 互信息低。
    · 通过对比 z_pos 和 z_neg，迫使编码器学习图结构中的有意义信息。
    """
    x = data.x[torch.randperm(data.x.size(0))]
    return CorruptedGraphData(x, data.edge_index, data.edge_attr)


def train_CellNEST_lr_graph(args, data_loader, in_channels):
    """在 LR-pair Graph 上训练 DGI 模型。

    训练流程
    --------
    1. 正样本前向传播：原始图 → LRGraphEncoder → z_pos
    2. 负样本前向传播：打乱节点特征的图 → LRGraphEncoder → z_neg
    3. 全图摘要：s = sigmoid(mean(z_pos))
    4. DGI 损失：最大化 (z_pos, s) 互信息，最小化 (z_neg, s) 互信息
    5. 每 500 个 epoch 打印损失，并在损失降低时保存检查点。

    参数
    ----
    args        : argparse.Namespace
        命令行参数（来自 run_CellNEST_lr_graph_scrna.py）。
    data_loader : DataLoader
        封装 LR-pair Graph 的 DataLoader。
    in_channels : int
        节点特征维度 M（等于活跃细胞类型对数）。

    返回
    ----
    DGI_model : DeepGraphInfomax
        训练完毕并加载了最优检查点的模型。
    """
    loss_curve = np.zeros((args.num_epoch // 500 + 1))
    loss_curve_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备：%s' % device)

    # -----------------------------------------------------------------------
    # 构建 DGI 模型
    # -----------------------------------------------------------------------
    # DeepGraphInfomax 三个核心组件：
    #   · encoder   : LRGraphEncoder（两层 GATv2）
    #   · summary   : lambda z -> sigmoid(mean(z)) —— 全图摘要函数
    #                  将所有节点嵌入取平均再经 sigmoid，得到图级别的全局向量
    #   · corruption: 腐蚀函数，生成负样本
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=LRGraphEncoder(
            in_channels=in_channels,
            hidden_channels=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
        ),
        summary=lambda z, *a, **kw: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate)
    DGI_filename = (args.model_path + 'DGI_lr_graph_'
                    + args.model_name + '.pth.tar')

    # -----------------------------------------------------------------------
    # 加载检查点（可选，继续训练）
    # -----------------------------------------------------------------------
    if args.load == 1:
        print('正在加载已保存的模型检查点...')
        checkpoint = torch.load(DGI_filename)
        DGI_model.load_state_dict(checkpoint['model_state_dict'])
        DGI_model.to(device)
        DGI_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        min_loss = checkpoint['loss']
        print('已加载检查点，上次最优损失 = %g' % min_loss)
    else:
        print('保存初始模型状态...')
        torch.save({
            'epoch': 0,
            'model_state_dict': DGI_model.state_dict(),
            'optimizer_state_dict': DGI_optimizer.state_dict(),
        }, args.model_path + 'DGI_lr_graph_init_' + args.model_name + '.pth.tar')
        min_loss = 10000
        epoch_start = 0

    import datetime
    start_time = datetime.datetime.now()

    # -----------------------------------------------------------------------
    # 训练循环
    # -----------------------------------------------------------------------
    for epoch in range(epoch_start, args.num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        DGI_all_loss = []

        for data in data_loader:
            data = data.to(device)

            # DGI 前向传播：
            #   pos_z: 正样本节点嵌入 (N', hidden)
            #   neg_z: 负样本节点嵌入 (N', hidden)（打乱特征后的图）
            #   summary: 全图摘要向量 (hidden,)
            pos_z, neg_z, summary = DGI_model(data=data)

            # DGI 损失（Jensen-Shannon 散度形式的互信息估计）
            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            DGI_loss.backward()
            DGI_all_loss.append(DGI_loss.item())
            DGI_optimizer.step()

        # 每 500 个 epoch 记录损失，并在损失降低时保存检查点
        if (epoch % 500) == 0:
            cur_loss = np.mean(DGI_all_loss)
            print('Epoch: {:05d}, Loss: {:.4f}'.format(epoch + 1, cur_loss))
            loss_curve[loss_curve_counter] = cur_loss
            loss_curve_counter += 1

            if cur_loss < min_loss:
                min_loss = cur_loss

                # 保存模型检查点
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': DGI_model.state_dict(),
                    'optimizer_state_dict': DGI_optimizer.state_dict(),
                    'loss': min_loss,
                }, DGI_filename)

                # 保存节点嵌入向量（shape: N' × hidden）
                # 这是最终用于下游聚类分析的输出
                X_embedding = pos_z.cpu().detach().numpy()
                embed_filename = (args.embedding_path + args.model_name
                                  + '_lr_graph_Embed_X')
                with gzip.open(embed_filename, 'wb') as fp:
                    pickle.dump(X_embedding, fp)

                # 保存注意力分数（用于分析哪些 LR 对之间互相关注程度最高）
                X_attention_index = (
                    DGI_model.encoder.attention_scores_mine[0]
                    .cpu().detach().numpy()
                )
                X_attention_score_normalized_l1 = (
                    DGI_model.encoder.attention_scores_mine_l1[1]
                    .cpu().detach().numpy()
                )
                X_attention_score_unnormalized_l1 = (
                    DGI_model.encoder.attention_scores_mine_unnormalized_l1
                    .cpu().detach().numpy()
                )
                X_attention_score_normalized = (
                    DGI_model.encoder.attention_scores_mine[1]
                    .cpu().detach().numpy()
                )
                X_attention_score_unnormalized = (
                    DGI_model.encoder.attention_scores_mine_unnormalized
                    .cpu().detach().numpy()
                )

                X_attention_bundle = [
                    X_attention_index,
                    X_attention_score_normalized_l1,
                    X_attention_score_unnormalized,
                    X_attention_score_unnormalized_l1,
                    X_attention_score_normalized,
                ]
                attn_filename = (args.embedding_path + args.model_name
                                 + '_lr_graph_attention')
                with gzip.open(attn_filename, 'wb') as fp:
                    pickle.dump(X_attention_bundle, fp)

                # 保存损失曲线
                logfile = open(
                    args.model_path + 'DGI_lr_graph_' + args.model_name
                    + '_loss_curve.csv', 'wb')
                np.savetxt(logfile, loss_curve, delimiter=',')
                logfile.close()

    end_time = datetime.datetime.now()
    print('训练总耗时（秒）：%d' % (end_time - start_time).seconds)

    # 加载最优检查点
    checkpoint = torch.load(DGI_filename)
    DGI_model.load_state_dict(checkpoint['model_state_dict'])
    DGI_model.to(device)
    DGI_model.eval()

    # 验证最终损失
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            pos_z, neg_z, summary = DGI_model(data=data)
    DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
    print('最终最优模型损失：%g' % DGI_loss.item())

    return DGI_model
