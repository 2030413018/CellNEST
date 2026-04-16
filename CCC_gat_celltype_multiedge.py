# ============================================================================
# CCC_gat_celltype_multiedge.py
#
# 目标
# ----
# 在"细胞类型多重边图（Cell-Type Multigraph）"上进行 GAT + DGI 训练：
#
#   · 节点  = 细胞类型（每个 subCluster 一个节点）
#   · 节点特征 = 该细胞类型在 LR 基因集合上的平均表达向量（已 L2 归一化）
#   · 边    = 有向多重边；同一 (u, v) 对可有多条平行边，
#             每条对应一个活跃配受体对，边特征 = 1 维通讯强度标量：
#                 w = mean_expr_u(l) × mean_expr_v(r)
#   · 编码器 = 两层 GATv2（与 CellNEST 其他模块结构一致），edge_dim=1
#   · 训练目标 = DGI（Deep Graph Infomax）
#
# DGI 训练原理
# ============
#   DGI 是一种自监督学习方法，通过最大化节点嵌入与全图摘要之间的互信息
#   （Mutual Information）来驱使编码器学习图结构信息，无需任何标签。
#
#   正样本：原始图 G_s → GATv2 编码器 → 节点嵌入 z_pos（保留图结构）
#   负样本：打乱节点特征的图 → 编码器 → z_neg（图结构对应关系被破坏）
#   全图摘要：s = sigmoid(mean(z_pos))（全局通讯模式汇聚向量）
#   损失函数（JS 散度形式）：
#       L = -E[log D(z_pos, s)] - E[log(1 - D(z_neg, s))]
#   最小化 L ⟺ 最大化 I(z_pos; s) — I(z_neg; s)
#
# GAT + DGI 对细胞类型多重边图的意义
# =====================================
#   · 节点（细胞类型）通过注意力机制聚合其邻居的信息，注意力权重由 GATv2
#     从边特征（通讯强度）和节点特征（基因表达）中联合学习。
#   · 多重边设计：同一对细胞类型间的多条平行边分别对应不同 LR 对，
#     GATv2 独立处理每条边，赋予不同注意力权重，反映各 LR 通路的重要性差异。
#   · DGI 驱使嵌入编码：不仅是细胞类型自身的基因表达特征，
#     还有该类型在整张样本通讯网络中的"角色"（与哪些类型通讯、通过哪些通路）。
#   · 训练收敛后：
#       - 嵌入相近的细胞类型 → 在样本中扮演相似通讯角色 → 属于同一通讯模块
#       - 跨样本比较同一细胞类型的嵌入，可揭示样本间通讯网络的差异
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


def get_celltype_multigraph(training_data):
    """加载细胞类型多重边图，并构建 PyG Data 对象。

    参数
    ----
    training_data : str
        由 data_preprocess_celltype_multiedge_scrna_CellNEST.py 生成的图文件路径。
        文件内容（pickle 列表）：
            [row_col, edge_weight, num_nodes, X_feature, ct_id_to_name, lr_gene_list]

    返回
    ----
    data_loader : DataLoader
        封装单张细胞类型多重边图的 DataLoader。
    num_feature : int
        节点特征维度 d（= LR 基因集合大小）。
    ct_id_to_name : dict
        {节点 id -> cell_type_name}，供下游分析使用。
    lr_gene_list : list
        节点特征列对应的基因名称列表（长度 d）。
    """
    with gzip.open(training_data, 'rb') as f:
        payload = pickle.load(f)

    row_col        = payload[0]   # list of [src_ct_id, dst_ct_id]
    edge_weight    = payload[1]   # list of [w]（1 维通讯强度）
    num_nodes      = payload[2]   # |V|：细胞类型节点数
    X_feature      = payload[3]   # ndarray (|V|, d)，已 L2 归一化
    ct_id_to_name  = payload[4]   # {node_id -> cell_type_name}
    lr_gene_list   = payload[5]   # 基因名列表（长度 d）

    num_feature = X_feature.shape[1]  # d

    print('细胞类型多重边图：%d 个节点（细胞类型），特征维度 d=%d'
          % (num_nodes, num_feature))
    print('有向多重边总数：%d' % len(row_col))
    if len(row_col) > 0:
        # 统计平行边分布（用于诊断）
        from collections import Counter
        pair_cnt = Counter((rc[0], rc[1]) for rc in row_col)
        max_parallel = max(pair_cnt.values())
        print('最多平行边数（同一细胞类型对）：%d' % max_parallel)

    # -----------------------------------------------------------------------
    # 构建 PyG Data 对象
    # -----------------------------------------------------------------------
    if len(row_col) == 0:
        # 无边的退化图（训练时会发出警告）
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float)
    else:
        # edge_index: shape (2, num_edges)，dtype=long
        # 注意：多重图中同一 (src, dst) 对可多次出现
        edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T

        # edge_attr: shape (num_edges, 1)，通讯强度（配体均值 × 受体均值）
        edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    # x: shape (|V|, d)，节点特征矩阵（L2 归一化后的 LR 基因平均表达）
    x = torch.tensor(X_feature, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_loader = DataLoader([graph], batch_size=1)

    print('细胞类型多重边图加载完毕')
    return data_loader, num_feature, ct_id_to_name, lr_gene_list


class CellTypeMultiEdgeEncoder(nn.Module):
    """两层 GATv2 编码器，用于细胞类型多重边图。

    结构
    ----
    · 第一层 GATv2：in_channels → hidden_channels（edge_dim=1）
    · 第二层 GATv2：hidden_channels → hidden_channels（edge_dim=1）
    · PReLU 激活

    与 CellPairEncoder（CCC_gat_cellpair_graph.py）的区别
    -------------------------------------------------------
    · edge_dim=1：边特征只有通讯强度这一个标量（而 cellpair 版本是 3 维）。
    · add_self_loops=False：多重图结构，关闭自环以避免形状冲突
      （节点特征已包含该类型自身的表达信息，自环信息冗余）。

    注意力机制的作用
    ----------------
    GATv2 的注意力权重计算：
        α_{ij} ∝ exp(a^T · tanh(Θ·x_i + Θ·x_j + lin_edge(edge_attr_{ij})))
    因此，权重同时取决于：
      1. 发送方细胞类型 u 的基因表达特征
      2. 接收方细胞类型 v 的基因表达特征
      3. 该条边对应的配受体通讯强度 w
    模型可区分"通过强通讯信号连接的邻居"与"通过弱信号连接的邻居"的重要性差异。
    多重边（同一 u→v 的多条平行边）会产生多个独立的注意力权重，
    最终通过 sum 聚合（concat=False）累加贡献，体现多条 LR 通路的整体效应。
    """

    def __init__(self, in_channels, hidden_channels, heads, dropout):
        super(CellTypeMultiEdgeEncoder, self).__init__()
        print('CellTypeMultiEdgeEncoder: in_channels=%d, hidden_channels=%d, heads=%d'
              % (in_channels, hidden_channels, heads))

        # 第一层 GATv2：in_channels → hidden_channels
        # edge_dim=1：边特征 = 通讯强度（1 个标量）
        # add_self_loops=False：多重有向图，关闭自环
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels,
            edge_dim=1, heads=heads, concat=False,
            add_self_loops=False,
        )

        # 第二层 GATv2：hidden_channels → hidden_channels
        self.conv2 = GATv2Conv(
            hidden_channels, hidden_channels,
            edge_dim=1, heads=heads, concat=False,
            add_self_loops=False,
        )

        self.prelu = nn.PReLU(hidden_channels)

        # 存储注意力权重（训练后可分析：哪些细胞类型间的注意力最强）
        self.attention_scores_mine = None
        self.attention_scores_mine_unnormalized = None
        self.attention_scores_mine_l1 = None
        self.attention_scores_mine_unnormalized_l1 = None

    def forward(self, data):
        """前向传播。

        Parameters
        ----------
        data : torch_geometric.data.Data 或 CorruptedGraphData
            包含 x, edge_index, edge_attr 的图数据对象。

        Returns
        -------
        x : Tensor, shape (|V|, hidden_channels)
            所有细胞类型节点的嵌入向量。
            训练收敛后，嵌入相近的细胞类型在该样本中具有相似的通讯网络角色。
        """
        # ---------- 第一层 GATv2 ----------
        x, attention_scores_l1, attn_unnorm_l1 = self.conv1(
            data.x, data.edge_index,
            edge_attr=data.edge_attr,
            return_attention_weights=True,
        )
        self.attention_scores_mine_l1 = attention_scores_l1
        self.attention_scores_mine_unnormalized_l1 = attn_unnorm_l1

        # ---------- 第二层 GATv2 ----------
        x, attention_scores_l2, attn_unnorm_l2 = self.conv2(
            x, data.edge_index,
            edge_attr=data.edge_attr,
            return_attention_weights=True,
        )
        self.attention_scores_mine = attention_scores_l2
        self.attention_scores_mine_unnormalized = attn_unnorm_l2

        # PReLU 激活（可学习参数，允许负值，避免 ReLU 的梯度消失问题）
        x = self.prelu(x)
        return x


class CorruptedGraphData:
    """DGI 负样本容器：保存打乱节点特征后的图（边结构不变）。"""

    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data):
    """DGI 腐蚀函数：随机打乱节点特征行顺序，破坏节点-图结构对应关系。

    作用
    ----
    · 将节点特征矩阵 X 的行随机重排，保持图拓扑（边）不变。
    · 打乱后，节点 u 的特征不再对应其原本在图中的细胞类型，
      节点特征与通讯关系之间的对应关系被破坏。
    · 编码器对这种"损坏图"输出的嵌入 z_neg 应与全图摘要 s 互信息低。
    · 通过对比正负样本，迫使编码器学习图结构与特征之间的有意义对应。
    """
    x = data.x[torch.randperm(data.x.size(0))]
    return CorruptedGraphData(x, data.edge_index, data.edge_attr)


def train_CellNEST_celltype_multiedge(args, data_loader, in_channels):
    """在细胞类型多重边图上训练 DGI 模型。

    训练流程（每个 epoch）
    --------------------
    1. 正样本前向传播：原始图 → CellTypeMultiEdgeEncoder → z_pos (|V|, hidden)
    2. 负样本前向传播：腐蚀图（节点特征打乱）→ 编码器 → z_neg (|V|, hidden)
    3. 全图摘要：s = sigmoid(mean(z_pos, dim=0))，shape (hidden,)
    4. DGI 损失：L = -E[log D(z_pos,s)] - E[log(1-D(z_neg,s))]
    5. 反向传播 + Adam 更新
    6. 每 500 个 epoch 记录损失，损失降低时保存检查点并导出嵌入向量。

    参数
    ----
    args        : argparse.Namespace
        命令行参数（来自 run_CellNEST_celltype_multiedge_scrna.py）。
    data_loader : DataLoader
        封装细胞类型多重边图的 DataLoader。
    in_channels : int
        节点特征维度 d（= LR 基因集合大小）。

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
    #   · encoder    : CellTypeMultiEdgeEncoder（两层 GATv2，edge_dim=1）
    #   · summary    : s = sigmoid(mean(z_pos)) —— 全图通讯模式汇聚向量
    #   · corruption : 打乱节点特征生成负样本
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=CellTypeMultiEdgeEncoder(
            in_channels=in_channels,
            hidden_channels=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
        ),
        summary=lambda z, *a, **kw: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate)
    DGI_filename = (args.model_path + 'DGI_celltype_multiedge_'
                    + args.model_name + '.pth.tar')

    # -----------------------------------------------------------------------
    # 加载检查点（可选，继续上次训练）
    # -----------------------------------------------------------------------
    if args.load == 1:
        print('正在加载已保存的模型检查点：%s' % DGI_filename)
        checkpoint = torch.load(DGI_filename)
        DGI_model.load_state_dict(checkpoint['model_state_dict'])
        DGI_model.to(device)
        DGI_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        min_loss = checkpoint['loss']
        print('已加载检查点，上次最优损失 = %g，从 epoch %d 继续'
              % (min_loss, epoch_start))
    else:
        print('保存初始模型状态...')
        torch.save({
            'epoch': 0,
            'model_state_dict': DGI_model.state_dict(),
            'optimizer_state_dict': DGI_optimizer.state_dict(),
        }, args.model_path + 'DGI_celltype_multiedge_init_' + args.model_name + '.pth.tar')
        min_loss = 10000
        epoch_start = 0

    import datetime
    start_time = datetime.datetime.now()

    # -----------------------------------------------------------------------
    # 主训练循环
    # -----------------------------------------------------------------------
    for epoch in range(epoch_start, args.num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        DGI_all_loss = []

        for data in data_loader:
            data = data.to(device)

            # DGI 前向传播：
            #   pos_z   : 正样本嵌入 (|V|, hidden) ← 原始图（保留特征-结构对应）
            #   neg_z   : 负样本嵌入 (|V|, hidden) ← 腐蚀图（特征行打乱）
            #   summary : 全图摘要向量 (hidden,)
            pos_z, neg_z, summary = DGI_model(data=data)

            # DGI 损失（JS 散度形式的互信息估计）
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

                # 保存节点嵌入向量（shape: |V| × hidden）
                # · 每行对应一个细胞类型节点
                # · 嵌入相近的细胞类型 → 在该样本中扮演相似的通讯角色
                # · 跨样本比较时，可按细胞类型名称对齐嵌入向量
                X_embedding = pos_z.cpu().detach().numpy()
                embed_filename = (args.embedding_path + args.model_name
                                  + '_celltype_multiedge_Embed_X')
                with gzip.open(embed_filename, 'wb') as fp:
                    pickle.dump(X_embedding, fp)

                # 保存注意力分数
                # attention_scores_mine      : (edge_index, normalized_weights)
                # attention_scores_mine_l1   : 第一层注意力
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
                                 + '_celltype_multiedge_attention')
                with gzip.open(attn_filename, 'wb') as fp:
                    pickle.dump(X_attention_bundle, fp)

                # 保存损失曲线
                logfile = open(
                    args.model_path + 'DGI_celltype_multiedge_' + args.model_name
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
