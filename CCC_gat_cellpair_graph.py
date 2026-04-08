# ============================================================================
# CCC_gat_cellpair_graph.py
# Written by: Fatema Tuz Zohora（参照 CellNEST 框架）
#
# 目标
# ----
# 在"细胞对图（Cell-Pair Graph）"上进行 GAT + DGI 训练：
#   · 节点 = 有序细胞类型对（typeA → typeB）
#   · 节点特征 = 该细胞类型对在所有活跃配受体对上的通讯分数向量（length = M）
#   · 边   = 配受体对（每条边直接对应一个具体的配受体对分子）
#   · 边特征 = [源节点LR通讯分数, 目标节点LR通讯分数, LR对ID]（shape: edge × 3）
#   · 编码器 = 两层 GATv2（与 CellNEST 其他模块结构一致）
#   · 训练目标 = DGI（Deep Graph Infomax）
#
# DGI 训练原理
# ============
#   DGI 是一种自监督（无标签）学习方法，通过最大化节点嵌入与全图摘要之间的
#   互信息（Mutual Information）来驱使编码器捕获图的结构信息。
#
#   · 正样本：原始图 → GATv2 编码器 → 节点嵌入 z_pos（包含图结构信息）
#   · 负样本：打乱节点特征的图 → GATv2 编码器 → z_neg（图结构信息被破坏）
#   · 全图摘要：s = sigmoid(mean(z_pos))（将所有节点嵌入取均值再 sigmoid）
#   · 损失函数（JS 散度形式）：
#       L = -[log D(z_pos, s) + log(1 - D(z_neg, s))]
#     其中 D 是一个判别器，用于区分来自原始图与腐蚀图的节点嵌入
#
# GAT + DGI 对本任务（细胞对图）的意义
# =====================================
#   · 每个节点（细胞类型对）通过注意力机制聚合其邻居（通过同一LR对通讯的其他
#     细胞类型对）的信息，注意力权重由 GATv2 从边特征+节点特征中动态学习。
#   · DGI 驱使嵌入向量不仅保留节点自身的通讯分数，还编码其在图结构中的
#     角色（即与哪些其他细胞类型对共用哪些配受体通路）。
#   · 训练收敛后：
#       - 嵌入相近的细胞类型对 → 使用相似的配受体通路 → 属于同一通讯模块
#       - 可通过聚类（K-Means、Louvain）发现具有协同通讯模式的细胞类型对群落
#
# 与 CellNEST 双分图版本（CCC_gat_bipartite.py）的区别
# ======================================================
#   · 双分图：LR对节点 ↔ 细胞类型对节点，学习"哪些LR对在相同细胞背景中共激活"
#   · 本模块（细胞对图）：细胞类型对节点 ↔ 细胞类型对节点（通过LR对边连接），
#     学习"哪些细胞类型通讯对具有相似的分子通讯模式"
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


def get_cellpair_graph(training_data):
    """加载细胞对图（Cell-Pair Graph），并构建 PyG Data 对象。

    参数
    ----
    training_data : str
        由 data_preprocess_cellpair_graph_scrna_CellNEST.py 生成的图文件路径。
        文件内容（pickle 列表）：
            [row_col, edge_weight, lig_rec, num_nodes, X_feature, cp_id_to_pair]

    返回
    ----
    data_loader : DataLoader
        封装单张细胞对图的 DataLoader。
    num_feature : int
        节点特征维度（= 活跃配受体对总数 M）。
    cp_id_to_pair : dict
        {节点 id -> (typeA, typeB)}，供下游分析使用。
    """
    with gzip.open(training_data, 'rb') as f:
        payload = pickle.load(f)

    row_col      = payload[0]   # list of [src_cp_id, dst_cp_id]
    edge_weight  = payload[1]   # list of [score_src, score_dst, lr_pair_id]
    lig_rec      = payload[2]   # list of [ligand, receptor]（该边对应的配受体对）
    num_nodes    = payload[3]   # N：细胞类型对节点数
    X_feature    = payload[4]   # ndarray (N, M)，已 L2 归一化
    cp_id_to_pair = payload[5]  # {cp_id -> (typeA, typeB)}

    num_feature = X_feature.shape[1]  # M：节点特征维度 = 配受体对总数

    print('细胞对图：%d 个节点（细胞类型对），特征维度 M=%d'
          % (num_nodes, num_feature))
    print('有向边总数：%d（多重有向图，每条边对应一个配受体对）' % len(row_col))

    # -----------------------------------------------------------------------
    # 构建 PyG Data 对象
    # -----------------------------------------------------------------------
    # edge_index: shape (2, num_edges)，dtype=long
    # 注意：多重有向图中，相同 (src, dst) 节点对可能出现多次（对应不同LR对）
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T

    # edge_attr: shape (num_edges, 3)
    # edge_attr[e] = [score_src_e, score_dst_e, lr_pair_id_e]
    # 其中：
    #   score_src_e = 源细胞类型对在该LR对上的通讯分数（体现通讯强度）
    #   score_dst_e = 目标细胞类型对在该LR对上的通讯分数
    #   lr_pair_id_e = 配受体对ID（标记哪个LR对连接了这两个细胞类型对）
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    # x: shape (N, M)，节点特征矩阵（已L2归一化）
    x = torch.tensor(X_feature, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_loader = DataLoader([graph], batch_size=1)

    print('细胞对图加载完毕')
    return data_loader, num_feature, cp_id_to_pair


class CellPairEncoder(nn.Module):
    """两层 GATv2 编码器，用于细胞对图（Cell-Pair Graph）。

    结构
    ----
    · 第一层 GATv2：in_channels → hidden_channels（利用边特征 edge_dim=3）
    · 第二层 GATv2：hidden_channels → hidden_channels
    · PReLU 激活

    边特征的作用
    -----------
    edge_attr = [score_src, score_dst, lr_pair_id]，维度为 3。
    在 GATv2 的注意力计算中，边特征通过 lin_edge 映射到与节点特征相同的空间：
        e_{ij}^{mapped} = lin_edge(edge_attr_{ij})
    然后与节点特征合并：
        α_{ij} ∝ exp(a^T · tanh(Θ·x_i + Θ·x_j + e_{ij}^{mapped}))
    因此，注意力权重同时取决于：
    1. 源节点的通讯特征（该细胞类型对使用哪些LR通路）
    2. 目标节点的通讯特征
    3. 连接它们的具体配受体对（边特征中的LR分数和LR pair ID）
    这使得模型能够区分"通过不同LR对相连的同一对细胞类型对"的重要性差异。

    add_self_loops=False 的原因
    --------------------------
    本图为多重有向图（multigraph），且边特征维度为3。
    当 add_self_loops=True 时，PyG 需要为自环生成边特征（通过 fill_value 聚合），
    在多重图中容易引起形状不匹配问题。
    此外，节点特征已包含该节点自身的通讯模式，自环的信息冗余，关闭不影响性能。
    """

    def __init__(self, in_channels, hidden_channels, heads, dropout):
        super(CellPairEncoder, self).__init__()
        print('CellPairEncoder: in_channels=%d, hidden_channels=%d, heads=%d'
              % (in_channels, hidden_channels, heads))

        # 第一层 GATv2：in_channels → hidden_channels
        # edge_dim=3：边特征 [score_src, score_dst, lr_pair_id]
        # add_self_loops=False：多重图，关闭自环避免形状冲突
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels,
            edge_dim=3, heads=heads, concat=False,
            add_self_loops=False,
        )

        # 第二层 GATv2：hidden_channels → hidden_channels
        self.conv2 = GATv2Conv(
            hidden_channels, hidden_channels,
            edge_dim=3, heads=heads, concat=False,
            add_self_loops=False,
        )

        self.prelu = nn.PReLU(hidden_channels)

        # 存储注意力权重（训练后可分析：哪些细胞类型对对之间的注意力权重最高）
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
        x : Tensor, shape (N, hidden_channels)
            所有细胞类型对节点的嵌入向量。
            训练收敛后，嵌入相近的细胞类型对具有相似的分子通讯模式。
        """
        # ---------- 第一层 GATv2 ----------
        # 输入：节点特征 (N, in_channels) + 边索引 + 边特征 (E, 3)
        # 输出：更新后的节点特征 (N, hidden_channels) + 注意力分数
        x, attention_scores_l1, attn_unnorm_l1 = self.conv1(
            data.x, data.edge_index,
            edge_attr=data.edge_attr,
            return_attention_weights=True,
        )
        self.attention_scores_mine_l1 = attention_scores_l1
        self.attention_scores_mine_unnormalized_l1 = attn_unnorm_l1

        # ---------- 第二层 GATv2 ----------
        # 输入：第一层输出 (N, hidden_channels) + 边索引 + 边特征
        # 输出：最终节点嵌入 (N, hidden_channels) + 注意力分数
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
    · 将节点特征矩阵 X 的行随机重排，但保持图的拓扑结构（边）不变。
    · 打乱后，节点 i 的特征不再对应其原本在图中的位置/邻居关系。
    · 编码器对这种"损坏图"输出的嵌入 z_neg 应与全图摘要 s 互信息低：
      因为特征与图结构之间的对应关系被破坏，嵌入无法捕获有效的图信息。
    · 通过对比 z_pos（正样本嵌入）和 z_neg（负样本嵌入），
      迫使编码器学习"哪些节点特征与图结构信息是一致的、有意义的"。
    """
    x = data.x[torch.randperm(data.x.size(0))]
    return CorruptedGraphData(x, data.edge_index, data.edge_attr)


def train_CellNEST_cellpair(args, data_loader, in_channels):
    """在细胞对图上训练 DGI 模型。

    训练流程（每个 epoch）
    --------------------
    1. 正样本前向传播：
       原始图（节点特征 X 保持不变）→ CellPairEncoder → 正样本嵌入 z_pos (N, hidden)
    2. 负样本前向传播：
       腐蚀图（节点特征行被随机打乱）→ CellPairEncoder → 负样本嵌入 z_neg (N, hidden)
    3. 全图摘要向量：
       s = sigmoid(mean(z_pos, dim=0))   shape: (hidden,)
       将所有节点的正样本嵌入取行均值，再经 sigmoid 非线性化，
       得到代表整张细胞对图"全局通讯模式"的向量。
    4. DGI 损失（Jensen-Shannon 散度形式的互信息估计）：
       L = -E[log D(z_pos, s)] - E[log(1 - D(z_neg, s))]
       其中 D(z, s) = sigmoid(z^T · W · s) 是双线性判别器。
       最大化互信息 = 最小化 L（因为符号为负）。
    5. 反向传播 + Adam 优化器更新参数。
    6. 每 500 个 epoch 记录损失，损失降低时保存检查点并保存嵌入向量。

    参数
    ----
    args        : argparse.Namespace
        命令行参数（来自 run_CellNEST_cellpair_graph_scrna.py）。
    data_loader : DataLoader
        封装细胞对图的 DataLoader。
    in_channels : int
        节点特征维度 M（= 配受体对总数）。

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
    #   · encoder    : CellPairEncoder（两层 GATv2，学习细胞类型对的嵌入表示）
    #   · summary    : lambda z -> sigmoid(mean(z))
    #                   全图摘要函数：将所有节点嵌入取均值再 sigmoid，
    #                   生成代表整张细胞对图全局通讯模式的向量。
    #   · corruption : 腐蚀函数，打乱节点特征生成负样本
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=CellPairEncoder(
            in_channels=in_channels,
            hidden_channels=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
        ),
        summary=lambda z, *a, **kw: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate)
    DGI_filename = (args.model_path + 'DGI_cellpair_'
                    + args.model_name + '.pth.tar')

    # -----------------------------------------------------------------------
    # 加载检查点（可选，继续上次训练）
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
        }, args.model_path + 'DGI_cellpair_init_' + args.model_name + '.pth.tar')
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
            #   pos_z   : 正样本节点嵌入 (N, hidden) ← 原始图
            #   neg_z   : 负样本节点嵌入 (N, hidden) ← 腐蚀图（节点特征打乱）
            #   summary : 全图摘要向量   (hidden,)
            pos_z, neg_z, summary = DGI_model(data=data)

            # 计算 DGI 损失（最大化 pos_z 与 summary 的互信息）
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

                # 保存节点嵌入向量（shape: N × hidden）
                # 这是最终用于下游分析的关键输出：
                #   · 每行对应一个细胞类型对（typeA → typeB）
                #   · 嵌入相近的细胞类型对 → 相似的分子通讯模式 → 同一通讯模块
                X_embedding = pos_z.cpu().detach().numpy()
                embed_filename = (args.embedding_path + args.model_name
                                  + '_cellpair_Embed_X')
                with gzip.open(embed_filename, 'wb') as fp:
                    pickle.dump(X_embedding, fp)

                # 保存注意力分数
                # attention_scores_mine[0]: edge_index (2, E)
                # attention_scores_mine[1]: 归一化注意力权重 (E, heads)
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
                                 + '_cellpair_attention')
                with gzip.open(attn_filename, 'wb') as fp:
                    pickle.dump(X_attention_bundle, fp)

                # 保存损失曲线
                logfile = open(
                    args.model_path + 'DGI_cellpair_' + args.model_name
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
