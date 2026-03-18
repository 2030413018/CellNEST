# Written By
# Fatema Tuz Zohora
#
# GAT training on a bipartite cell-communication graph.
#
# Graph semantics
# ===============
# The input graph produced by data_preprocess_bipartite_CellNEST.py is
# **bipartite**: edges only connect nodes of *different* types.
#
#   Node Type 1 – "Signal nodes" (indices 0 … num_lr_nodes-1)
#       One node per unique L-R pair (e.g. CCL2-CCR2, TNF-TNFRSF1A).
#       Initial feature vector: one-hot encoding of the L-R pair index.
#
#   Node Type 2 – "Cell-direction nodes" (indices num_lr_nodes … total-1)
#       One node per unique ordered cell pair (sender → receiver).
#       Initial feature vector: aggregated L-R activity profile –
#           feature[k] = mean communication score of L-R pair k in this cell pair
#                        (0 if that L-R pair is not active here).
#
# Why this bipartite structure?
# ==============================
# The GAT can now learn *which L-R pathways co-occur in the same cell-pair
# contexts* (pathway crosstalk).  After training, L-R pair nodes whose
# embeddings cluster together tend to be active in similar cell contexts –
# i.e., they belong to the same co-signalling module (e.g. an inflammation
# module or a proliferation module).

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


def get_bipartite_graph(training_data):
    """Load a bipartite cell-communication graph and build node features.

    The saved file contains::

        [row_col, edge_weight, lig_rec, num_lr_nodes, num_cell_pair_nodes,
         cell_pair_to_id]

    Node features
    -------------
    * L-R pair nodes  (0 … num_lr_nodes-1):
        One-hot vector of length ``num_lr_nodes``.  Node i's feature is the
        i-th standard basis vector, encoding "I am L-R pair i".

    * Cell-pair nodes  (num_lr_nodes … total_nodes-1):
        A vector of length ``num_lr_nodes`` where entry k holds the mean
        communication score of L-R pair k in this cell pair (0 if absent).
        This encodes "which L-R pathways are active in me, and how strongly".

    Args:
        training_data: Path to the bipartite adjacency records file.

    Returns:
        data_loader: DataLoader wrapping a single torch_geometric.data.Data graph.
        num_feature:  Feature dimension (== num_lr_nodes).
    """
    with gzip.open(training_data, 'rb') as f:
        payload = pickle.load(f)
    row_col, edge_weight, lig_rec = payload[0], payload[1], payload[2]
    num_lr_nodes, num_cell_pair_nodes = payload[3], payload[4]

    total_nodes = num_lr_nodes + num_cell_pair_nodes
    feature_dim = num_lr_nodes  # same dimension for both node types

    print('Bipartite graph: %d L-R pair nodes + %d cell-pair nodes = %d total nodes'
          % (num_lr_nodes, num_cell_pair_nodes, total_nodes))
    print('Total number of directed edges in the bipartite input graph: %d'
          % len(row_col))

    # -----------------------------------------------------------------------
    # Build node feature matrix  X  (shape: total_nodes × feature_dim)
    # -----------------------------------------------------------------------
    X = np.zeros((total_nodes, feature_dim), dtype=np.float32)

    # L-R pair nodes: one-hot identity rows
    for lr_id in range(num_lr_nodes):
        X[lr_id, lr_id] = 1.0

    # Cell-pair nodes: aggregated L-R activity profile
    # Accumulate scores and counts separately so we can average.
    cp_lr_score_sum = np.zeros((num_cell_pair_nodes, feature_dim), dtype=np.float64)
    cp_lr_count = np.zeros((num_cell_pair_nodes, feature_dim), dtype=np.float64)

    for k in range(len(row_col)):
        src, dst = row_col[k][0], row_col[k][1]
        # Process only L-R → cell-pair direction edges to avoid double-counting
        if src < num_lr_nodes and dst >= num_lr_nodes:
            lr_node_id = src
            cp_local_id = dst - num_lr_nodes
            lr_score = edge_weight[k][1]           # communication score
            cp_lr_score_sum[cp_local_id, lr_node_id] += lr_score
            cp_lr_count[cp_local_id, lr_node_id] += 1.0

    # Average over repeated edges for the same (cell-pair, L-R) combination
    mask = cp_lr_count > 0
    cp_lr_avg = np.zeros_like(cp_lr_score_sum)
    np.divide(cp_lr_score_sum, cp_lr_count, out=cp_lr_avg, where=mask)

    X[num_lr_nodes:, :] = cp_lr_avg.astype(np.float32)

    # L2-normalise every row so the two node types are on a comparable scale
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X = X / row_norms

    print('Node feature matrix X has dimension', X.shape)

    # -----------------------------------------------------------------------
    # Build PyG Data object
    # -----------------------------------------------------------------------
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    data_loader = DataLoader([graph], batch_size=1)

    print('Bipartite input graph generation done')
    return data_loader, feature_dim


class BipartiteEncoder(nn.Module):
    """Two-layer GATv2 encoder for the bipartite cell-communication graph.

    Self-loops are disabled because nodes of different types cannot have
    self-loops in a bipartite graph (a node would need an edge to itself, which
    would mean the node belongs to both sets – contradicting bipartiteness).
    """

    def __init__(self, in_channels, hidden_channels, heads, dropout):
        super(BipartiteEncoder, self).__init__()
        print('incoming channel %d' % in_channels)

        # add_self_loops=False: bipartite graphs have no self-loops
        self.conv = GATv2Conv(
            in_channels, hidden_channels,
            edge_dim=3, heads=heads, concat=False,
            add_self_loops=False,
        )
        self.conv_2 = GATv2Conv(
            hidden_channels, hidden_channels,
            edge_dim=3, heads=heads, concat=False,
            add_self_loops=False,
        )

        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        # layer 1
        x, attention_scores, attention_scores_unnormalized = self.conv(
            data.x, data.edge_index,
            edge_attr=data.edge_attr,
            return_attention_weights=True,
        )
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores_unnormalized

        # layer 2
        x, attention_scores, attention_scores_unnormalized = self.conv_2(
            x, data.edge_index,
            edge_attr=data.edge_attr,
            return_attention_weights=True,
        )
        self.attention_scores_mine = attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized

        x = self.prelu(x)
        return x


class CorruptedGraphData:
    """Lightweight container for a shuffled graph used in DGI corruption."""

    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data):
    """DGI corruption: shuffle node feature rows."""
    x = data.x[torch.randperm(data.x.size(0))]
    return CorruptedGraphData(x, data.edge_index, data.edge_attr)


def train_CellNEST_bipartite(args, data_loader, in_channels):
    """Train a Deep Graph Infomax model on the bipartite graph.

    The training objective (DGI) maximises mutual information between node
    embeddings and a global graph summary, encouraging the encoder to produce
    embeddings that capture the structural role of each node:

    * L-R pair node embeddings capture *co-occurrence patterns* across cell-pair
      contexts → similar embeddings = similar pathway activity profiles.
    * Cell-pair node embeddings capture *which combination of L-R pathways* is
      active between those two cells.

    Args:
        args:         Parsed command-line arguments (see run_CellNEST_bipartite.py).
        data_loader:  DataLoader wrapping the bipartite graph.
        in_channels:  Input feature dimension (== num_lr_nodes).

    Returns:
        Trained DGI model.
    """
    loss_curve = np.zeros((args.num_epoch // 500 + 1))
    loss_curve_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=BipartiteEncoder(
            in_channels=in_channels,
            hidden_channels=args.hidden,
            heads=args.heads,
            dropout=args.dropout,
        ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate)
    DGI_filename = args.model_path + 'DGI_bipartite_' + args.model_name + '.pth.tar'

    if args.load == 1:
        print('loading model')
        checkpoint = torch.load(DGI_filename)
        DGI_model.load_state_dict(checkpoint['model_state_dict'])
        DGI_model.to(device)
        DGI_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        min_loss = checkpoint['loss']
        print('min_loss was %g' % min_loss)
    else:
        print('Saving init model state ...')
        torch.save({
            'epoch': 0,
            'model_state_dict': DGI_model.state_dict(),
            'optimizer_state_dict': DGI_optimizer.state_dict(),
        }, args.model_path + 'DGI_bipartite_init_model_optimizer_'
            + args.model_name + '.pth.tar')
        min_loss = 10000
        epoch_start = 0

    import datetime
    start_time = datetime.datetime.now()

    for epoch in range(epoch_start, args.num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        DGI_all_loss = []

        for data in data_loader:
            data = data.to(device)
            pos_z, neg_z, summary = DGI_model(data=data)
            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            DGI_loss.backward()
            DGI_all_loss.append(DGI_loss.item())
            DGI_optimizer.step()

        if ((epoch) % 500) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch + 1,
                                                        np.mean(DGI_all_loss)))
            loss_curve[loss_curve_counter] = np.mean(DGI_all_loss)
            loss_curve_counter = loss_curve_counter + 1

            if np.mean(DGI_all_loss) < min_loss:
                min_loss = np.mean(DGI_all_loss)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': DGI_model.state_dict(),
                    'optimizer_state_dict': DGI_optimizer.state_dict(),
                    'loss': min_loss,
                }, DGI_filename)

                # save node embedding
                X_embedding = pos_z.cpu().detach().numpy()
                X_embedding_filename = (args.embedding_path + args.model_name
                                        + '_bipartite_Embed_X')
                with gzip.open(X_embedding_filename, 'wb') as fp:
                    pickle.dump(X_embedding, fp)

                # save attention scores
                X_attention_index = (DGI_model.encoder.attention_scores_mine[0]
                                     .cpu().detach().numpy())

                # layer 1
                X_attention_score_normalized_l1 = (
                    DGI_model.encoder.attention_scores_mine_l1[1]
                    .cpu().detach().numpy())
                X_attention_score_unnormalized_l1 = (
                    DGI_model.encoder.attention_scores_mine_unnormalized_l1
                    .cpu().detach().numpy())

                # layer 2
                X_attention_score_normalized = (
                    DGI_model.encoder.attention_scores_mine[1]
                    .cpu().detach().numpy())
                X_attention_score_unnormalized = (
                    DGI_model.encoder.attention_scores_mine_unnormalized
                    .cpu().detach().numpy())

                X_attention_bundle = [
                    X_attention_index,
                    X_attention_score_normalized_l1,
                    X_attention_score_unnormalized,
                    X_attention_score_unnormalized_l1,
                    X_attention_score_normalized,
                ]
                X_attention_filename = (args.embedding_path + args.model_name
                                        + '_bipartite_attention')
                with gzip.open(X_attention_filename, 'wb') as fp:
                    pickle.dump(X_attention_bundle, fp)

                logfile = open(
                    args.model_path + 'DGI_bipartite_' + args.model_name
                    + '_loss_curve.csv', 'wb')
                np.savetxt(logfile, loss_curve, delimiter=',')
                logfile.close()

    end_time = datetime.datetime.now()
    print('Training time in seconds: ', (end_time - start_time).seconds)

    checkpoint = torch.load(DGI_filename)
    DGI_model.load_state_dict(checkpoint['model_state_dict'])
    DGI_model.to(device)
    DGI_model.eval()
    print("debug loss")
    DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
    print("debug loss latest tuple %g" % DGI_loss.item())

    return DGI_model
