# ============================================================================
# run_CellNEST_lrpair_mae_scrna.py
# Written by: Fatema Tuz Zohora（参照 CellNEST 框架扩展）
#
# 目标
# ----
# 在“LR 对 Token 矩阵”上训练 Masked Autoencoder (MAE)：
#   · 输入：X_lr (N_lr × M_ctpair)
#   · Tokenization：线性映射 M → d_model
#   · Encoder：多层 Transformer Encoder（无位置编码，纯集合注意力）
#   · 训练策略：随机 Mask 30% 数值并重构
#   · 输出：每个 LR 对的嵌入向量（N_lr × d_model）
#
# 输出文件
# --------
# embedding_data/<data_name>/<model_name>_r<run_id>_lrpair_mae_embed
#   · ndarray (N_lr, d_model)
# model/<data_name>/MAE_lrpair_<model_name>_r<run_id>.pth.tar
#   · 最佳模型参数
# ============================================================================

import os
import numpy as np
import argparse
import random
import gzip
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim


class LRPairMaskedAutoencoder(nn.Module):
    """无位置编码的 LR 对 MAE Transformer。"""

    def __init__(self, input_dim, embed_dim, num_layers, num_heads,
                 dropout, mlp_ratio):
        super().__init__()
        self.token_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, input_dim)
        )

    def forward(self, x, mask):
        """前向传播。

        Parameters
        ----------
        x : Tensor, shape (1, N_lr, M_ctpair)
        mask : BoolTensor, shape (1, N_lr, M_ctpair)
        """
        x_masked = x.masked_fill(mask, 0.0)
        z = self.encoder(self.token_proj(x_masked))
        z = self.encoder_norm(z)
        recon = self.decoder(z)
        return recon, z

    def encode(self, x):
        """获取 LR 对嵌入向量。"""
        z = self.encoder(self.token_proj(x))
        return self.encoder_norm(z)


def sample_mask(x, mask_ratio, mask_nonzero_only):
    """对输入矩阵采样随机掩码。"""
    if mask_nonzero_only:
        candidate = (x != 0)
        if candidate.sum().item() == 0:
            candidate = torch.ones_like(x, dtype=torch.bool)
    else:
        candidate = torch.ones_like(x, dtype=torch.bool)

    candidate_idx = torch.nonzero(candidate, as_tuple=False)
    num_candidates = candidate_idx.shape[0]
    if num_candidates == 0:
        raise RuntimeError(
            'No maskable entries found in the input matrix. '
            'Please verify preprocessing outputs and ensure the input '
            'contains nonzero communication scores.'
        )

    num_mask = max(1, int(num_candidates * mask_ratio))
    perm = torch.randperm(num_candidates, device=x.device)[:num_mask]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[candidate_idx[perm][:, 0], candidate_idx[perm][:, 1]] = True
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CellNEST scRNA-seq LR 对 MAE 训练脚本'
    )

    # =================== 必填参数 =============================================
    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称（与预处理一致）')
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型名称（用于输出文件命名）')
    parser.add_argument('--run_id', type=int, required=True,
                        help='运行编号（0,1,2,...）')

    # =================== 可选参数（已设默认值） ================================
    parser.add_argument('--num_epoch', type=int, default=20000,
                        help='训练迭代次数（默认 20000）')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Token 隐向量维度（默认 256）')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Transformer Encoder 层数（默认 4）')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='多头注意力头数（默认 8）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 比率（默认 0.1）')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='前馈网络扩展比例（默认 4.0）')
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                        help='掩码比例（默认 0.3）')
    parser.add_argument('--mask_nonzero_only', type=int, default=1,
                        help='只对非零元素掩码（默认 1）')
    parser.add_argument('--lr_rate', type=float, default=1e-4,
                        help='Adam 学习率（默认 1e-4）')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减（默认 1e-5）')
    parser.add_argument('--training_data', type=str, default='input_graph/',
                        help='LR Token 矩阵路径')
    parser.add_argument('--embedding_path', type=str, default='embedding_data/',
                        help='嵌入向量保存路径')
    parser.add_argument('--model_path', type=str, default='model/',
                        help='模型权重保存路径')
    parser.add_argument('--manual_seed', type=str, default='no',
                        help='是否手动设置随机种子（"yes" 或 "no"）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子值（仅在 --manual_seed=yes 时生效）')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='多少轮打印一次损失（默认 200）')
    parser.add_argument('--max_lr_pairs', type=int, default=4000,
                        help='训练允许的最大 LR 对数量（默认 4000；<=0 表示不限制）')
    args = parser.parse_args()

    # =================== 路径拼接 =============================================
    args.training_data = (args.training_data + args.data_name + '/'
                          + args.data_name + '_lrpair_mae_tokens')
    args.embedding_path = args.embedding_path + args.data_name + '/'
    args.model_path = args.model_path + args.data_name + '/'
    args.model_name = args.model_name + '_r' + str(args.run_id)

    # =================== 随机种子 =============================================
    if args.manual_seed == 'yes':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f'Manual seed set: {args.seed} / 已设置随机种子：{args.seed}')

    # =================== 创建输出目录 =========================================
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print('========================= Parameters / 参数摘要 ==========================')
    print(f'Dataset: {args.data_name} / 数据集：{args.data_name}')
    print(f'Training data: {args.training_data} / 训练数据：{args.training_data}')
    print(f'Model name: {args.model_name} / 模型名称：{args.model_name}')
    print(f'Embedding dim: {args.embedding_dim} / 嵌入维度：{args.embedding_dim}')
    print(f'Transformer layers: {args.num_layers} / Transformer 层数：{args.num_layers}')
    print(f'Attention heads: {args.num_heads} / 注意力头数：{args.num_heads}')
    print(f'Mask ratio: {args.mask_ratio:.2f} / 掩码比例：{args.mask_ratio:.2f}')
    print(f'Learning rate: {args.lr_rate:g} / 学习率：{args.lr_rate:g}')
    print(f'Epochs: {args.num_epoch} / 训练轮次：{args.num_epoch}')
    print('==========================================================================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} / 使用设备：{device}')

    # =================== 加载 LR Token 矩阵 ===================================
    print('Loading LR token matrix... / 正在加载 LR Token 矩阵...')
    with gzip.open(args.training_data, 'rb') as fp:
        payload = pickle.load(fp)
    X_lr = payload[0]
    lr_id_to_pair = payload[1]
    cp_id_to_pair = payload[2]

    num_lr, num_cp = X_lr.shape
    print(
        f'Token matrix shape: N={num_lr}, M={num_cp} / '
        f'Token 矩阵维度：N={num_lr}, M={num_cp}'
    )
    if args.max_lr_pairs > 0 and num_lr > args.max_lr_pairs:
        raise RuntimeError(
            'N_lr (%d) exceeds --max_lr_pairs (%d). '
            'MAE training uses full self-attention over LR-pair tokens with '
            'O(N_lr^2) memory/time complexity, which can be very slow or OOM. '
            'Please lower --top_lr_pairs during preprocessing or increase '
            '--max_lr_pairs with caution.'
            % (num_lr, args.max_lr_pairs)
        )

    X_tensor = torch.tensor(X_lr, dtype=torch.float, device=device)
    X_tensor = X_tensor.unsqueeze(0)  # batch size = 1

    # =================== 初始化模型 ===========================================
    model = LRPairMaskedAutoencoder(
        input_dim=num_cp,
        embed_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate,
                           weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss(reduction='mean')

    best_loss = float('inf')
    best_epoch = -1

    # =================== 训练循环 =============================================
    print('Training MAE... / 开始训练 MAE...')
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        optimizer.zero_grad()

        mask = sample_mask(
            X_tensor[0],
            mask_ratio=args.mask_ratio,
            mask_nonzero_only=(args.mask_nonzero_only == 1)
        ).unsqueeze(0)

        recon, _ = model(X_tensor, mask)
        masked_target = X_tensor[mask]
        masked_pred = recon[mask]
        loss = mse_loss(masked_pred, masked_target)

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0 or epoch == 1:
            print('Epoch %d | Loss %.6f' % (epoch, loss.item()))

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            model_path = os.path.join(
                args.model_path,
                'MAE_lrpair_%s.pth.tar' % args.model_name
            )
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': best_loss}, model_path)

    print(
        f'Training complete. Best loss {best_loss:.6f} (epoch {best_epoch}) / '
        f'训练完毕。最佳损失 {best_loss:.6f} (epoch {best_epoch})'
    )

    # =================== 导出嵌入 =============================================
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(X_tensor).squeeze(0).cpu().numpy()

    embed_path = os.path.join(
        args.embedding_path,
        '%s_lrpair_mae_embed' % args.model_name
    )
    with gzip.open(embed_path, 'wb') as fp:
        pickle.dump(embeddings, fp)

    print(f'Embeddings saved: {embed_path} / 嵌入向量已保存至：{embed_path}')
    print('Next: run cluster_lrpair_mae_modules.py for Leiden clustering. / '
          '下一步建议：运行 cluster_lrpair_mae_modules.py 进行 Leiden 聚类。')
