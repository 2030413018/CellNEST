# ============================================================================
# cluster_lrpair_mae_modules.py
# Written by: Fatema Tuz Zohora（参照 CellNEST 框架扩展）
#
# 目标
# ----
# 对 MAE 输出的 LR 对嵌入进行 Leiden 聚类，输出共表达模块。
#
# 输入
# ----
# embedding_data/<data_name>/<model_name>_r<run_id>_lrpair_mae_embed
# input_graph/<data_name>/<data_name>_lrpair_mae_tokens
#
# 输出
# ----
# output/<data_name>/<model_name>_r<run_id>_lrpair_mae_leiden.csv
#   - lr_id, ligand, receptor, module
# output/<data_name>/<model_name>_r<run_id>_lrpair_mae_module_summary.csv
#   - module, size
# ============================================================================

import os
import gzip
import pickle
import argparse
import numpy as np
import pandas as pd
import scanpy as sc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CellNEST LR 对 MAE 嵌入 Leiden 聚类'
    )

    # =================== 必填参数 =============================================
    parser.add_argument('--data_name', type=str, required=True,
                        help='数据集名称（与预处理一致）')
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型名称（与训练一致）')
    parser.add_argument('--run_id', type=int, required=True,
                        help='运行编号（与训练一致）')

    # =================== 可选参数（已设默认值） ================================
    parser.add_argument('--embedding_path', type=str, default='embedding_data/',
                        help='嵌入向量路径')
    parser.add_argument('--input_graph', type=str, default='input_graph/',
                        help='LR Token 矩阵路径')
    parser.add_argument('--output_path', type=str, default='output/',
                        help='聚类结果输出路径')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='Leiden 邻居数（默认 15）')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Leiden 分辨率（默认 1.0）')
    parser.add_argument('--metric', type=str, default='cosine',
                        help='邻居图距离度量（默认 cosine）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认 42）')
    args = parser.parse_args()

    args.embedding_path = args.embedding_path + args.data_name + '/'
    args.input_graph = args.input_graph + args.data_name + '/'
    args.output_path = args.output_path + args.data_name + '/'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model_tag = args.model_name + '_r' + str(args.run_id)

    # =================== 加载嵌入向量 =========================================
    embed_file = args.embedding_path + model_tag + '_lrpair_mae_embed'
    print('Loading embeddings: %s / 正在加载嵌入向量：%s' % (embed_file, embed_file))
    with gzip.open(embed_file, 'rb') as fp:
        embeddings = pickle.load(fp)

    # =================== 加载 LR 对映射 =======================================
    token_file = args.input_graph + args.data_name + '_lrpair_mae_tokens'
    print('Loading LR mapping: %s / 正在加载 LR 对映射：%s' % (token_file, token_file))
    with gzip.open(token_file, 'rb') as fp:
        payload = pickle.load(fp)
    lr_id_to_pair = payload[1]

    if embeddings.shape[0] != len(lr_id_to_pair):
        raise ValueError('嵌入数量与 LR 对数量不一致，请确认训练数据与聚类数据一致。')

    # =================== Leiden 聚类 ==========================================
    print('Running Leiden clustering... / 开始 Leiden 聚类...')
    adata = sc.AnnData(X=embeddings)
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, metric=args.metric)

    try:
        sc.tl.leiden(adata, resolution=args.resolution, key_added='module',
                     random_state=args.seed)
    except Exception as exc:
        raise RuntimeError(
            'Leiden clustering failed. Please ensure leidenalg and igraph '
            'are installed.\n'
            'Leiden 聚类失败。请确认已安装 leidenalg 与 igraph。\n'
            '例如：pip install leidenalg igraph\n'
            '原始错误 / Original error: %s' % exc
        )

    modules = adata.obs['module'].astype(str).tolist()

    # =================== 输出结果 =============================================
    records = []
    for lr_id in range(len(modules)):
        lig, rec = lr_id_to_pair[lr_id]
        records.append({
            'lr_id': lr_id,
            'ligand': lig,
            'receptor': rec,
            'module': modules[lr_id]
        })

    df = pd.DataFrame(records)
    output_csv = args.output_path + model_tag + '_lrpair_mae_leiden.csv'
    df.to_csv(output_csv, index=False)

    summary = df.groupby('module').size().reset_index(name='size')
    summary_csv = args.output_path + model_tag + '_lrpair_mae_module_summary.csv'
    summary.to_csv(summary_csv, index=False)

    print('Clustering saved: %s / 聚类结果保存至：%s' % (output_csv, output_csv))
    print('Module summary saved: %s / 模块统计保存至：%s' % (summary_csv, summary_csv))
    print('Done. / 完成。')
