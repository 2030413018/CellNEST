# Written By
# Fatema Tuz Zohora
#
# Bipartite graph construction for CellNEST pathway-crosstalk analysis.
#
# Graph design
# ============
# The resulting graph is **bipartite**: it contains two disjoint sets of nodes
# and edges are drawn only *between* these two sets – never within the same set.
#
#   Node Type 1 – "Signal nodes" (indices 0 … num_lr_nodes-1)
#       Each node represents one unique ligand-receptor (L-R) pair present in
#       the dataset, e.g. CCL2-CCR2 or TNF-TNFRSF1A.
#
#   Node Type 2 – "Cell-direction nodes" (indices num_lr_nodes … total_nodes-1)
#       Each node represents one unique *ordered* cell pair (sender → receiver),
#       e.g. Myeloid cell 3 → T cell 17.
#
#   Edges
#       An undirected edge connects L-R signal node L and cell-pair node P(i,j)
#       when the L-R communication L is active in the cell pair (i → j).
#       Edge features: [distance_weight, lr_coexpression_score, lr_pair_id]
#
# Purpose
# =======
# Running a Graph Attention Network (GAT) on this bipartite structure lets
# the model learn *pathway crosstalk*: it discovers which L-R signalling pairs
# tend to co-occur across the same cell-pair contexts (inflammation modules,
# proliferation modules, etc.).

print('package loading')
import numpy as np
import pickle
from scipy import sparse
import qnorm
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
import gzip
import argparse
import os
import scanpy as sc
import json
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
print('user input reading')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################## Mandatory ####################################################################
    parser.add_argument('--data_name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--data_from', type=str, required=True,
                        help='Path to the dataset to read from. Space Ranger outs/ folder is '
                             'preferred. Otherwise, provide the *.mtx file of the gene expression '
                             'matrix.')
    ################# default is set ################################################################
    parser.add_argument('--data_type', type=str, default='visium',
                        help='Set one of [visium, anndata, scrna]. Use scrna for single-cell '
                             'RNA-seq without spatial coordinates; positions will be derived from '
                             'expression via PCA and UMAP.')
    parser.add_argument('--data_to', type=str, default='input_graph/',
                        help='Path to save the input graph (to be passed to GAT)')
    parser.add_argument('--metadata_to', type=str, default='metadata/',
                        help='Path to save the metadata')
    parser.add_argument('--filter_min_cell', type=int, default=1,
                        help='Minimum number of cells for gene filtering')
    parser.add_argument('--threshold_gene_exp', type=float, default=98,
                        help='Threshold percentile for gene expression. Genes above this '
                             'percentile are considered active.')
    parser.add_argument('--tissue_position_file', type=str, default='None',
                        help='If your --data_from argument points to a *.mtx file instead of '
                             'Space Ranger, then please provide the path to tissue position file.')
    parser.add_argument('--juxtacrine_distance', type=float, default=-1,
                        help='Distance for filtering ligand-receptor pairs based on cell-cell '
                             'contact information. Automatically calculated unless provided. It '
                             'has the same unit as the coordinates (for Visium, that is pixel).')
    parser.add_argument('--split', type=int, default=0,
                        help='Set to 1 if you plan to run training by splitting the input graph '
                             'into multiple subgraphs')
    parser.add_argument('--distance_measure', type=str, default='fixed',
                        help='Set neighborhood cutoff criteria. Choose from [knn, fixed]')
    parser.add_argument('--k', type=int, default=50,
                        help='Set neighborhood cutoff number. This will be used if '
                             '--distance_measure=knn')
    parser.add_argument('--neighborhood_threshold', type=float, default=0,
                        help='Set neighborhood threshold distance in terms of same unit as the '
                             'coordinates')
    parser.add_argument('--block_autocrine', type=int, default=0,
                        help='Set to 1 if you want to ignore autocrine signals.')
    parser.add_argument('--block_juxtacrine', type=int, default=0,
                        help='Set to 1 if you want to ignore juxtacrine signals.')
    parser.add_argument('--database_path', type=str,
                        default='database/CellNEST_database.csv',
                        help='Provide your desired ligand-receptor database path here. Default '
                             'database is a combination of CellChat and NicheNet database.')
    parser.add_argument('--set_ROI', type=int, default=0,
                        help='Set to 1 if you want to use ROI')
    parser.add_argument('--x_min', type=int, default=-1, help='Set if you want to use ROI')
    parser.add_argument('--x_max', type=int, default=-1, help='Set if you want to use ROI')
    parser.add_argument('--y_min', type=int, default=-1, help='Set if you want to use ROI')
    parser.add_argument('--y_max', type=int, default=-1, help='Set if you want to use ROI')
    args = parser.parse_args()

    if args.data_to == 'input_graph/':
        args.data_to = args.data_to + args.data_name + '/'
    if not os.path.exists(args.data_to):
        os.makedirs(args.data_to)

    if args.metadata_to == 'metadata/':
        args.metadata_to = args.metadata_to + args.data_name + '/'
    if not os.path.exists(args.metadata_to):
        os.makedirs(args.metadata_to)

    ####### get the gene id, cell barcode, cell coordinates ######
    print('Input data reading')
    if args.tissue_position_file == 'None':  # Data is available in Space Ranger output format
        if args.data_type == 'visium':
            adata_h5 = sc.read_visium(path=args.data_from,
                                      count_file='filtered_feature_bc_matrix.h5')
            print('input data read done')
            gene_count_before = len(list(adata_h5.var_names))
            sc.pp.filter_genes(adata_h5, min_cells=args.filter_min_cell)
            gene_count_after = len(list(adata_h5.var_names))
            print('Gene filtering done. Number of genes reduced from %d to %d'
                  % (gene_count_before, gene_count_after))
            gene_ids = list(adata_h5.var_names)
            coordinates = adata_h5.obsm['spatial']
            cell_barcode = np.array(adata_h5.obs.index)
            print('Number of barcodes: %d' % cell_barcode.shape[0])
            print('Applying quantile normalization')
            temp = qnorm.quantile_normalize(
                np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))
            cell_vs_gene = np.transpose(temp)
            if args.juxtacrine_distance == -1:
                file = open(args.data_from + '/spatial/scalefactors_json.json', 'r')
                data = json.load(file)
                spot_diameter = data["spot_diameter_fullres"]
                args.juxtacrine_distance = spot_diameter

        elif args.data_type == 'anndata':
            adata_h5 = sc.read_h5ad(args.data_from)
            print('input data read done')
            gene_count_before = len(list(adata_h5.var_names))
            sc.pp.filter_genes(adata_h5, min_cells=args.filter_min_cell)
            gene_count_after = len(list(adata_h5.var_names))
            print('Gene filtering done. Number of genes reduced from %d to %d'
                  % (gene_count_before, gene_count_after))
            gene_ids = list(adata_h5.var_names)
            coordinates = np.array(adata_h5.obsm['spatial'])
            if args.set_ROI == 1:
                if args.x_min == -1:
                    args.x_min = np.min(coordinates[:][0])
                if args.x_max == -1:
                    args.x_max = np.max(coordinates[:][0])
                if args.y_min == -1:
                    args.y_min = np.min(coordinates[:][1])
                if args.y_max == -1:
                    args.y_max = np.max(coordinates[:][1])

                keep_cells = []
                for i in range(0, coordinates.shape[0]):
                    if args.x_min <= coordinates[i][0] <= args.x_max:
                        if args.y_min <= coordinates[i][1] <= args.y_max:
                            keep_cells.append(i)

                adata_h5 = adata_h5[keep_cells]
                print('after ROI cropping:')

            print(adata_h5)

            cell_barcode = np.array(adata_h5.obs_names)
            print('Number of barcodes: %d' % cell_barcode.shape[0])
            print('Applying quantile normalization')
            temp = qnorm.quantile_normalize(
                np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))
            cell_vs_gene = np.transpose(temp)
        elif args.data_type == 'scrna':
            adata_h5 = sc.read_h5ad(args.data_from)
            print('scRNA-seq input data read done')
            gene_count_before = len(list(adata_h5.var_names))
            sc.pp.filter_genes(adata_h5, min_cells=args.filter_min_cell)
            gene_count_after = len(list(adata_h5.var_names))
            print('Gene filtering done. Number of genes reduced from %d to %d'
                  % (gene_count_before, gene_count_after))
            gene_ids = list(adata_h5.var_names)
            cell_barcode = np.array(adata_h5.obs_names)
            print('Number of cells: %d' % cell_barcode.shape[0])

            print('Computing 2D embedding from expression data (PCA + UMAP) ...')
            adata_for_embed = adata_h5.copy()
            sc.pp.normalize_total(adata_for_embed, target_sum=1e4)
            sc.pp.log1p(adata_for_embed)
            n_pcs = min(50, adata_for_embed.shape[0] - 1, adata_for_embed.shape[1] - 1)
            sc.tl.pca(adata_for_embed, n_comps=n_pcs)
            n_nb = min(30, adata_for_embed.shape[0] - 1)
            sc.pp.neighbors(adata_for_embed, n_neighbors=n_nb, n_pcs=n_pcs)
            sc.tl.umap(adata_for_embed)
            coordinates = np.array(adata_for_embed.obsm['X_umap'], dtype=np.float64)
            del adata_for_embed
            gc.collect()
            print('2D embedding computed for %d cells' % coordinates.shape[0])

            args.distance_measure = 'knn'
            args.k = min(args.k, cell_barcode.shape[0] - 1)
            args.block_juxtacrine = 1
            print('scRNA-seq mode: using KNN (k=%d) neighborhood; juxtacrine filtering '
                  'disabled' % args.k)

            print('Applying quantile normalization')
            temp = qnorm.quantile_normalize(
                np.transpose(sparse.csr_matrix.toarray(adata_h5.X)))
            cell_vs_gene = np.transpose(temp)
    else:  # Data is not available in Space Ranger output format
        temp = sc.read_10x_mtx(args.data_from)
        print('*.mtx file read done')
        gene_count_before = len(list(temp.var_names))
        sc.pp.filter_genes(temp, min_cells=args.filter_min_cell)
        gene_count_after = len(list(temp.var_names))
        print('Gene filtering done. Number of genes reduced from %d to %d'
              % (gene_count_before, gene_count_after))
        gene_ids = list(temp.var_names)
        cell_barcode = np.array(temp.obs.index)
        print('Number of barcodes: %d' % cell_barcode.shape[0])
        print('Applying quantile normalization')
        temp = qnorm.quantile_normalize(
            np.transpose(sparse.csr_matrix.toarray(temp.X)))
        cell_vs_gene = np.transpose(temp)

        df = pd.read_csv(args.tissue_position_file, sep=",", header=None)
        tissue_position = df.values
        barcode_vs_xy = dict()
        for i in range(0, tissue_position.shape[0]):
            barcode_vs_xy[tissue_position[i][0]] = [tissue_position[i][4],
                                                     tissue_position[i][5]]

        coordinates = np.zeros((cell_barcode.shape[0], 2))
        for i in range(0, cell_barcode.shape[0]):
            coordinates[i, 0] = barcode_vs_xy[cell_barcode[i]][0]
            coordinates[i, 1] = barcode_vs_xy[cell_barcode[i]][1]

    ##################### make metadata: barcode_info ###################################
    i = 0
    barcode_info = []
    for cell_code in cell_barcode:
        barcode_info.append([cell_code, coordinates[i, 0], coordinates[i, 1], 0])
        i = i + 1

    gene_info = dict()
    for gene in gene_ids:
        gene_info[gene] = ''

    gene_index = dict()
    i = 0
    for gene in gene_ids:
        gene_index[gene] = i
        i = i + 1

    # build physical distance matrix
    print('Build physical distance matrix')
    if args.distance_measure == 'fixed':
        if args.neighborhood_threshold == 0:
            distance_matrix = euclidean_distances(coordinates, coordinates)
            sorted_first_row = np.sort(distance_matrix[0, :])
            distance_a_b = sorted_first_row[1]
            args.neighborhood_threshold = distance_a_b * 4
            distance_matrix = 0
            gc.collect()

        nbrs = NearestNeighbors(radius=args.neighborhood_threshold, algorithm='kd_tree',
                                n_jobs=-1)
        nbrs.fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        print('Neighborhood distance is set to be %g (same unit as the coordinates)'
              % args.neighborhood_threshold)
    else:
        print('Neighborhood distance is set to be %d nearest neighbors' % args.k)
        nbrs = NearestNeighbors(n_neighbors=args.k, algorithm='kd_tree', n_jobs=-1)
        nbrs.fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)

    unique_distances = np.unique(distances)
    distance_a_b = sorted(unique_distances)[1]

    print('Assign weight to the neighborhood relations based on neighborhood distance')
    weightdict_i_to_j = defaultdict(dict)
    for cell_idx in range(0, indices.shape[0]):
        max_value = np.max(distances[cell_idx, :])
        min_value = np.min(distances[cell_idx, :])
        for neigh_idx in range(0, indices.shape[1]):
            neigh_cell_idx = indices[cell_idx][neigh_idx]
            distance_neigh_cell = distances[cell_idx][neigh_idx]
            flipped_distance_neigh_cell = (
                1 - (distance_neigh_cell - min_value) / (max_value - min_value))
            weightdict_i_to_j[neigh_cell_idx][cell_idx] = flipped_distance_neigh_cell

    if args.juxtacrine_distance == -1:
        args.juxtacrine_distance = distance_a_b

    if args.block_juxtacrine == 0:
        print("Auto calculated juxtacrine distance is %g. To change it use "
              "--juxtacrine_distance" % args.juxtacrine_distance)

    i = 0
    node_id_sorted_xy = []
    for cell_code in cell_barcode:
        node_id_sorted_xy.append([i, coordinates[i, 0], coordinates[i, 1]])
        i = i + 1
    node_id_sorted_xy = sorted(node_id_sorted_xy, key=lambda x: (x[1], x[2]))

    if args.split > 0:
        with gzip.open(args.metadata_to + args.data_name + '_' + 'node_id_sorted_xy',
                       'wb') as fp:
            pickle.dump(node_id_sorted_xy, fp)

    ####################################################################
    # ligand - receptor database
    print('ligand-receptor database reading.')
    df = pd.read_csv(args.database_path, sep=",")
    print('ligand-receptor database reading done.')
    print('Preprocess start.')
    ligand_dict_dataset = defaultdict(list)
    cell_cell_contact = dict()
    count_pair = 0
    for i in range(0, df["Ligand"].shape[0]):
        ligand = df["Ligand"][i]
        if ligand not in gene_info:
            continue

        receptor = df["Receptor"][i]
        if receptor not in gene_info:
            continue

        ligand_dict_dataset[ligand].append(receptor)
        gene_info[ligand] = 'included'
        gene_info[receptor] = 'included'
        count_pair = count_pair + 1

        if df["Annotation"][i] == 'Cell-Cell Contact':
            cell_cell_contact[receptor] = ''

    print('number of ligands %d ' % len(ligand_dict_dataset.keys()))

    included_gene = []
    for gene in gene_info.keys():
        if gene_info[gene] == 'included':
            included_gene.append(gene)

    print('Total genes in this dataset: %d, number of genes working as ligand and/or '
          'receptor: %d ' % (len(gene_ids), len(included_gene)))

    # assign id to each L-R pair in the database
    # l_r_pair[ligand][receptor] = lr_node_id   (also serves as the signal-node index)
    l_r_pair = dict()
    lr_id = 0
    for gene in list(ligand_dict_dataset.keys()):
        ligand_dict_dataset[gene] = list(set(ligand_dict_dataset[gene]))
        l_r_pair[gene] = dict()
        for receptor_gene in ligand_dict_dataset[gene]:
            l_r_pair[gene][receptor_gene] = lr_id
            lr_id = lr_id + 1

    num_lr_nodes = lr_id  # total number of L-R pair (signal) nodes
    print('number of ligand-receptor pairs in this dataset %d ' % num_lr_nodes)

    #####################################################################################
    # Set threshold gene percentile
    cell_percentile = []
    for i in range(0, cell_vs_gene.shape[0]):
        y = sorted(cell_vs_gene[i])
        active_cutoff = np.percentile(y, args.threshold_gene_exp)
        if active_cutoff == min(cell_vs_gene[i][:]):
            times = 1
            while active_cutoff == min(cell_vs_gene[i][:]):
                new_threshold = args.threshold_gene_exp + 5 * times
                if new_threshold >= 100:
                    active_cutoff = max(cell_vs_gene[i][:])
                    if active_cutoff == min(cell_vs_gene[i][:]):
                        active_cutoff = max(cell_vs_gene[i][:]) + 1
                    break
                active_cutoff = np.percentile(y, new_threshold)
                times = times + 1

        cell_percentile.append(active_cutoff)

    print('set threshold gene percentile done')

    ##############################################################################
    # Enumerate all active communications: cells_ligand_vs_receptor[i][j] holds
    # a list of (ligand, receptor, score, lr_pair_id) for cell pair (i → j).
    count_total_edges = 0
    cells_ligand_vs_receptor = defaultdict(dict)

    ligand_list = list(ligand_dict_dataset.keys())
    start_index = 0
    end_index = len(ligand_list)
    print('some preprocessing before making the input graph')
    for g in range(start_index, end_index):
        gene = ligand_list[g]
        for i in weightdict_i_to_j:
            if cell_vs_gene[i][gene_index[gene]] < cell_percentile[i]:
                continue
            for j in weightdict_i_to_j[i]:
                if args.block_autocrine == 1 and i == j:
                    continue
                for gene_rec in ligand_dict_dataset[gene]:
                    if cell_vs_gene[j][gene_index[gene_rec]] >= cell_percentile[j]:
                        if (gene_rec in cell_cell_contact) and (
                                args.block_juxtacrine == 1 or
                                euclidean_distances(coordinates[i:i + 1],
                                                    coordinates[j:j + 1])
                                > args.juxtacrine_distance):
                            continue

                        communication_score = (cell_vs_gene[i][gene_index[gene]]
                                               * cell_vs_gene[j][gene_index[gene_rec]])
                        relation_id = l_r_pair[gene][gene_rec]

                        if communication_score <= 0:
                            print('zero valued ccc score found. Might be a potential '
                                  'ERROR!! ')
                            continue

                        if i in cells_ligand_vs_receptor:
                            if j in cells_ligand_vs_receptor[i]:
                                cells_ligand_vs_receptor[i][j].append(
                                    [gene, gene_rec, communication_score, relation_id])
                            else:
                                cells_ligand_vs_receptor[i][j] = [
                                    [gene, gene_rec, communication_score, relation_id]]
                        else:
                            cells_ligand_vs_receptor[i][j] = [
                                [gene, gene_rec, communication_score, relation_id]]

                        count_total_edges = count_total_edges + 1

        print('%d/%d ligand genes processed' % (g + 1, len(ligand_list)), end='\r')

    print('')

    ################################################################################
    # Bipartite graph construction
    # ============================================================
    # Node Type 1 – Signal nodes (L-R pairs):
    #     Index range: 0 … num_lr_nodes-1
    #     Each node is one L-R pair; its index equals l_r_pair[ligand][receptor].
    #
    # Node Type 2 – Cell-direction nodes (cell pairs):
    #     Index range: num_lr_nodes … num_lr_nodes + num_cell_pair_nodes - 1
    #     Each node represents a unique ordered cell pair (sender_cell, receiver_cell).
    #     cell_pair_to_id[(i, j)] → local id (0-based); global node id = num_lr_nodes + local_id
    #
    # Edges (undirected, stored as bidirectional pairs):
    #     For every active (cell_i → cell_j, L-R pair lr_id) triple:
    #       add edge  lr_node_id  ↔  (num_lr_nodes + cell_pair_local_id)
    # ============================================================

    # Step 1: enumerate unique cell pairs and assign local IDs
    cell_pair_to_id = {}   # (cell_i, cell_j) → local cell-pair node id
    cell_pair_id_counter = 0

    for i in cells_ligand_vs_receptor.keys():
        for j in cells_ligand_vs_receptor[i].keys():
            if (i in weightdict_i_to_j and j in weightdict_i_to_j[i]
                    and len(cells_ligand_vs_receptor[i][j]) > 0):
                pair_key = (i, j)
                if pair_key not in cell_pair_to_id:
                    cell_pair_to_id[pair_key] = cell_pair_id_counter
                    cell_pair_id_counter += 1

    num_cell_pair_nodes = cell_pair_id_counter
    total_num_nodes = num_lr_nodes + num_cell_pair_nodes

    print('Bipartite graph: %d L-R pair nodes + %d cell-pair nodes = %d total nodes'
          % (num_lr_nodes, num_cell_pair_nodes, total_num_nodes))

    # Step 2: build bipartite edge list
    row_col = []     # [source_node_id, target_node_id]
    edge_weight = [] # [distance_weight, lr_coexpression_score, lr_pair_id]
    lig_rec = []     # [ligand_name, receptor_name]

    for i in cells_ligand_vs_receptor.keys():
        for j in cells_ligand_vs_receptor[i].keys():
            if (i in weightdict_i_to_j and j in weightdict_i_to_j[i]
                    and len(cells_ligand_vs_receptor[i][j]) > 0):
                cp_local_id = cell_pair_to_id[(i, j)]
                cp_node_id = num_lr_nodes + cp_local_id
                dist_weight = weightdict_i_to_j[i][j]

                for k in range(len(cells_ligand_vs_receptor[i][j])):
                    gene = cells_ligand_vs_receptor[i][j][k][0]
                    gene_rec = cells_ligand_vs_receptor[i][j][k][1]
                    lr_score = cells_ligand_vs_receptor[i][j][k][2]
                    lr_node_id = cells_ligand_vs_receptor[i][j][k][3]  # == l_r_pair[gene][gene_rec]

                    edge_feat = [dist_weight, lr_score, float(lr_node_id)]

                    # L-R signal node → cell-pair direction node
                    row_col.append([lr_node_id, cp_node_id])
                    edge_weight.append(edge_feat)
                    lig_rec.append([gene, gene_rec])

                    # cell-pair direction node → L-R signal node  (reverse direction)
                    row_col.append([cp_node_id, lr_node_id])
                    edge_weight.append(edge_feat)
                    lig_rec.append([gene, gene_rec])

    print('total number of nodes is %d (LR: %d, cell-pairs: %d), '
          'and edges (directed) is %d in the bipartite input graph'
          % (total_num_nodes, num_lr_nodes, num_cell_pair_nodes, len(row_col)))
    print('preprocess done.')
    print('writing data ...')

    ################## bipartite input graph ###########################################
    with gzip.open(args.data_to + args.data_name + '_bipartite_adjacency_records',
                   'wb') as fp:
        pickle.dump([row_col, edge_weight, lig_rec,
                     num_lr_nodes, num_cell_pair_nodes, cell_pair_to_id], fp)

    ################# metadata #####################################################
    with gzip.open(args.metadata_to + args.data_name + '_barcode_info', 'wb') as fp:
        pickle.dump(barcode_info, fp)

    ################## required for the CellNEST interactive version ###################
    df = pd.DataFrame(gene_ids)
    df.to_csv(args.metadata_to + 'gene_ids_' + args.data_name + '.csv',
              index=False, header=False)
    df = pd.DataFrame(cell_barcode)
    df.to_csv(args.metadata_to + 'cell_barcode_' + args.data_name + '.csv',
              index=False, header=False)
    df = pd.DataFrame(coordinates)
    df.to_csv(args.metadata_to + 'coordinates_' + args.data_name + '.csv',
              index=False, header=False)

    # Save LR pair ID mapping for downstream analysis
    lr_pair_to_id = {}  # (ligand, receptor) → lr_node_id
    for lig in l_r_pair:
        for rec in l_r_pair[lig]:
            lr_pair_to_id[(lig, rec)] = l_r_pair[lig][rec]

    with gzip.open(args.metadata_to + args.data_name + '_lr_pair_to_id', 'wb') as fp:
        pickle.dump(lr_pair_to_id, fp)

    with gzip.open(args.metadata_to + args.data_name + '_cell_pair_to_id', 'wb') as fp:
        pickle.dump(cell_pair_to_id, fp)

    # Save quantile-transformed expression matrix for downstream analysis
    with gzip.open(args.data_to + args.data_name + '_cell_vs_gene_quantile_transformed',
                   'wb') as fp:
        pickle.dump(cell_vs_gene, fp)

    print('write data done')
