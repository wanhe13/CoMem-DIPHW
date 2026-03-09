#!/usr/bin/env python
# coding: utf-8
# Revised to output 5 metrics: ACC, NMI, ARI, AMI, F1
# This script varies avg_genes_per_module

import argparse
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
import datetime
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import zscore
from scipy.optimize import linear_sum_assignment
import networkx as nx
import igraph as ig
import community
import leidenalg as la
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, f1_score)
from sklearn.manifold import TSNE
from umap import UMAP
import csv
import warnings
warnings.filterwarnings("ignore")

# Import custom modules
from Simulation_functions import *
from CoMemDIPHW import *
from DIPHW import *

parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)
from functions import *


# =============================================================================
# Metric Functions
# =============================================================================
def cluster_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.dtype.kind in ['U', 'S', 'O']:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
    
    y_true = y_true.astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def compute_f1_score(y_true, y_pred):
    """Compute weighted F1 score with Hungarian matching"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.dtype.kind in ['U', 'S', 'O']:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
    
    y_true = y_true.astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
    
    return f1_score(y_true, y_pred_mapped, average='weighted')


def compute_all_metrics(y_true, y_pred):
    """Compute all 5 clustering metrics"""
    acc = cluster_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    f1 = compute_f1_score(y_true, y_pred)
    return acc, nmi, ari, ami, f1


def store_metrics(clustering_performance, method_name, y_true, y_pred):
    """Compute and store all 5 metrics for a method"""
    acc, nmi, ari, ami, f1 = compute_all_metrics(y_true, y_pred)
    clustering_performance[f'{method_name}_ACC'] = acc
    clustering_performance[f'{method_name}_NMI'] = nmi
    clustering_performance[f'{method_name}_ARI'] = ari
    clustering_performance[f'{method_name}_AMI'] = ami
    clustering_performance[f'{method_name}_F1'] = f1
    print(f"{method_name}: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, AMI={ami:.4f}, F1={f1:.4f}")
    return clustering_performance


# =============================================================================
# Helper Functions
# =============================================================================
def normalize_matrix_cpm(matrix):
    """Normalize using Counts Per Million (CPM)"""
    matrix = matrix.copy()
    column_sums = np.sum(matrix, axis=0)
    column_sums[column_sums == 0] = 1
    normalized_matrix = 1e6 * matrix / column_sums
    return normalized_matrix


def filter_zero_expression_community(expression_df_with_CommunityAssignment, thres=0):
    """Filter out genes and cells with zero expression"""
    df = expression_df_with_CommunityAssignment
    last_row = df.iloc[-1, :]
    row_sums = df.iloc[:-1, :].sum(axis=1)
    col_sums = df.iloc[:-1, :].sum(axis=0)
    filtered_genes = row_sums > thres
    filtered_df = df.iloc[:-1, :].loc[filtered_genes, :]
    filtered_cells = col_sums > thres
    filtered_df = filtered_df.loc[:, filtered_cells]
    last_row_filtered = last_row.loc[filtered_cells]
    filtered_df = pd.concat([filtered_df, last_row_filtered.to_frame().T])
    return filtered_df


# =============================================================================
# Main Script
# =============================================================================
if __name__ == "__main__":
    # Parameters from command line
    avg_genes_per_module = int(sys.argv[1])
    run_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    # Set random seed based on run index for reproducibility
    RANDOM_SEED = 42 + run_idx
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Fixed parameters
    rows = 3000
    cols = 1000
    num_modules = 30
    avg_cells_per_module = 38
    target_density = 0.03
    module_density = 0.3
    inter_module_density = 0.1
    inter_module_connection_probability = 0.6
    lambda_background = 10
    lambda_module = 20
    inter_module_lambda = 10
    
    # Random walk parameters
    num_walks_per_node = 30
    walk_length = 30
    embedded_dimensions = 30
    preference_exponent = 50
    
    print(f'avg_genes_per_module: {avg_genes_per_module}')
    print(f'run_idx: {run_idx}')
    print(f'random_seed: {RANDOM_SEED}')
    print(f'num_modules: {num_modules}')
    
    # Create output directory and set filename
    os.makedirs('results', exist_ok=True)
    filename = 'results/ClusteringPerformance_ngenes_5metrics.csv'
    
    # Generate simulated data
    modules = simulate_input_modules(rows, cols, num_modules, avg_genes_per_module, avg_cells_per_module)
    
    print("Generated Modules:")
    for i, (start_row, end_row, start_col, end_col) in enumerate(modules):
        print(f"Module {i + 1}: Rows {start_row}-{end_row}, Columns {start_col}-{end_col}")
    
    sparse_data = create_sparse_matrix_with_inter_module_variance(
        rows, cols, modules, target_density, module_density, inter_module_density,
        inter_module_connection_probability, lambda_background, lambda_module, inter_module_lambda
    )
    
    # Prepare data
    data_df_community = pd.DataFrame.sparse.from_spmatrix(sparse_data.T)
    community_assignments = modules_to_community_dict(modules, cols)
    data_df_community['community'] = list(community_assignments.values())
    data_df_community = data_df_community.T
    
    # Shuffle columns
    column_names = data_df_community.columns.tolist()
    shuffled_column_names = np.random.permutation(column_names)
    shuffled_df_community = data_df_community[shuffled_column_names]
    
    # Preprocess
    non_zero_df = filter_zero_expression_community(shuffled_df_community)
    non_zero_data = np.array(non_zero_df.iloc[:-1].values)
    community_assignments = dict(zip(non_zero_df.columns, non_zero_df.loc['community']))
    normalized_data = normalize_matrix_cpm(non_zero_data).astype(float)
    preprocessed_data = np.log2(normalized_data + 1)
    
    preprocessed_df = non_zero_df.copy()
    nrows, ncols = preprocessed_df.shape
    preprocessed_df.iloc[:-1, :ncols] = preprocessed_data
    node_list = list(preprocessed_df.columns)
    
    # Create correlation networks
    correlation_threshold_percentile = 99
    cell_coexp_HT, cell_coexp = create_correlation_network(preprocessed_data, correlation_threshold_percentile)
    gene_coexp_HT, gene_coexp = create_correlation_network(preprocessed_data.T, correlation_threshold_percentile)
    
    # Create graphs
    cell_map = {i: j for i, j in enumerate(node_list)}
    G_cell = nx.from_numpy_array(cell_coexp)
    G_cell_HT = nx.from_numpy_array(cell_coexp_HT)
    G_cell = nx.relabel_nodes(G_cell, cell_map)
    G_cell_HT = nx.relabel_nodes(G_cell_HT, cell_map)
    
    # Get true labels
    sorted_dict = {k: community_assignments[k] for k in node_list}
    true_labels = list(sorted_dict.values())
    
    num_clusters = num_modules
    data = preprocessed_data
    clustering_performance = defaultdict()
    
    # KMeans with fixed random state
    kmeans = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED, n_init=10)
    
    # =========================================================================
    # Graph-based Clustering Methods
    # =========================================================================
    
    # --- Louvain (weighted) ---
    partition = community.best_partition(G_cell, weight='weight', random_state=RANDOM_SEED)
    predicted_labels = [partition[node] for node in node_list]
    clustering_performance = store_metrics(clustering_performance, 'Louvain', true_labels, predicted_labels)
    
    # --- Leiden ---
    G_ig = ig.Graph.TupleList(G_cell.edges(data=True), weights=True)
    G_ig.vs['name'] = list(G_cell.nodes())
    partition_leiden = la.find_partition(
        G_ig, la.RBConfigurationVertexPartition,
        weights=[e['weight'] for e in G_ig.es['weight']], resolution_parameter=1.1, n_iterations=-1, seed=RANDOM_SEED
    )
    vertex_names = G_ig.vs['name']
    community_dict_leiden = {vertex_names[v]: cid for cid, comm in enumerate(partition_leiden) for v in comm}
    predicted_labels = [community_dict_leiden[node] for node in node_list]
    clustering_performance = store_metrics(clustering_performance, 'Leiden', true_labels, predicted_labels)
    
    # --- Greedy Modularity ---
    partition_gm = nx.community.greedy_modularity_communities(G_cell, weight='weight')
    community_dict_gm = {i: n for n, c in enumerate(partition_gm) for i in c}
    predicted_labels = [community_dict_gm[node] for node in node_list]
    clustering_performance = store_metrics(clustering_performance, 'GreedyModularity', true_labels, predicted_labels)
    
    # --- Louvain HT ---
    partition_HT = community.best_partition(G_cell_HT, weight='weight', random_state=RANDOM_SEED)
    predicted_labels = [partition_HT[node] for node in node_list]
    clustering_performance = store_metrics(clustering_performance, 'Louvain_HT', true_labels, predicted_labels)
    
    # --- Infomap ---
    G_ig_HT = ig.Graph.TupleList(G_cell_HT.edges(data=True), weights=True)
    G_ig_HT.vs['name'] = list(G_cell_HT.nodes())
    partition_infomap = G_ig_HT.community_infomap(edge_weights=[e['weight'] for e in G_ig_HT.es['weight']])
    vertex_names_HT = G_ig_HT.vs['name']
    community_dict_infomap = {vertex_names_HT[v]: cid for cid, comm in enumerate(partition_infomap) for v in comm}
    predicted_labels = [community_dict_infomap[node] for node in node_list]
    clustering_performance = store_metrics(clustering_performance, 'Infomap', true_labels, predicted_labels)
    
    # --- Leading Eigenvector ---
    partition_eigen = G_ig_HT.community_leading_eigenvector(
        clusters=num_clusters, weights=[e['weight'] for e in G_ig_HT.es['weight']]
    )
    community_dict_eigen = {vertex_names_HT[v]: cid for cid, comm in enumerate(partition_eigen) for v in comm}
    predicted_labels = [community_dict_eigen[node] for node in node_list]
    clustering_performance = store_metrics(clustering_performance, 'Eigenvector', true_labels, predicted_labels)
    
    # --- Multilevel ---
    partition_multilevel = G_ig_HT.community_multilevel(
        weights=[e['weight'] for e in G_ig_HT.es['weight']], resolution=1
    )
    community_dict_multilevel = {vertex_names_HT[v]: cid for cid, comm in enumerate(partition_multilevel) for v in comm}
    predicted_labels = [community_dict_multilevel[node] for node in node_list]
    clustering_performance = store_metrics(clustering_performance, 'Multilevel', true_labels, predicted_labels)
    
    # =========================================================================
    # Embedding-based Methods
    # =========================================================================
    
    # --- PCA ---
    n_components = 30
    pca = PCA(n_components, random_state=RANDOM_SEED)
    pca_embedding = pca.fit_transform(data.T)
    predicted_labels = list(kmeans.fit_predict(pca_embedding))
    clustering_performance = store_metrics(clustering_performance, 'PCA', true_labels, predicted_labels)
    
    # --- UMAP ---
    umap_reducer = UMAP(n_components=30, random_state=RANDOM_SEED)
    umap_embedding = umap_reducer.fit_transform(data.T)
    predicted_labels = list(kmeans.fit_predict(umap_embedding))
    clustering_performance = store_metrics(clustering_performance, 'UMAP', true_labels, predicted_labels)
    
    # --- tSNE ---
    tsne = TSNE(n_components=30, method='exact', random_state=RANDOM_SEED)
    tsne_embedding = tsne.fit_transform(data.T)
    predicted_labels = list(kmeans.fit_predict(tsne_embedding))
    clustering_performance = store_metrics(clustering_performance, 'tSNE', true_labels, predicted_labels)
    
    # --- Node2Vec ---
    corr = cell_coexp.copy()
    np.fill_diagonal(corr, 0)
    row_sums = corr.sum(axis=1)
    row_sums[row_sums == 0] = 1
    P_n2v = corr / row_sums[:, np.newaxis]
    
    random_walks = generate_random_walks_for_all_nodes(P_n2v, num_walks_per_node, walk_length)
    model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5, min_count=0, sg=1, workers=4, seed=RANDOM_SEED)
    embeddings_n2v = np.array([model.wv[i] for i in range(len(node_list))])
    predicted_labels = list(kmeans.fit_predict(embeddings_n2v))
    clustering_performance = store_metrics(clustering_performance, 'node2vec', true_labels, predicted_labels)
    
    # --- EDHW (Edge-Dependent Hypergraph Walk) ---
    H = data.copy()
    I = Incidence_matrix(H)
    delta_e = degree_edge(H)
    edge_weights = np.sum(H, axis=1)
    W = W_hyperedge(I, edge_weights)
    dv = degree_vertex(W)
    P_edhw = P_original(dv, delta_e, W, H)
    
    random_walks = generate_random_walks_for_all_nodes(P_edhw, num_walks_per_node, walk_length)
    model = Word2Vec(sentences=random_walks, vector_size=walk_length, window=5, min_count=0, sg=1, workers=4, seed=RANDOM_SEED)
    embeddings_edhw = np.array([model.wv[i] for i in range(len(node_list))])
    predicted_labels = list(kmeans.fit_predict(embeddings_edhw))
    clustering_performance = store_metrics(clustering_performance, 'EDHW', true_labels, predicted_labels)
    
    # --- DIPHW ---
    P_diphw = P_adjusted(H, preference_exponent)
    random_walks = generate_random_walks_for_all_nodes(P_diphw, num_walks_per_node, walk_length)
    model = Word2Vec(sentences=random_walks, vector_size=walk_length, window=5, min_count=0, sg=1, workers=4, seed=RANDOM_SEED)
    embeddings_diphw = np.array([model.wv[i] for i in range(len(node_list))])
    predicted_labels = list(kmeans.fit_predict(embeddings_diphw))
    clustering_performance = store_metrics(clustering_performance, f'DIPHW_{preference_exponent}', true_labels, predicted_labels)
    
    # --- CoMem ---
    G_E = gene_coexp.copy()
    G_V = cell_coexp.copy()
    np.fill_diagonal(G_E, 0)
    np.fill_diagonal(G_V, 0)
    Gamma = data.copy()
    
    P_comem = unipartite_transition_probability_vectorized(G_E, G_V, Gamma, 'optimal')
    P_comem_norm = normalize_transition_probability_matrix(P_comem)
    
    random_walks = generate_random_walks_for_all_nodes(P_comem_norm, num_walks_per_node, walk_length)
    model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5, min_count=0, sg=1, workers=4, seed=RANDOM_SEED)
    embeddings_comem = np.array([model.wv[i] for i in range(len(node_list))])
    predicted_labels = list(kmeans.fit_predict(embeddings_comem))
    clustering_performance = store_metrics(clustering_performance, 'CoMem', true_labels, predicted_labels)
    
    # --- CoMem-DIPHW ---
    P_comem_diphw = unipartite_transition_probability_adjusted_vectorized(G_E, G_V, Gamma, preference_exponent, 'optimal')
    P_comem_diphw_norm = normalize_transition_probability_matrix(P_comem_diphw)
    
    random_walks = generate_random_walks_for_all_nodes(P_comem_diphw_norm, num_walks_per_node, walk_length)
    model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5, min_count=0, sg=1, workers=4, seed=RANDOM_SEED)
    embeddings_cd = np.array([model.wv[i] for i in range(len(node_list))])
    predicted_labels = list(kmeans.fit_predict(embeddings_cd))
    clustering_performance = store_metrics(clustering_performance, f'CoMem-DIPHW_{preference_exponent}', true_labels, predicted_labels)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    results = {
        'run_idx': run_idx,
        'rows': rows,
        'cols': cols,
        'num_modules': num_modules,
        'avg_genes_per_module': avg_genes_per_module,
        'avg_cells_per_module': avg_cells_per_module,
        'target_density': target_density,
        'module_density': module_density,
        'inter_module_density': inter_module_density,
        'inter_module_connection_probability': inter_module_connection_probability,
        'lambda_background': lambda_background,
        'lambda_module': lambda_module,
        'inter_module_lambda': inter_module_lambda
    }
    
    # Add all clustering metrics
    for key, value in clustering_performance.items():
        results[key] = value
    
    # Write to CSV
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
    
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writerow(results)
    
    print(f"\nResults saved to {filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    methods = ['Louvain', 'Leiden', 'GreedyModularity', 'Louvain_HT', 'Infomap',
               'Eigenvector', 'Multilevel', 'PCA', 'UMAP', 'tSNE', 'node2vec',
               'EDHW', f'DIPHW_{preference_exponent}', 'CoMem', f'CoMem-DIPHW_{preference_exponent}']
    
    print(f"{'Method':<25} {'ACC':>8} {'NMI':>8} {'ARI':>8} {'AMI':>8} {'F1':>8}")
    print("-" * 70)
    for method in methods:
        acc = clustering_performance.get(f'{method}_ACC', 0)
        nmi = clustering_performance.get(f'{method}_NMI', 0)
        ari = clustering_performance.get(f'{method}_ARI', 0)
        ami = clustering_performance.get(f'{method}_AMI', 0)
        f1 = clustering_performance.get(f'{method}_F1', 0)
        print(f"{method:<25} {acc:>8.4f} {nmi:>8.4f} {ari:>8.4f} {ami:>8.4f} {f1:>8.4f}")
