#!/usr/bin/env python
# coding: utf-8
# Single run benchmark script for SLURM array jobs

import argparse
import os
import sys
import numpy as np
import pandas as pd
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
from scipy.optimize import linear_sum_assignment
import umap
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)
from functions import *
from Simulation_functions import *
from CoMemDIPHW import *
from DIPHW import *
from plot_functions import *


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
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    f1 = compute_f1_score(y_true, y_pred)
    return acc, nmi, ari, ami, f1


DATASET_CONFIG = {
    '3cl': {
        'seq_data_name': 'benchmark3cl',
        'counts_path': 'data/sce_sc_10x_qc_counts_matrix.csv',
        'groundtruth_path': 'data/groundtruth.csv'
    },
    '5cl': {
        'seq_data_name': 'benchmark5cl',
        'counts_path': 'data/sce_sc_10x_5cl_qc_counts_matrix.csv',
        'groundtruth_path': 'data/groundtruth_5cl.csv'
    }
}

# All other parameters (n_top_genes, n_clusters, etc.) come from command line arguments

METHODS = ['louvain', 'leiden', 'greedy_modularity', 'louvain_HT', 'Infomap',
           'multilevel', 'PCA', 'tSNE', 'UMAP', 'EDVW', 'node2vec', 'DIPHW',
           'CoMem', 'CoMem_DIPHW']


# =============================================================================
# Data Loading
# =============================================================================
def load_and_preprocess_data(config, n_top_genes):
    """Load and preprocess dataset"""
    df = pd.read_csv(config['counts_path'], index_col=0)
    groundtruth = pd.read_csv(config['groundtruth_path'])
    
    df = df.dropna()
    counts = df.copy()
    
    data_normalized = normalize_matrix_cpm(counts).astype(float)
    preprocessed_data = np.log(data_normalized + 1)
    
    preprocessed_data = preprocessed_data.loc[preprocessed_data.sum(axis=1) != 0]
    preprocessed_data = preprocessed_data.loc[:, preprocessed_data.sum(axis=0) != 0]
    
    adata = sc.AnnData(preprocessed_data.T)
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]
    seurat_preprocessed_data = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names).T
    
    zero_cells = np.sum(seurat_preprocessed_data, axis=0) == 0
    zero_genes = np.sum(seurat_preprocessed_data, axis=1) == 0
    preprocessed_data = seurat_preprocessed_data.loc[~zero_genes, ~zero_cells]
    
    node_list = list(preprocessed_data.columns)
    
    original_cell_order = list(counts.columns)
    groundtruth_labels = groundtruth.iloc[:, 1].tolist()
    
    min_len = min(len(original_cell_order), len(groundtruth_labels))
    groundtruth_dict = {original_cell_order[i]: groundtruth_labels[i] for i in range(min_len)}
    true_labels = [groundtruth_dict[node] for node in node_list]
    
    return preprocessed_data, node_list, true_labels


# =============================================================================
# Single Run
# =============================================================================
def run_single_experiment(args):
    """Run a single experiment for all methods"""
    
    config = DATASET_CONFIG[args.dataset]
    seq_data_name = config['seq_data_name']
    
    # Set random seed based on run index
    seed = 42 + args.run
    np.random.seed(seed)
    
    n_clusters = args.n_clusters
    n_top_genes = args.n_top_genes
    preference_exponent = args.preference_exponent
    walk_length = args.walk_length
    num_walks_per_node = args.num_walks_per_node
    embedded_dimensions = args.embedded_dimensions
    n_components_PCA = args.n_components_PCA
    
    print(f"Loading data for {args.dataset}...")
    preprocessed_data, node_list, true_labels = load_and_preprocess_data(config, n_top_genes)
    data_array = np.array(preprocessed_data)
    print(f"Data shape: {preprocessed_data.shape}")
    
    # Create correlation networks
    print("Creating correlation networks...")
    correlation_threshold_percentile = 99
    cell_coexp_HT, cell_coexp = create_correlation_network(preprocessed_data, correlation_threshold_percentile)
    gene_coexp_HT, gene_coexp = create_correlation_network(preprocessed_data.T, correlation_threshold_percentile)
    
    cell_map = {i: j for i, j in enumerate(node_list)}
    G_cell = nx.from_numpy_array(cell_coexp)
    G_cell_HT = nx.from_numpy_array(cell_coexp_HT)
    G_cell = nx.relabel_nodes(G_cell, cell_map)
    G_cell_HT = nx.relabel_nodes(G_cell_HT, cell_map)
    
    predicted_labels = {}
    
    # --- Graph-based Methods ---
    
    partition_louvain = community.best_partition(G_cell, weight='weight', random_state=seed)
    predicted_labels['louvain'] = [partition_louvain[node] for node in node_list]
    
    G_ig = ig.Graph.TupleList(G_cell.edges(data=True), weights=True)
    G_ig.vs['name'] = list(G_cell.nodes)
    partition_leiden = la.find_partition(
        G_ig, la.RBConfigurationVertexPartition,
        weights=[e['weight'] for e in G_ig.es['weight']], n_iterations=-1, seed=seed
    )
    community_dict_leiden = {name: mem for name, mem in zip(G_ig.vs['name'], partition_leiden.membership)}
    predicted_labels['leiden'] = [community_dict_leiden[node] for node in node_list]
    
    partition_gm = nx.community.greedy_modularity_communities(G_cell, weight='weight')
    community_dict_gm = {node: idx for idx, comm in enumerate(partition_gm) for node in comm}
    predicted_labels['greedy_modularity'] = [community_dict_gm[node] for node in node_list]
    
    partition_louvain_HT = community.best_partition(G_cell_HT, weight='weight', random_state=seed)
    predicted_labels['louvain_HT'] = [partition_louvain_HT[node] for node in node_list]
    
    G_ig_HT = ig.Graph.TupleList(G_cell_HT.edges(data=True), weights=True)
    partition_infomap = G_ig_HT.community_infomap(edge_weights=[e['weight'] for e in G_ig_HT.es['weight']])
    community_dict_infomap = {name: mem for name, mem in zip(G_ig_HT.vs['name'], partition_infomap.membership)}
    predicted_labels['Infomap'] = [community_dict_infomap[node] for node in node_list]
    
    partition_multilevel = G_ig_HT.community_multilevel(weights=[e['weight'] for e in G_ig_HT.es['weight']])
    community_dict_multilevel = {name: mem for name, mem in zip(G_ig_HT.vs['name'], partition_multilevel.membership)}
    predicted_labels['multilevel'] = [community_dict_multilevel[node] for node in node_list]
    
    # --- Embedding-based Methods ---
    
    pca = PCA(n_components=n_components_PCA, random_state=seed)
    pca_embedding = pca.fit_transform(preprocessed_data.T)
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['PCA'] = list(kmeans_pca.fit_predict(pca_embedding))
    
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
    tsne_embedding = tsne.fit_transform(preprocessed_data.T)
    kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['tSNE'] = list(kmeans_tsne.fit_predict(tsne_embedding))
    
    umap_reducer = umap.UMAP(n_components=n_components_PCA, random_state=seed)
    umap_embedding = umap_reducer.fit_transform(preprocessed_data.T)
    kmeans_umap = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['UMAP'] = list(kmeans_umap.fit_predict(umap_embedding))
    
    print("Running random walk methods...")
    
    corr = cell_coexp.copy()
    np.fill_diagonal(corr, 0)
    row_sums = corr.sum(axis=1)
    row_sums[row_sums == 0] = 1
    P_n2v = corr / row_sums[:, np.newaxis]
    
    random_walks_n2v = generate_random_walks_for_all_nodes(P_n2v, num_walks_per_node, walk_length)
    model_n2v = Word2Vec(sentences=random_walks_n2v, vector_size=embedded_dimensions,
                         window=5, min_count=0, sg=1, workers=4, seed=seed)
    embeddings_n2v = np.array([model_n2v.wv[i] for i in range(len(node_list))])
    kmeans_n2v = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['node2vec'] = list(kmeans_n2v.fit_predict(embeddings_n2v))
    
    H = data_array.copy()
    I = Incidence_matrix(H)
    delta_e = degree_edge(H)
    edge_weights_hyper = np.sum(H, axis=1)
    W = W_hyperedge(I, edge_weights_hyper)
    dv = degree_vertex(W)
    P_edvw = P_original(dv, delta_e, W, H)
    
    random_walks_edvw = generate_random_walks_for_all_nodes(P_edvw, num_walks_per_node, walk_length)
    model_edvw = Word2Vec(sentences=random_walks_edvw, vector_size=embedded_dimensions,
                          window=5, min_count=0, sg=1, workers=4, seed=seed)
    embeddings_edvw = np.array([model_edvw.wv[i] for i in range(len(node_list))])
    kmeans_edvw = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['EDVW'] = list(kmeans_edvw.fit_predict(embeddings_edvw))
    
    P_diphw = P_adjusted(H, preference_exponent)
    random_walks_diphw = generate_random_walks_for_all_nodes(P_diphw, num_walks_per_node, walk_length)
    model_diphw = Word2Vec(sentences=random_walks_diphw, vector_size=embedded_dimensions,
                           window=5, min_count=0, sg=1, workers=4, seed=seed)
    embeddings_diphw = np.array([model_diphw.wv[i] for i in range(len(node_list))])
    kmeans_diphw = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['DIPHW'] = list(kmeans_diphw.fit_predict(embeddings_diphw))
    
    G_E = gene_coexp.copy()
    G_V = cell_coexp.copy()
    np.fill_diagonal(G_E, 0)
    np.fill_diagonal(G_V, 0)
    Gamma = data_array.copy()
    
    P_comem = unipartite_transition_probability_vectorized(G_E, G_V, Gamma, 'optimal')
    P_comem_norm = normalize_transition_probability_matrix(P_comem)
    
    random_walks_comem = generate_random_walks_for_all_nodes(P_comem_norm, num_walks_per_node, walk_length)
    model_comem = Word2Vec(sentences=random_walks_comem, vector_size=embedded_dimensions,
                           window=5, min_count=0, sg=1, workers=4, seed=seed)
    embeddings_comem = np.array([model_comem.wv[i] for i in range(len(node_list))])
    kmeans_comem = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['CoMem'] = list(kmeans_comem.fit_predict(embeddings_comem))
    
    P_comem_diphw = unipartite_transition_probability_adjusted_vectorized(G_E, G_V, Gamma, preference_exponent, 'optimal')
    P_comem_diphw_norm = normalize_transition_probability_matrix(P_comem_diphw)
    
    random_walks_cd = generate_random_walks_for_all_nodes(P_comem_diphw_norm, num_walks_per_node, walk_length)
    model_cd = Word2Vec(sentences=random_walks_cd, vector_size=embedded_dimensions,
                        window=5, min_count=0, sg=1, workers=4, seed=seed)
    embeddings_cd = np.array([model_cd.wv[i] for i in range(len(node_list))])
    kmeans_cd = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted_labels['CoMem_DIPHW'] = list(kmeans_cd.fit_predict(embeddings_cd))
    
    rows = []
    for method in METHODS:
        acc, nmi, ari, ami, f1 = compute_all_metrics(true_labels, predicted_labels[method])
        rows.append({
            'dataset': args.dataset,
            'run': args.run,
            'method': method,
            'ACC': acc,
            'NMI': nmi,
            'ARI': ari,
            'AMI': ami,
            'F1': f1
        })
        print(f"  {method}: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, AMI={ami:.4f}, F1={f1:.4f}")
    
    return pd.DataFrame(rows), predicted_labels, node_list, true_labels, seq_data_name


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single benchmark experiment')
    parser.add_argument('--dataset', type=str, required=True, choices=['3cl', '5cl'],
                        help='Dataset: 3cl or 5cl')
    parser.add_argument('--run', type=int, required=True,
                        help='Run index (0-4)')
    parser.add_argument('--n_top_genes', type=int, required=True,
                        help='Number of top genes')
    parser.add_argument('--n_clusters', type=int, required=True,
                        help='Number of clusters')
    parser.add_argument('--preference_exponent', type=int, required=True,
                        help='Preference exponent')
    parser.add_argument('--walk_length', type=int, required=True,
                        help='Walk length')
    parser.add_argument('--num_walks_per_node', type=int, required=True,
                        help='Number of walks per node')
    parser.add_argument('--embedded_dimensions', type=int, required=True,
                        help='Embedded dimensions')
    parser.add_argument('--n_components_PCA', type=int, required=True,
                        help='Number of PCA components')
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Running benchmark: dataset={args.dataset}, run={args.run}")
    print(f"Random seed: {42 + args.run}")
    print(f"Parameters:")
    print(f"  n_top_genes: {args.n_top_genes}")
    print(f"  n_clusters: {args.n_clusters}")
    print(f"  preference_exponent: {args.preference_exponent}")
    print(f"  walk_length: {args.walk_length}")
    print(f"  num_walks_per_node: {args.num_walks_per_node}")
    print(f"  embedded_dimensions: {args.embedded_dimensions}")
    print(f"  n_components_PCA: {args.n_components_PCA}")
    print(f"{'='*60}")
    
    # Create output directories
    os.makedirs('results/clustering_results', exist_ok=True)
    os.makedirs('fig', exist_ok=True)
    
    # Run experiment
    results_df, predicted_labels, node_list, true_labels, seq_data_name = run_single_experiment(args)
    
    # Save metrics results
    metrics_filename = (f"evaluation_metrics_{seq_data_name}_run{args.run}_"
                        f"prefexp{args.preference_exponent}_topgenes{args.n_top_genes}_"
                        f"walks{args.num_walks_per_node}_length{args.walk_length}_"
                        f"dims{args.embedded_dimensions}_clusters{args.n_clusters}.csv")
    metrics_path = os.path.join('results/clustering_results', metrics_filename)
    results_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save clustering assignments
    clustering_filename = (f"clustering_results_{seq_data_name}_run{args.run}_"
                           f"prefexp{args.preference_exponent}_topgenes{args.n_top_genes}_"
                           f"walks{args.num_walks_per_node}_length{args.walk_length}_"
                           f"dims{args.embedded_dimensions}_clusters{args.n_clusters}.csv")
    clustering_path = os.path.join('results/clustering_results', clustering_filename)
    df_clustering = pd.DataFrame(predicted_labels)
    df_clustering.insert(0, 'cell_id', node_list)
    df_clustering['true_labels'] = true_labels
    df_clustering.to_csv(clustering_path, index=False)
    print(f"Clustering results saved to: {clustering_path}")
    
    print("\nDone!")