#!/usr/bin/env python
# coding: utf-8

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
import networkx as nx
import igraph as ig
import community
import leidenalg as la
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import itertools
import csv
from umap import UMAP
import time
from cdlib import algorithms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import scanpy as sc
import pickle
import umap

parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)
from functions import *

from Simulation_functions import *
from CoMemDIPHW import *
from DIPHW import *
from plot_functions import *


parser = argparse.ArgumentParser(description='Process clustering parameters.')
parser.add_argument('--n_top_genes', type=int, help='Number of top genes')
parser.add_argument('--n_clusters', type=int, help='Number of clusters')
parser.add_argument('--preference_exponent', type=int, help='Preference exponent')
parser.add_argument('--walk_length', type=int, help='Walk length')
parser.add_argument('--num_walks_per_node', type=int, help='Number of Walks')
parser.add_argument('--embedded_dimensions', type=int, help='Embedded Dimensions')
parser.add_argument('--n_components_PCA', type=int, help='N components PCA')

args = parser.parse_args()

n_top_genes = args.n_top_genes
n_clusters = args.n_clusters
preference_exponent = args.preference_exponent
walk_length = args.walk_length
num_walks_per_node = args.num_walks_per_node
embedded_dimensions = args.embedded_dimensions
n_components_PCA = args.n_components_PCA

seq_data_name = 'human_brain_fetal'

file_suffix = '_revised' 

human_brain = pd.read_csv('data/GSE75140_hOrg.fetal.master.data.frame.txt', delimiter='\t', index_col=0)

human_brain = human_brain.dropna()
print("Any NA values:", human_brain.isna().any().any())

counts = human_brain.drop(human_brain.columns[-1], axis=1).T
print("Counts shape:", counts.shape)


# =============================================================================
# PREPROCESSING
# =============================================================================
# step1. normalise data by cells to make sure total reads by cells are the same
# step2. compute variance by genes
# step3. get high variance genes and filter out low variance genes in the original count data
# step4. log transform reduce i. skewness ii. reduce variance iii. mitigate outliers
# step5. Filter out zero expression cells or genes

# step 1
data_normalized_by_cells = normalize_matrix_cpm(counts).astype(float)

preprocessed_data = np.log(data_normalized_by_cells + 1)

# Remove rows that sum to zero
preprocessed_data = preprocessed_data.loc[preprocessed_data.sum(axis=1) != 0]

# Remove columns that sum to zero
preprocessed_data = preprocessed_data.loc[:, preprocessed_data.sum(axis=0) != 0]

print('Data shape after removing zero sums:', preprocessed_data.shape)

nrows, ncols = preprocessed_data.shape

adata = sc.AnnData(preprocessed_data.T)

# Identify highly variable genes using the Seurat
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes)
# sc.pl.highly_variable_genes(adata) 


# Filter the data to keep only highly variable genes for downstream analysis
adata = adata[:, adata.var['highly_variable']]
seurat_preprocessed_data = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names).T

print("Seurat preprocessed shape:", seurat_preprocessed_data.shape)
print('number of zero expression cells,', sum(np.sum(seurat_preprocessed_data) == 0))
print('number of zero expression genes,', sum(np.sum(seurat_preprocessed_data, axis=1) == 0))

zero_expression_cells = np.sum(seurat_preprocessed_data, axis=0) == 0
preprocessed_data = seurat_preprocessed_data.loc[:, ~zero_expression_cells]
zero_expression_genes = np.sum(seurat_preprocessed_data, axis=1) == 0
preprocessed_data = preprocessed_data.loc[~zero_expression_genes, :]
print("Final preprocessed data shape:", preprocessed_data.shape)


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================
correlation_threshold_percentile = 99
cell_coexpression_weighted_HT, cell_coexpression_weighted = create_correlation_network(
    preprocessed_data, correlation_threshold_percentile)
gene_coexpression_weighted_HT, gene_coexpression_weighted = create_correlation_network(
    preprocessed_data.T, correlation_threshold_percentile)

node_list = list(preprocessed_data.columns)
print('Node list length:', len(node_list))

G_cell_weighted = nx.from_numpy_array(cell_coexpression_weighted)
G_cell_weighted_HT = nx.from_numpy_array(cell_coexpression_weighted_HT)

print("Cell Weighted Shape:", cell_coexpression_weighted.shape)
print("Gene Weighted Shape:", gene_coexpression_weighted.shape)

cell_map = {i: j for i, j in enumerate(node_list)}
G_cell_weighted = nx.relabel_nodes(G_cell_weighted, cell_map)
G_cell_weighted_HT = nx.relabel_nodes(G_cell_weighted_HT, cell_map)

print("Nodes in G_cell_weighted:", len(G_cell_weighted.nodes))

density = nx.density(G_cell_weighted_HT)
print("Density of the weighted graph after HT:", density)

density = nx.density(G_cell_weighted)
print("Density of the weighted graph without HT:", density)

non_zero_weights = [data['weight'] for _, _, data in G_cell_weighted_HT.edges(data=True) if data['weight'] > 0]

if non_zero_weights:
    min_weight = min(non_zero_weights)
    print("Minimum non-zero edge weight:", min_weight)

data = preprocessed_data


def normalize_sparsematrix_cpm(matrix):
    """Normalize the matrix using Counts Per Million (CPM) normalization."""
    dense_matrix = matrix.toarray()
    column_sums = np.sum(dense_matrix, axis=0)
    column_sums[column_sums == 0] = 1
    normalized_matrix = 1e6 * dense_matrix / column_sums
    return sp.csr_matrix(normalized_matrix)


def normalize_matrix_cpm(matrix):
    """Normalize the matrix using Counts Per Million (CPM) normalization."""
    matrix = matrix.copy()
    column_sums = np.sum(matrix, axis=0)
    column_sums[column_sums == 0] = 1
    normalized_matrix = 1e6 * matrix / column_sums
    return normalized_matrix


def filter_zero_expression_community(expression_df_with_CommunityAssignment, thres=0):
    """Filter out genes (rows) and cells (columns) with zero expression."""
    df = expression_df_with_CommunityAssignment
    last_row = df.iloc[-1, :]
    row_sums = df.iloc[:-1, :].sum(axis=1)
    col_sums = df.iloc[:-1, :].sum(axis=0)
    filtered_genes = row_sums > thres
    filtered_df = df.iloc[:-1, :].loc[filtered_genes, :]
    filtered_cells = col_sums > thres
    filtered_df = filtered_df.loc[:, filtered_cells]
    last_row_filtered = last_row.loc[filtered_cells]
    filtered_df = filtered_df.append(last_row_filtered)
    return filtered_df


def save_labels(labels, method, seq_data_name, preference_exponent, n_top_genes,
                num_walks_per_node, walk_length, embedded_dimensions, n_clusters):
    """Save labels to pickle file."""
    # Uses global file_suffix
    labels_filename = (
        f"labels/{seq_data_name}_{method}{file_suffix}_prefexp{preference_exponent}_"
        f"topgenes{n_top_genes}_walks{num_walks_per_node}_length{walk_length}_"
        f"dims{embedded_dimensions}_clusters{n_clusters}.pkl"
    )
    os.makedirs('labels', exist_ok=True)
    with open(labels_filename, 'wb') as f:
        pickle.dump(labels, f)
    print(f"Labels saved: {labels_filename}")


def save_embeddings(embeddings, method, seq_data_name, preference_exponent, n_top_genes,
                    num_walks_per_node, walk_length, embedded_dimensions, n_clusters):
    """Save embeddings to pickle file."""
    # Uses global file_suffix
    filename = (
        f"embeddings/{seq_data_name}_{method}{file_suffix}_prefexp{preference_exponent}_"
        f"topgenes{n_top_genes}_walks{num_walks_per_node}_length{walk_length}_"
        f"dims{embedded_dimensions}_clusters{n_clusters}.pkl"
    )
    os.makedirs('embeddings', exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved: {filename}")


# =============================================================================
# LOUVAIN (weighted, no HT)
# =============================================================================
G = G_cell_weighted
partition_louvain = community.best_partition(G, weight='weight')
size_threshold = 5
community_sizes = Counter(partition_louvain.values())

filtered_communities = {node: community_id for node, community_id in partition_louvain.items()
                        if community_sizes[community_id] >= size_threshold}
organized_communities = defaultdict(list)
for node, community_id in filtered_communities.items():
    organized_communities[community_id].append(node)
organized_communities = dict(organized_communities)


# =============================================================================
# LEIDEN
# =============================================================================
G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True)
G_ig.vs['name'] = list(G.nodes)

partition_leiden = la.find_partition(
    G_ig,
    la.RBConfigurationVertexPartition,
    weights=[i['weight'] for i in G_ig.es['weight']], n_iterations=-1)

community_dict_leiden = {name: membership for name, membership in zip(G_ig.vs['name'], partition_leiden.membership)}
sorted_dict = {k: community_dict_leiden[k] for k in node_list}
predicted_labels_leiden = list(sorted_dict.values())

method = 'leiden'
save_labels(predicted_labels_leiden, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# GREEDY MODULARITY
# =============================================================================
partition_greedy_modularity = nx.community.greedy_modularity_communities(G, weight='weight')
community_dict_gm = {i: n for n, c in enumerate(partition_greedy_modularity) for i in c}
sorted_dict = {k: community_dict_gm[k] for k in node_list}
predicted_labels_greedy_modularity = list(sorted_dict.values())

method = 'greedy_modularity'
save_labels(predicted_labels_greedy_modularity, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# LOUVAIN HT
# =============================================================================
G = G_cell_weighted_HT

partition = community.best_partition(G, weight='weight')
size_threshold = 5
community_sizes = Counter(partition.values())
filtered_communities = {node: community_id for node, community_id in partition.items()
                        if community_sizes[community_id] >= size_threshold}

organized_communities = defaultdict(list)
for node, community_id in filtered_communities.items():
    organized_communities[community_id].append(node)
organized_communities = dict(organized_communities)

community_dict = partition
sorted_dict = {k: community_dict[k] for k in node_list}
predicted_labels_louvain_HT = list(sorted_dict.values())

method = 'louvain_HT'
save_labels(predicted_labels_louvain_HT, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# INFOMAP
# =============================================================================
G = G_cell_weighted_HT
G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True)
partition = G_ig.community_infomap(edge_weights=[i['weight'] for i in G_ig.es['weight']])
community_dict_infomap = {name: membership for name, membership in zip(G_ig.vs['name'], partition.membership)}

sorted_dict = {k: community_dict_infomap[k] for k in node_list}
predicted_labels_Infomap = list(sorted_dict.values())

method = 'Infomap'
save_labels(predicted_labels_Infomap, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# MULTILEVEL
# =============================================================================
G = G_cell_weighted_HT
G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True)
partition = G_ig.community_multilevel(weights=[i['weight'] for i in G_ig.es['weight']])
community_dict_multilevel = {name: membership for name, membership in zip(G_ig.vs['name'], partition.membership)}
sorted_dict = {k: community_dict_multilevel[k] for k in node_list}
predicted_labels_multilevel = list(sorted_dict.values())

method = 'multilevel'
save_labels(predicted_labels_multilevel, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# PCA
# =============================================================================
method = 'PCA'

pca = PCA(n_components_PCA)
pca_embedding_counts = pca.fit_transform(data.T)
print("PCA embedding shape:", pca_embedding_counts.shape)

embeddings = dict(zip(node_list, pca_embedding_counts))
save_embeddings(embeddings, method, seq_data_name, preference_exponent,
                n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)

node_embeddings = np.array([list(embeddings[node]) for node_index, node in enumerate(node_list)])
UMAP_encoder = umap.UMAP()
X_UMAP_encoded = UMAP_encoder.fit_transform(node_embeddings)
print("UMAP shape:", np.shape(X_UMAP_encoded))

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_UMAP_encoded)

labels = kmeans.labels_
predicted_labels_PCA = list(labels)

save_labels(predicted_labels_PCA, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# NODE2VEC
# =============================================================================
method = 'node2vec'

corr = cell_coexpression_weighted.copy()
np.fill_diagonal(corr, 0)
row_sums = corr.sum(axis=1)
P = corr / row_sums[:, np.newaxis]

random_walks = generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)

model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5,
                 min_count=0, sg=1, workers=num_walks_per_node)

embeddings = {node_name: model.wv[node_id] for node_id, node_name in enumerate(node_list)}
save_embeddings(embeddings, method, seq_data_name, preference_exponent,
                n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)

embedding = np.array(list(embeddings.values()))

UMAP_encoder = umap.UMAP()
X_UMAP_encoded = UMAP_encoder.fit_transform(embedding)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_UMAP_encoded)

labels = kmeans.labels_
predicted_labels_node2vec = list(labels)

save_labels(predicted_labels_node2vec, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# EDVW (Edge Dependent Vertex Weighted Hypergraph Random Walk)
# =============================================================================
method = 'EDVW'

data = np.array(data)
H = data.copy()
I = Incidence_matrix(H)
delta_e = degree_edge(H)
edge_weights = np.sum(H, axis=1)
W = W_hyperedge(I, edge_weights)
dv = degree_vertex(W)
P = P_original(dv, delta_e, W, H)

random_walks = generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)

# Use embedded_dimensions for vector_size
model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5,
                 min_count=0, sg=1, workers=num_walks_per_node)

print('Method:', method)
print('Embedded dimensions:', embedded_dimensions)

embeddings = {node_name: model.wv[node_id] for node_id, node_name in enumerate(node_list)}
save_embeddings(embeddings, method, seq_data_name, preference_exponent,
                n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)

node_list = list(embeddings.keys())
node_embeddings = np.array([list(embeddings[node_name]) for node_id, node_name in enumerate(node_list)])

UMAP_encoder = umap.UMAP()
X_UMAP_encoded = UMAP_encoder.fit_transform(node_embeddings)
print("UMAP shape (EDVW):", np.shape(X_UMAP_encoded))

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_UMAP_encoded)

labels = kmeans.labels_
predicted_labels_EDVW = list(labels)

save_labels(predicted_labels_EDVW, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# DIPHW 
# =============================================================================
method = 'DIPHW'

H = data.copy()
P = P_adjusted(H, preference_exponent)

random_walks = generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)

model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5,
                 min_count=0, sg=1, workers=num_walks_per_node)

print('Method:', method)
print('Embedded dimensions:', embedded_dimensions)

embeddings = {node_name: model.wv[node_id] for node_id, node_name in enumerate(node_list)}
save_embeddings(embeddings, method, seq_data_name, preference_exponent,
                n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)

node_list = list(embeddings.keys())
node_embeddings = np.array([list(embeddings[node_name]) for node_id, node_name in enumerate(node_list)])

UMAP_encoder = umap.UMAP()
X_UMAP_encoded = UMAP_encoder.fit_transform(node_embeddings)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_UMAP_encoded)

labels = kmeans.labels_
predicted_labels_DIPHW = list(labels)

save_labels(predicted_labels_DIPHW, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# COMEM
# =============================================================================
HT = False
laziness = 'non-lazy'

if HT:
    G_E = gene_coexpression_weighted_HT
    G_V = cell_coexpression_weighted_HT
    method = 'CoMem_HT'
else:
    G_E = gene_coexpression_weighted
    G_V = cell_coexpression_weighted
    method = 'CoMem'

if laziness == 'non-lazy':
    np.fill_diagonal(G_E, 0)
    np.fill_diagonal(G_V, 0)

Gamma = data.copy()
P = unipartite_transition_probability_vectorized(G_E, G_V, Gamma, 'optimal')
P_vw_normalised = normalize_transition_probability_matrix(P)

random_walks = generate_random_walks_for_all_nodes(P_vw_normalised, num_walks_per_node, walk_length)

model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5,
                 min_count=0, sg=1, workers=num_walks_per_node)

embeddings = {node_name: model.wv[node_id] for node_id, node_name in enumerate(node_list)}
save_embeddings(embeddings, method, seq_data_name, preference_exponent,
                n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)

node_list = list(embeddings.keys())
node_embeddings = np.array([list(embeddings[node_name]) for node_id, node_name in enumerate(node_list)])

UMAP_encoder = umap.UMAP()
X_UMAP_encoded = UMAP_encoder.fit_transform(node_embeddings)
print("UMAP shape (CoMem):", np.shape(X_UMAP_encoded))

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_UMAP_encoded)

labels = kmeans.labels_
predicted_labels_CoMem = list(labels)

save_labels(predicted_labels_CoMem, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# COMEM-DIPHW
# =============================================================================
method = 'CoMem_DIPHW'

P = unipartite_transition_probability_adjusted_vectorized(G_E, G_V, Gamma, preference_exponent, 'optimal')
P_vw_normalised = normalize_transition_probability_matrix(P)

random_walks = generate_random_walks_for_all_nodes(P_vw_normalised, num_walks_per_node, walk_length)

model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5,
                 min_count=0, sg=1, workers=num_walks_per_node)

embeddings = {node_name: model.wv[node_id] for node_id, node_name in enumerate(node_list)}
save_embeddings(embeddings, method, seq_data_name, preference_exponent,
                n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)

node_list = list(embeddings.keys())
node_embeddings = np.array([list(embeddings[node_name]) for node_id, node_name in enumerate(node_list)])

UMAP_encoder = umap.UMAP()
X_UMAP_encoded = UMAP_encoder.fit_transform(node_embeddings)
print("UMAP shape (CoMem-DIPHW):", np.shape(X_UMAP_encoded))

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_UMAP_encoded)

labels = kmeans.labels_
predicted_labels_CoMem_DIPHW = list(labels)

save_labels(predicted_labels_CoMem_DIPHW, method, seq_data_name, preference_exponent,
            n_top_genes, num_walks_per_node, walk_length, embedded_dimensions, n_clusters)


# =============================================================================
# EVALUATION
# =============================================================================
methods = ['leiden', 'greedy_modularity', 'louvain_HT', 'Infomap', 'multilevel',
           'PCA', 'EDVW', 'node2vec', 'DIPHW', 'CoMem', 'CoMem_DIPHW']

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

X = preprocessed_data.T

results = {
    'method': [],
    'silhouette_score': [],
    'calinski_harabasz': [],
    'inverse_wcss': []
}

for method in methods:
    labels = eval(f"predicted_labels_{method}")

    # Silhouette Score (higher is better)
    silhouette_avg = silhouette_score(X, labels)

    # Calinski-Harabasz Index (higher is better)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    # Within-Cluster Sum of Squares (WCSS) (lower is better, so we take the inverse)
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    wcss = kmeans.inertia_
    inv_wcss = 1 / wcss if wcss != 0 else float('inf')

    results['method'].append(method)
    results['silhouette_score'].append(silhouette_avg)
    results['calinski_harabasz'].append(calinski_harabasz)
    results['inverse_wcss'].append(inv_wcss)


# =============================================================================
# VISUALIZATION
# =============================================================================
from sklearn.preprocessing import MinMaxScaler

methods_list = results['method']
silhouette_scores = results['silhouette_score']
calinski_harabasz_scores = results['calinski_harabasz']
inv_wcss_scores = results['inverse_wcss']

metrics = np.array([silhouette_scores, calinski_harabasz_scores, inv_wcss_scores])

scaler = MinMaxScaler()
normalized_metrics = scaler.fit_transform(metrics.T).T

barWidth = 0.2
fig, ax = plt.subplots(figsize=(12, 8))

r1 = np.arange(len(methods_list))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

colors = sns.color_palette(palette='Dark2')
plt.bar(r1, normalized_metrics[0], color=colors[0], width=barWidth, edgecolor='grey', label='Silhouette Score')
plt.bar(r2, normalized_metrics[1], color=colors[1], width=barWidth, edgecolor='grey', label='Calinski-Harabasz Index')
plt.bar(r3, normalized_metrics[2], color=colors[2], width=barWidth, edgecolor='grey', label='Inverse WCSS')

plt.xlabel('Clustering Methods', fontweight='bold', fontsize=14)
plt.ylabel('Normalized Metric Scores', fontweight='bold', fontsize=14)
plt.xticks([r + barWidth * 1.5 for r in range(len(methods_list))], methods_list, rotation=45, ha='right', fontsize=12)

words = seq_data_name.split('_')
capitalized_words = [word.capitalize() for word in words]
formatted_name = ' '.join(capitalized_words)

plt.title(formatted_name + f" {file_suffix}", fontweight='bold', fontsize=16)

plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.6)
plt.tight_layout(pad=2.0)

ax.tick_params(axis='y', which='major', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
plt.minorticks_on()
ax.grid(True, which='minor', axis='y', linestyle=':', linewidth='0.5', alpha=0.7)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, frameon=False)

os.makedirs('fig', exist_ok=True)
path = f"fig/{seq_data_name}{file_suffix}_{n_top_genes}Ntopgenes_{n_clusters}Nclusters_{preference_exponent}prefexpo_{walk_length}WalkLength_ClusteringPerformance.pdf"
plt.savefig(path, bbox_inches='tight')
print(f"Figure saved: {path}")


# =============================================================================
# SAVE RESULTS WITH CELL IDs
# =============================================================================
base_dir = "results/clustering_results/"
os.makedirs(base_dir, exist_ok=True)

methods = ['leiden', 'greedy_modularity', 'louvain_HT', 'Infomap', 'multilevel',
           'PCA', 'EDVW', 'node2vec', 'DIPHW', 'CoMem', 'CoMem_DIPHW']

filename = (
    f"clustering_results_{seq_data_name}{file_suffix}_prefexp{preference_exponent}_topgenes{n_top_genes}_"
    f"walks{num_walks_per_node}_length{walk_length}_dims{embedded_dimensions}_"
    f"clusters{n_clusters}.csv"
)

save_path = os.path.join(base_dir, filename)

# Build results dict (method -> labels only)
results = {}
for method in methods:
    labels = eval(f"predicted_labels_{method}")
    results[method] = labels

# Create DataFrame and insert cell_id as first column
df = pd.DataFrame(results)
df.insert(0, 'cell_id', node_list)

df.to_csv(save_path, index=False)
print(f"Results saved to: {save_path}")


# =============================================================================
# VERIFICATION
# =============================================================================


n_cells = len(node_list)
for method in methods:
    labels = eval(f"predicted_labels_{method}")
    if len(labels) != n_cells:
        print(f"WARNING: {method} has {len(labels)} labels but expected {n_cells}")
    else:
        print(f"{method:20s}: OK ({len(set(labels))} clusters)")

print(f"\nSpot check - first 3 cells:")
for i in range(min(3, n_cells)):
    print(f"  {node_list[i]}: PCA={predicted_labels_PCA[i]}, "
          f"DIPHW={predicted_labels_DIPHW[i]}, CoMem={predicted_labels_CoMem[i]}")