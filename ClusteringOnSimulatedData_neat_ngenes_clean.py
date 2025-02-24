#!/usr/bin/env python
# coding: utf-8

# this script does not save or plot the embeddings
# also, since we dont visualise the embeddings, we wont apply tsne or umap to the embeddings. We onlu use kmeans to cluster them to compute NMI and ARI


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


from Simulation_functions import *
from CoMemDIPHW import *
from DIPHW import *

parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)
from functions import *







rows=3000
cols=1000
num_modules=30
avg_genes_per_module = int(sys.argv[1])
avg_cells_per_module=38
target_density= 0.03
module_density=0.3
inter_module_density=0.1
inter_module_connection_probability=0.6
lambda_background=10
lambda_module=20
inter_module_lambda=10



def create_bar_plot(methods, values, title):
    plt.figure(figsize=(10, 6)) 

    # Set the style without grid
    sns.set_style("white")

    bar_width = 0.5  
    bars = plt.bar(methods, values, width=bar_width) 
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Method', fontsize=14, weight='bold')
    plt.ylabel('Value', fontsize=14, weight='bold')
    plt.xticks(rotation=45, fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')

    max_value_index = values.index(max(values))
    plt.text(x=methods[max_value_index], y=max(values) + 0.06, s='Highest', fontdict=dict(color='firebrick', fontsize=12), ha='center')

    for i, bar in enumerate(bars):
        if i == max_value_index:
            bar.set_color('firebrick')
        elif 'DIPHW' in methods[i] or 'CoMem' in methods[i]:
            print(methods[i])
            bar.set_color('darkorange') 
        else:
            bar.set_color('steelblue')
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()

    # Save plot to PDF
    #plt.savefig(f"Fig/{title}.pdf", bbox_inches='tight')
    plt.savefig(f"Fig/ngenes/{title}_{avg_genes_per_module}", bbox_inches='tight',dpi=300)

    plt.show()





# # 1.1 Generate a Simulated Sparse Matrix (count data)
# # 1.2 Normalise the sparse counts data
# # 2.  Correlation-Based Graph Projection




modules = simulate_input_modules(rows, cols, num_modules, avg_genes_per_module,avg_cells_per_module)

print("Generated Modules:")
for i, (start_row, end_row, start_col, end_col) in enumerate(modules):
    print(f"Module {i + 1}: Rows {start_row}-{end_row}, Columns {start_col}-{end_col}")

sparse_data = create_sparse_matrix_with_inter_module_variance(
    rows, cols, modules, target_density, module_density, inter_module_density, 
    inter_module_connection_probability, lambda_background, lambda_module, inter_module_lambda
)




data_df_community = pd.DataFrame.sparse.from_spmatrix(sparse_data.T)
community_assignments=modules_to_community_dict(modules, cols)
data_df_community['community']=list(community_assignments.values())
data_df_community=data_df_community.T
print(sum([i=='NA' for i in list(community_assignments.values())]))




column_names = data_df_community.columns.tolist()

shuffled_column_names = np.random.permutation(column_names)

shuffled_df_community = data_df_community[shuffled_column_names]

shuffled_sparse_data=sp.csr_matrix(shuffled_df_community.iloc[:-1].astype(np.float64))



# # Shuffle the simulated scRNAseq matrix
# # G_weighted_HT is a weighted graph but sparsified by HT
# # G_weighted is a fully-connected weighted graph without sparsification




def normalize_sparsematrix_cpm(matrix):
    """
    Normalize the matrix using Counts Per Million (CPM) normalization.

    :param matrix: A sparse matrix
    :return: Normalized matrix
    """
    dense_matrix = matrix.toarray()
    # Sum across rows (genes) for each column (cell)
    column_sums = np.sum(dense_matrix, axis=0)
    # Avoid division by zero
    column_sums[column_sums == 0] = 1
    # Normalize and multiply by one million
    normalized_matrix = 1e6 * dense_matrix / column_sums
    return sp.csr_matrix(normalized_matrix)



def normalize_matrix_cpm(matrix):
    """
    Normalize the matrix using Counts Per Million (CPM) normalization.

    :param matrix: A sparse matrix
    :return: Normalized matrix
    """
    matrix=matrix.copy()
    column_sums = np.sum(matrix, axis=0)
    # Avoid division by zero
    column_sums[column_sums == 0] = 1
    # Normalize and multiply by one million
    normalized_matrix = 1e6 * matrix / column_sums
    return normalized_matrix

def filter_zero_expression_community(expression_df_with_CommunityAssignment,thres=0):
    """
    Input: expression dataframe with the last row being the community assignment
    Filter out genes (rows) and cells (columns) with zero expression.
    :return: filtered expression dataframe with community assignment
    """
    df=expression_df_with_CommunityAssignment

    last_row = df.iloc[-1, :]

    row_sums = df.iloc[:-1, :].sum(axis=1)
    col_sums = df.iloc[:-1, :].sum(axis=0)

    filtered_genes = row_sums > thres
    filtered_df = df.iloc[:-1, :].loc[filtered_genes, :]

    filtered_cells = col_sums > thres
    filtered_df = filtered_df.loc[:, filtered_cells]
    last_row_filtered=last_row.loc[filtered_cells]
    filtered_df = filtered_df.append(last_row_filtered)

    return filtered_df



def generate_custom_colormap(num_colors):
    """
    Generate a custom colormap with a specified number of colors by combining multiple Sequential Color Brewer palettes.
    """
    colormaps = ['RdPu', 'Purples_d', 'flare','Wistia','Greens_d','Oranges_d','Blues','gray']  
    
    
    #colormaps = ['Spectral'] 
    n_maps=len(colormaps)
    avg_c_per_palette=int(np.floor(num_colors/n_maps))
    residual=num_colors%n_maps
    n_c_per_palette=[avg_c_per_palette for i in range(n_maps)]
    
    for i,j in enumerate(range(residual)):
        n_c_per_palette[i]+=1
        
    
    color_list = []

    while len(color_list) < num_colors:
        for i,cmap in enumerate(colormaps):
            colors = sns.color_palette(cmap, n_colors=n_c_per_palette[i])
            color_list.extend(colors)

    color_list = color_list[:num_colors]

    return color_list





num_clusters = num_modules 
num_colors=num_clusters
color_palette = generate_custom_colormap(num_colors)




non_zero_df = filter_zero_expression_community(shuffled_df_community)
non_zero_data=np.array(non_zero_df[:-1].values)
print(np.shape(non_zero_data))
community_assignments=dict(zip(non_zero_df.columns,non_zero_df.loc['community']))
normalized_data = normalize_matrix_cpm(non_zero_data).astype(float)


logtransform=True


if logtransform==True:
    preprocessed_data=np.log2(normalized_data+1)

else:
    preprocessed_data=normalized_data

preprocessed_df=non_zero_df.copy()
nrows,ncols=preprocessed_df.shape
preprocessed_df.iloc[:-1,:ncols]=preprocessed_data
node_list=list(preprocessed_df.columns)

preprocessed_df




correlation_threshold_percentile=99
cell_coexpression_weighted_HT,cell_coexpression_weighted=create_correlation_network(preprocessed_data, correlation_threshold_percentile)
gene_coexpression_weighted_HT,gene_coexpression_weighted=create_correlation_network(preprocessed_data.T, correlation_threshold_percentile)
G_cell_weighted = nx.from_numpy_array(cell_coexpression_weighted)
G_cell_weighted_HT = nx.from_numpy_array(cell_coexpression_weighted_HT)

print(cell_coexpression_weighted_HT.shape)
print(gene_coexpression_weighted_HT.shape)

cell_map={i:j for i,j in enumerate(list(preprocessed_df.columns))}
G_cell_weighted = nx.relabel_nodes(G_cell_weighted, cell_map)
G_cell_weighted_HT = nx.relabel_nodes(G_cell_weighted_HT, cell_map)

print(len(G_cell_weighted.nodes))

print(list(G_cell_weighted.nodes)==node_list)





density = nx.density(G_cell_weighted_HT)
print("Density of the weighted graph after HT:", density)

density = nx.density(G_cell_weighted)
print("Density of the weighted graph without HT:", density)

non_zero_weights = [data['weight'] for _, _, data in G_cell_weighted_HT.edges(data=True) if data['weight'] > 0]

if non_zero_weights:  
    min_weight = min(non_zero_weights)
    print("Minimum non-zero edge weight:", min_weight)

print(np.sum(np.sum(non_zero_data,axis=0)==1))
print(np.sum(np.sum(non_zero_data,axis=0)==0))
print(np.sum(np.sum(normalized_data,axis=0)==1))
print(np.sum(np.sum(normalized_data,axis=0)==0))
print(np.sum(np.sum(normalized_data,axis=0)))
print(np.sum(np.sum(normalized_data,axis=0)))




# Define the size threshold
size_threshold = 5

community_sizes = Counter(community_assignments.values())

filtered_communities = {node: community_id for node, community_id in community_assignments.items() if community_sizes[community_id] >= size_threshold}

organized_communities = defaultdict(list)

for node, community_id in filtered_communities.items():
    organized_communities[community_id].append(node)

organized_communities = dict(organized_communities)

edges_with_weights_HT = [(u, v, d['weight']) for u, v, d in G_cell_weighted_HT.edges(data=True)]
print(edges_with_weights_HT[:10])


# # correlation after hard thresholding percentile preprocessed_data_array





print(np.sum(np.sum(non_zero_data,axis=0)==1))
print(np.sum(np.sum(non_zero_data,axis=0)==0))

print(np.sum(np.sum(preprocessed_data,axis=0)==1))
print(np.sum(np.sum(preprocessed_data,axis=0)==0))


print(np.sum(np.sum(preprocessed_data,axis=0)))
print(np.sum(np.sum(preprocessed_data,axis=0)))

data=preprocessed_data

clustering_performance=defaultdict()


# # correlation network

# # clustering by community detection on correlation networks



print(preprocessed_data.shape)
print(sparse_data.shape)




import community
G=G_cell_weighted

partition = community.best_partition(G, weight='weight')



size_threshold = 5

community_sizes = Counter(partition.values())

# Filter communities based on the size threshold
filtered_communities = {node: community_id for node, community_id in partition.items() if community_sizes[community_id] >= size_threshold}

organized_communities = defaultdict(list)
for node, community_id in filtered_communities.items():
    organized_communities[community_id].append(node)
organized_communities = dict(organized_communities)



df = preprocessed_df.T.copy()


# Add the community labels as a column to this DataFrame
df['community'] = list(partition.values())

# Sort the DataFrame based on the community labels
df_sorted = df.sort_values('community')

# Drop the community column before plotting
df_sorted = df_sorted.drop('community', axis=1)







community_dict=partition
sorted_dict = {k: community_dict[k] for k in node_list}
predicted_labels = list(sorted_dict.values())

community_dict=community_assignments
sorted_dict = {k: community_dict[k] for k in node_list}
true_labels = list(sorted_dict.values())

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")



clustering_performance['Louvain_weighted_ARI']=ari_score
clustering_performance['Louvain_weighted_NMI']=nmi_score




'''
plt.figure(figsize=(8, 8))
ax = plt.gca()  
ax.set_xticks([])
ax.set_yticks([])
ax.set(xlabel=None)
ax.set(ylabel=None)
sns.heatmap(df_sorted.T.astype(float), cmap='viridis', xticklabels=False, yticklabels=False, cbar=False)
plt.title('Louvain', fontsize=20, weight='bold')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.savefig(f"Fig/ngenes/Louvain_neat.pdf", bbox_inches='tight')
'''

# # Girvan Newman doesnt scale well way too slow
# 




#communities = next(girvan_newman(G))
#community_dict = {node: i for i, community in enumerate(communities) for node in community}
#sorted_dict = {k: community_dict[k] for k in node_list}
#predicted_labels = list(sorted_dict.values())


# # Leiden



G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True)


partition_leiden = la.find_partition(
    G_ig, 
    la.RBConfigurationVertexPartition,  
    weights=[i['weight'] for i in G_ig.es['weight']], resolution_parameter=1.1,n_iterations=-1)

community_dict_leiden = {node: cid for cid, community in enumerate(partition_leiden) for node in community}
sorted_dict = {k: community_dict_leiden[k] for k in node_list}
predicted_labels = list(sorted_dict.values())

community_dict=community_assignments
sorted_dict = {k: community_dict[k] for k in node_list}
true_labels = list(sorted_dict.values())

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")


clustering_performance['Leiden_weighted_ARI']=ari_score
clustering_performance['Leiden_weighted_NMI']=nmi_score

print(clustering_performance)
print(len(set(predicted_labels)))




df = preprocessed_df.T.copy()


df['community'] = list(community_dict_leiden.values())

df_sorted = df.sort_values('community')

df_sorted = df_sorted.drop('community', axis=1)





plt.figure(figsize=(8, 8))
ax = plt.gca()  
ax.set_xticks([])
ax.set_yticks([])
ax.set(xlabel=None)
ax.set(ylabel=None)
sns.heatmap(df_sorted.T.astype(float), cmap='viridis', xticklabels=False, yticklabels=False, cbar=False)
plt.title('Leiden', fontsize=20, weight='bold')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.savefig(f"Fig/ngenes/Leiden_neat_{avg_genes_per_module}", bbox_inches='tight',dpi=100)




# # Clauset-Newman-Moore greedy modularity
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html
# 
# References
# 
# [1]
# Newman, M. E. J. “Networks: An Introduction”, page 224 Oxford University Press 2011.
# [2]
# Clauset, A., Newman, M. E., & Moore, C. “Finding community structure in very large networks.” Physical Review E 70(6), 2004.
# [3]
# Reichardt and Bornholdt “Statistical Mechanics of Community Detection” Phys. Rev. E74, 2006.
# [4]
# Newman, M. E. J.”Analysis of weighted networks” Physical Review E 70(5 Pt 2):056131, 2004.





partition_greedy_modularity = nx.community.greedy_modularity_communities(G,weight='weight')

community_dict_gm={i:n for n, c in enumerate(partition_greedy_modularity) for i in c}


sorted_dict = {k: community_dict_gm[k] for k in node_list}
predicted_labels = list(sorted_dict.values())

community_dict=community_assignments
sorted_dict = {k: community_dict[k] for k in node_list}
true_labels = list(sorted_dict.values())

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")



clustering_performance['GreedyModularity_weighted_ARI']=ari_score
clustering_performance['GreedyModularity_weighted_NMI']=nmi_score




df = preprocessed_df.T.copy()


df['community'] = [community_dict_gm[i] for i in node_list]

df_sorted = df.sort_values('community')

df_sorted = df_sorted.drop('community', axis=1)





plt.figure(figsize=(8, 8))
ax = plt.gca()  
ax.set_xticks([])
ax.set_yticks([])
ax.set(xlabel=None)
ax.set(ylabel=None)
sns.heatmap(df_sorted.T.astype(float), cmap='viridis', xticklabels=False, yticklabels=False, cbar=False)
plt.title('GreedyModularity', fontsize=20, weight='bold')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.savefig(f"Fig/ngenes/GreedyModularity_neat_{avg_genes_per_module}", bbox_inches='tight',dpi=100)






# # Community detection HT


# weighted but sparsified by HT
G=G_cell_weighted_HT

partition = community.best_partition(G, weight='weight')
size_threshold = 5
community_sizes = Counter(partition.values())
filtered_communities = {node: community_id for node, community_id in partition.items() if community_sizes[community_id] >= size_threshold}

organized_communities = defaultdict(list)
for node, community_id in filtered_communities.items():
    organized_communities[community_id].append(node)
organized_communities = dict(organized_communities)

community_dict=partition
sorted_dict = {k: community_dict[k] for k in node_list}
predicted_labels = list(sorted_dict.values())

community_dict=community_assignments
sorted_dict = {k: community_dict[k] for k in node_list}
true_labels = list(sorted_dict.values())

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print("community_detection_weightedHT")
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")

clustering_performance['Louvain_weightedHT_ARI']=ari_score
clustering_performance['Louvain_weightedHT_NMI']=nmi_score


# # Infomap



G=G_cell_weighted_HT
G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True)

partition = G_ig.community_infomap(edge_weights=[i['weight'] for i in G_ig.es['weight']])
community_dict_infomap = {node: cid for cid, community in enumerate(partition) for node in community}
print(len(set(community_dict_infomap.values())))




sorted_dict = {k: community_dict_infomap[k] for k in node_list}
predicted_labels = list(sorted_dict.values())

community_dict=community_assignments
sorted_dict = {k: community_dict[k] for k in node_list}
true_labels = list(sorted_dict.values())

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")


clustering_performance['infomap_weightedHT_ARI']=ari_score
clustering_performance['infomap_weightedHT_NMI']=nmi_score


# # leading_eigenvector


G=G_cell_weighted_HT
G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True)

partition = G_ig.community_leading_eigenvector(clusters=num_clusters,weights=[i['weight'] for i in G_ig.es['weight']])
community_dict_eigen = {node: cid for cid, community in enumerate(partition) for node in community}
print(len(set(community_dict_eigen.values())))




sorted_dict = {k: community_dict_eigen[k] for k in node_list}
predicted_labels = list(sorted_dict.values())

community_dict=community_assignments
sorted_dict = {k: community_dict[k] for k in node_list}
true_labels = list(sorted_dict.values())

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")


clustering_performance['eigenvector_weightedHT_ARI']=ari_score
clustering_performance['eigenvector_weightedHT_NMI']=nmi_score



# # community_multilevel


G=G_cell_weighted_HT
G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True)

partition = G_ig.community_multilevel(weights=[i['weight'] for i in G_ig.es['weight']], resolution=1)
community_dict_multilevel = {node: cid for cid, community in enumerate(partition) for node in community}
print(len(set(community_dict_multilevel.values())))




sorted_dict = {k: community_dict_multilevel[k] for k in node_list}
predicted_labels = list(sorted_dict.values())

community_dict=community_assignments
sorted_dict = {k: community_dict[k] for k in node_list}
true_labels = list(sorted_dict.values())

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")



clustering_performance['Multilevel_weightedHT_ARI']=ari_score
clustering_performance['Multilevel_weightedHT_NMI']=nmi_score


# # PCA


print(data.shape)



n_components=30
pca = PCA(n_components)
pca_embedding_counts = pca.fit_transform(data.T)
print(pca_embedding_counts.shape)



embedding_method='PCA'
embeddings=dict(zip(node_list,pca_embedding_counts))

node_embeddings=np.array([list(embeddings[node]) for node_index,node in enumerate(node_list)])

community_size_min=3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

sorted_dict = {k: community_assignments[k] for k in node_list}
true_labels = list(sorted_dict.values())

################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(node_embeddings)
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")
clustering_performance[f'{embedding_method}_ARI']=ari_score
clustering_performance[f'{embedding_method}_NMI']=nmi_score

################################################################################################################################################

# # UMAP

umap = UMAP(n_components=30)
UMAP_embedding_counts = umap.fit_transform(data.T)

embedding_method='UMAP'
embeddings=dict(zip(node_list,UMAP_embedding_counts))

node_embeddings=np.array([list(embeddings[node]) for node_index,node in enumerate(node_list)])

community_size_min=3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

sorted_dict = {k: community_assignments[k] for k in node_list}
true_labels = list(sorted_dict.values())
################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(node_embeddings)
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")
clustering_performance[f'{embedding_method}_ARI']=ari_score
clustering_performance[f'{embedding_method}_NMI']=nmi_score

################################################################################################################################################


# # tSNE



tsne = TSNE(n_components=30, method='exact', random_state=42)
tSNE_embedding_counts = tsne.fit_transform(data.T)


embedding_method='tSNE'
embeddings=dict(zip(node_list,tSNE_embedding_counts))

node_embeddings=np.array([list(embeddings[node]) for node_index,node in enumerate(node_list)])

community_size_min=3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

sorted_dict = {k: community_assignments[k] for k in node_list}
true_labels = list(sorted_dict.values())


################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(node_embeddings)
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")
clustering_performance[f'{embedding_method}_ARI']=ari_score
clustering_performance[f'{embedding_method}_NMI']=nmi_score

################################################################################################################################################


# # Node2vec

# # construct unweighted graph

# # a small example of random walk on simple graph + word2vec




num_walks_per_node = 30  # Number of random walks per node
walk_length = 30  # Length of each random walk
embedded_dimensions=30




corr=cell_coexpression_weighted.copy()
np.fill_diagonal(corr, 0)
row_sums = corr.sum(axis=1)
P = corr / row_sums[:, np.newaxis]




random_walks = generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)

# Train Word2Vec model on the random walks
model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5, min_count=0, sg=1, workers=num_walks_per_node)

# Get node embeddings
node_ids = list(G.nodes)
embeddings_node2vec={node_name:model.wv[node_id] for node_id,node_name in enumerate(node_list)}

################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(np.array(list(embeddings_node2vec.values())))
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f'node2vec')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")

clustering_performance[f'node2vec_ARI']=ari_score
clustering_performance[f'node2vec_NMI']=nmi_score





# # edge dependent Hypergraph Random Walk
# # EDHW

# # Let's correct the hypergraph implementation
# # the original edge-dependent hypergraph random walk


# original version
embedding_method='EDHW'

H=data.copy()
I=Incidence_matrix(H)
delta_e=degree_edge(H)
edge_weights=np.sum(H, axis=1)
W=W_hyperedge(I,edge_weights)
dv=degree_vertex(W)
P=P_original(dv,delta_e,W,H)



exec(f'random_walks_{walk_length}_{num_walks_per_node}= generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)')
random_walks = generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)
exec(f"random_walks=random_walks_{walk_length}_{num_walks_per_node}")
# Train Word2Vec model on the random walks


model = Word2Vec(sentences=random_walks, vector_size=walk_length, window=5, min_count=0, sg=1, workers=num_walks_per_node)

embeddings={node_name:model.wv[node_id] for node_id,node_name in enumerate(node_list)}

exec(f"embeddings_{walk_length}_{num_walks_per_node}=embeddings")


node_list=list(embeddings.keys())

node_embeddings=np.array([list(embeddings[node_name]) for node_id,node_name in enumerate(node_list)])

community_size_min=3
num_clusters = num_modules 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

sorted_dict = {k: community_assignments[k] for k in node_list}
true_labels = list(sorted_dict.values())

################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(node_embeddings)
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")
clustering_performance[f'{embedding_method}_ARI']=ari_score
clustering_performance[f'{embedding_method}_NMI']=nmi_score
################################################################################################################################################







# # Adjusted version with preference exponent and hyperedge selection probability based also on weight of the current vertex in the edge
# 
# # DIPHW



#num_walks_per_node =30
#walk_length = 30
#embedded_dimensions=30

H=data.copy()
preference_exponent=50
P=P_adjusted(H,preference_exponent)
embedding_method=f'DIPHW_{preference_exponent}'



exec(f'random_walks_{walk_length}_{num_walks_per_node}= generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)')
random_walks = generate_random_walks_for_all_nodes(P, num_walks_per_node, walk_length)
exec(f"random_walks=random_walks_{walk_length}_{num_walks_per_node}")
# Train Word2Vec model on the random walks


model = Word2Vec(sentences=random_walks, vector_size=walk_length, window=5, min_count=0, sg=1, workers=num_walks_per_node)

embeddings={node_name:model.wv[node_id] for node_id,node_name in enumerate(node_list)}

exec(f"embeddings_{walk_length}_{num_walks_per_node}=embeddings")


node_list=list(embeddings.keys())

node_embeddings=np.array([list(embeddings[node_name]) for node_id,node_name in enumerate(node_list)])

community_size_min=3
num_clusters = num_modules 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

sorted_dict = {k: community_assignments[k] for k in node_list}
true_labels = list(sorted_dict.values())



################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(node_embeddings)
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")
clustering_performance[f'{embedding_method}_ARI']=ari_score
clustering_performance[f'{embedding_method}_NMI']=nmi_score
################################################################################################################################################


# # CoMem



HT=False
laziness='non-lazy'


if HT==True:
    G_E = gene_coexpression_weighted_HT
    G_V = cell_coexpression_weighted_HT
    embedding_method='CoMem_HT'

else:
    G_E = gene_coexpression_weighted
    G_V = cell_coexpression_weighted 
    embedding_method='CoMem'

if laziness=='non-lazy':
    np.fill_diagonal(G_E, 0)
    np.fill_diagonal(G_V, 0)    


Gamma = data.copy()
P = unipartite_transition_probability_vectorized(G_E, G_V, Gamma, 'optimal')
P_vw_normalised = normalize_transition_probability_matrix(P)





exec(f'random_walks_{walk_length}_{num_walks_per_node}= generate_random_walks_for_all_nodes(P_vw_normalised, num_walks_per_node, walk_length)')
random_walks = generate_random_walks_for_all_nodes(P_vw_normalised, num_walks_per_node, walk_length)
exec(f"random_walks=random_walks_{walk_length}_{num_walks_per_node}")
# Train Word2Vec model on the random walks


model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5, min_count=0, sg=1, workers=num_walks_per_node)

embeddings={node_name:model.wv[node_id] for node_id,node_name in enumerate(node_list)}

exec(f"embeddings_{walk_length}_{num_walks_per_node}=embeddings")


node_list=list(embeddings.keys())

node_embeddings=np.array([list(embeddings[node_name]) for node_id,node_name in enumerate(node_list)])

community_size_min=3
num_clusters = num_modules 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

sorted_dict = {k: community_assignments[k] for k in node_list}
true_labels = list(sorted_dict.values())


################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(node_embeddings)
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")
clustering_performance[f'{embedding_method}_ARI']=ari_score
clustering_performance[f'{embedding_method}_NMI']=nmi_score
################################################################################################################################################



# # CoMem-DIPHW



preference_exponent=50



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

P = unipartite_transition_probability_adjusted_vectorized(G_E, G_V, Gamma,preference_exponent, 'optimal')
P_vw_normalised = normalize_transition_probability_matrix(P)

embedding_method=f'CoMem-DIPHW_{preference_exponent}'


exec(f'random_walks_{walk_length}_{num_walks_per_node}= generate_random_walks_for_all_nodes(P_vw_normalised, num_walks_per_node, walk_length)')
random_walks = generate_random_walks_for_all_nodes(P_vw_normalised, num_walks_per_node, walk_length)
exec(f"random_walks=random_walks_{walk_length}_{num_walks_per_node}")
# Train Word2Vec model on the random walks


model = Word2Vec(sentences=random_walks, vector_size=embedded_dimensions, window=5, min_count=0, sg=1, workers=num_walks_per_node)

embeddings={node_name:model.wv[node_id] for node_id,node_name in enumerate(node_list)}

exec(f"embeddings_{walk_length}_{num_walks_per_node}=embeddings")


node_list=list(embeddings.keys())

node_embeddings=np.array([list(embeddings[node_name]) for node_id,node_name in enumerate(node_list)])

community_size_min=3
num_clusters = num_modules 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

sorted_dict = {k: community_assignments[k] for k in node_list}
true_labels = list(sorted_dict.values())


################################################################################################################################################



# Apply KMeans clustering to the embedding before tSNE or UMAP


clusters = kmeans.fit_predict(node_embeddings)
predicted_labels = list(clusters)

ari_score = adjusted_rand_score(true_labels, predicted_labels)
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')
print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information: {nmi_score}")
clustering_performance[f'{embedding_method}_ARI']=ari_score
clustering_performance[f'{embedding_method}_NMI']=nmi_score
################################################################################################################################################







results = {}

for key, value in clustering_performance.items():

    if 'weighted' in key:
        method,_,metric=key.split('_')
        results[f'{method}_{metric}'] = value
    else:
        
        results[key] = value
results





methods=['Louvain',
 'GreedyModularity',
 'infomap',
 'Multilevel',
 'PCA',
 'node2vec',
 'UMAP',
 'tSNE',
 'EDHW',
 'DIPHW_50',
 'CoMem',
 'CoMem-DIPHW_50']



ari_data = {k.replace('_ARI', '').replace('_UMAP', ''): v for k, v in results.items() if '_ARI' in k}
#methods=list(ari_data.keys())

values = [ari_data[method] for method in methods]
create_bar_plot(methods, values, 'ARI')



nmi_data = {k.replace('_NMI', '').replace('_UMAP', ''): v for k, v in results.items() if '_NMI' in k}
#methods=list(nmi_data.keys())

values = [nmi_data[method] for method in methods]
create_bar_plot(methods, values, 'NMI')

results=defaultdict()
results = {'rows': rows,
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
        'inter_module_lambda': inter_module_lambda}


for (i,j) in list(clustering_performance.items()):
    results[i]=j


filename = 'ClusteringPerformance_ngenes_clean_20250212.csv'


if not os.path.isfile(filename):

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()


with open(filename, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    writer.writerow(results)

