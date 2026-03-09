#!/usr/bin/env python
# coding: utf-8
"""
Benchmark Script for graph-sc on Simulated Data
===============================================
Runs graph-sc clustering on simulated scRNA-seq data for ngenes or nmodules experiments.
Uses the same data generation as ClusteringOnSimulatedData_ngenes_5metrics.py.
"""

import argparse
import sys
import os
import time
import random
import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import csv
import warnings

warnings.filterwarnings("ignore")

parent_directory = os.path.abspath('..')  # NSF/
graphsc_dir = os.path.join(parent_directory, 'CoMemEmpirical', 'graph-sc')


# Import graph-sc modules
sys.path.insert(0, graphsc_dir)
import train
import models
sys.path.remove(graphsc_dir)

# Import simulation functions
from Simulation_functions import *
from functions import *


EXPERIMENT_CONFIG = {
    'ngenes': {
        'results_file': 'results/ClusteringPerformance_ngenes_5metrics.csv',
        'output_file': 'graph-sc_simulated_ngenes.csv',
        'description': 'Variable number of genes experiment'
    },
    'nmodules': {
        'results_file': 'results/ClusteringPerformance_nmodules_5metrics.csv',
        'output_file': 'graph-sc_simulated_nmodules.csv',
        'description': 'Variable number of modules experiment'
    }
}

# =============================================================================
# Graph-sc Hyperparameters
# =============================================================================
GRAPHSC_PARAMS = {
    'nb_genes': 3000,
    'pca_size': 50,
    'normalize_weights': 'log_per_cell',
    'n_layers': 2,
    'hidden_dim': 128,
    'hidden': [128],
    'dropout': 0.1,
    'learning_rate': 1e-5,
    'epochs': 100,
    'batch_size': 128,
}


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy and F1 score using the Hungarian algorithm.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.dtype.kind in ['U', 'S', 'O']:
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
    y_true = y_true.astype(np.int64)
    
    if y_pred.dtype.kind in ['U', 'S', 'O']:
        le_pred = LabelEncoder()
        y_pred = le_pred.fit_transform(y_pred)
    y_pred = y_pred.astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    acc = w[row_ind, col_ind].sum() / len(y_pred)
    
    map_dict = {r: c for r, c in zip(row_ind, col_ind)}
    new_predict = np.array([map_dict.get(x, x) for x in y_pred])
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    
    return acc, f1_macro


def compute_all_metrics(y_true, y_pred):
    """
    Compute all five clustering metrics: ACC, NMI, AMI, ARI, F1.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    if y_true_arr.dtype.kind in ['U', 'S', 'O']:
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true_arr)
    else:
        y_true_encoded = y_true_arr.astype(np.int64)
    
    if y_pred_arr.dtype.kind in ['U', 'S', 'O']:
        le_pred = LabelEncoder()
        y_pred_encoded = le_pred.fit_transform(y_pred_arr)
    else:
        y_pred_encoded = y_pred_arr.astype(np.int64)
    
    acc, f1 = cluster_acc(y_true, y_pred)
    
    return {
        'ACC': acc,
        'NMI': metrics.normalized_mutual_info_score(y_true_encoded, y_pred_encoded, average_method='arithmetic'),
        'AMI': metrics.adjusted_mutual_info_score(y_true_encoded, y_pred_encoded, average_method='arithmetic'),
        'ARI': metrics.adjusted_rand_score(y_true_encoded, y_pred_encoded),
        'F1': f1
    }


# =============================================================================
# Helper Functions
# =============================================================================
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


def generate_simulated_data(sim_params, random_seed):
    """Generate simulated scRNA-seq data."""
    print("Generating simulated data...")
    
    # Generate modules
    modules = simulate_input_modules(
        sim_params['rows'], sim_params['cols'], sim_params['num_modules'],
        sim_params['avg_genes_per_module'], sim_params['avg_cells_per_module']
    )
    
    # Create sparse matrix
    sparse_data = create_sparse_matrix_with_inter_module_variance(
        sim_params['rows'], sim_params['cols'], modules, sim_params['target_density'],
        sim_params['module_density'], sim_params['inter_module_density'],
        sim_params['inter_module_connection_probability'], sim_params['lambda_background'],
        sim_params['lambda_module'], sim_params['inter_module_lambda']
    )
    
    # Prepare data
    data_df_community = pd.DataFrame.sparse.from_spmatrix(sparse_data.T)
    community_assignments = modules_to_community_dict(modules, sim_params['cols'])
    data_df_community['community'] = list(community_assignments.values())
    data_df_community = data_df_community.T
    
    # Shuffle columns
    column_names = data_df_community.columns.tolist()
    shuffled_column_names = np.random.permutation(column_names)
    shuffled_df_community = data_df_community[shuffled_column_names]
    
    # Filter zero expression
    non_zero_df = filter_zero_expression_community(shuffled_df_community)
    raw_counts_array = np.array(non_zero_df.iloc[:-1].values)
    community_assignments = dict(zip(non_zero_df.columns, non_zero_df.loc['community']))
    
    node_list = list(non_zero_df.columns)
    gene_list = list(non_zero_df.index[:-1])
    num_clusters = sim_params['num_modules']
    
    # Get true labels
    sorted_dict = {k: community_assignments[k] for k in node_list}
    true_labels = list(sorted_dict.values())
    
    raw_counts_df = pd.DataFrame(raw_counts_array, index=gene_list, columns=node_list)
    
    print(f"  Data shape: {raw_counts_array.shape} (genes x cells)")
    print(f"  Number of cells: {len(node_list)}")
    print(f"  Number of genes: {len(gene_list)}")
    print(f"  Number of clusters: {num_clusters}")
    
    return raw_counts_df, true_labels, num_clusters


def run_graphsc(raw_counts_df, true_labels, num_clusters, random_seed):
    """Run graph-sc clustering."""
    print("\nRunning graph-sc...")
    start_time = time.time()
    
    # Prepare data
    X = raw_counts_df.T.values.astype(np.float32)
    Y = np.array(true_labels)
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    
    print(f"  Input shape: {X.shape[0]} cells x {X.shape[1]} genes")
    
    nb_genes = GRAPHSC_PARAMS['nb_genes']
    if X.shape[1] < nb_genes:
        print(f"  WARNING: Total genes ({X.shape[1]}) < nb_genes ({nb_genes}). Using all available.")
        nb_genes = X.shape[1]
    
    # Preprocess: filter genes
    print("  Filtering genes...")
    genes_idx, cells_idx = train.filter_data(X, highly_genes=nb_genes)
    
    X = X[cells_idx][:, genes_idx]
    Y_encoded = Y_encoded[cells_idx]
    
    print(f"  After filtering: {X.shape[0]} cells x {X.shape[1]} genes")
    
    if X.shape[0] == 0:
        raise ValueError("All cells filtered out.")
    
    # Create graph
    print("  Creating graph...")
    graph = train.make_graph(
        X,
        Y_encoded,
        dense_dim=GRAPHSC_PARAMS['pca_size'],
        normalize_weights=GRAPHSC_PARAMS['normalize_weights'],
    )
    
    labels = graph.ndata["label"]
    train_ids = np.where(labels != -1)[0]
    
    # Create DataLoader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(GRAPHSC_PARAMS['n_layers'])
    dataloader = dgl.dataloading.DataLoader(
        graph,
        train_ids,
        sampler,
        batch_size=GRAPHSC_PARAMS['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    
    # Create model
    device = train.get_device(use_cpu=not torch.cuda.is_available())
    print(f"  Device: {device}")
    
    model = models.GCNAE(
        in_feats=GRAPHSC_PARAMS['pca_size'],
        n_hidden=GRAPHSC_PARAMS['hidden_dim'],
        n_layers=GRAPHSC_PARAMS['n_layers'],
        activation=F.relu,
        dropout=GRAPHSC_PARAMS['dropout'],
        hidden=GRAPHSC_PARAMS['hidden'],
    ).to(device)
    
    # Train
    print(f"  Training: epochs={GRAPHSC_PARAMS['epochs']}, lr={GRAPHSC_PARAMS['learning_rate']}")
    optim = torch.optim.Adam(model.parameters(), lr=GRAPHSC_PARAMS['learning_rate'])
    
    results = train.train(
        model,
        optim,
        GRAPHSC_PARAMS['epochs'],
        dataloader,
        num_clusters,
        plot=False,
        save=False,
        cluster=["KMeans"],
        use_cpu=not torch.cuda.is_available()
    )
    
    elapsed_time = time.time() - start_time
    print(f"  graph-sc completed in {elapsed_time/60:.1f} minutes")
    
    # Get predictions and compute metrics
    preds = np.array(results['kmeans_pred'])
    all_metrics = compute_all_metrics(Y_encoded, preds)
    
    return {
        'predictions': preds,
        'metrics': all_metrics,
        'elapsed_time': elapsed_time
    }


# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Graph-sc Benchmark for Simulated scRNA-seq Data'
    )
    parser.add_argument('--experiment', type=str, required=True, choices=['ngenes', 'nmodules'],
                        help='Experiment type: ngenes or nmodules')
    parser.add_argument('--row_idx', type=int, required=True,
                        help='Row index from the simulation parameters CSV')
    parser.add_argument('--output_dir', type=str, default='graph-sc_Simulated_Results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Get experiment configuration
    exp_config = EXPERIMENT_CONFIG[args.experiment]
    
    print("=" * 70)
    print(f"Graph-sc Benchmark - {exp_config['description']}")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Row index: {args.row_idx}")
    print("=" * 70)
    
    # Load simulation parameters from results file
    if not os.path.exists(exp_config['results_file']):
        print(f"ERROR: Results file not found: {exp_config['results_file']}")
        sys.exit(1)
    
    params_df = pd.read_csv(exp_config['results_file'])
    
    if args.row_idx >= len(params_df):
        print(f"ERROR: Row index {args.row_idx} out of range (max: {len(params_df)-1})")
        sys.exit(1)
    
    param_row = params_df.iloc[args.row_idx]
    
    # Get run_idx to compute correct random seed
    run_idx = int(param_row.get('run_idx', args.row_idx))
    RANDOM_SEED = 42 + run_idx
    
    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    print(f"run_idx: {run_idx}")
    print(f"random_seed: {RANDOM_SEED}")
    
    # Extract simulation parameters
    sim_params = {
        'rows': int(param_row['rows']),
        'cols': int(param_row['cols']),
        'num_modules': int(param_row['num_modules']),
        'avg_genes_per_module': int(param_row['avg_genes_per_module']),
        'avg_cells_per_module': int(param_row['avg_cells_per_module']),
        'target_density': float(param_row['target_density']),
        'module_density': float(param_row['module_density']),
        'inter_module_density': float(param_row['inter_module_density']),
        'inter_module_connection_probability': float(param_row['inter_module_connection_probability']),
        'lambda_background': int(param_row['lambda_background']),
        'lambda_module': int(param_row['lambda_module']),
        'inter_module_lambda': int(param_row['inter_module_lambda'])
    }
    
    print(f"\nSimulation parameters:")
    print(f"  Rows (genes): {sim_params['rows']}")
    print(f"  Cols (cells): {sim_params['cols']}")
    print(f"  Modules: {sim_params['num_modules']}")
    print(f"  avg_genes_per_module: {sim_params['avg_genes_per_module']}")
    
    # Generate data and run graph-sc
    raw_counts_df, true_labels, num_clusters = generate_simulated_data(sim_params, RANDOM_SEED)
    
    try:
        results = run_graphsc(raw_counts_df, true_labels, num_clusters, RANDOM_SEED)
        met = results['metrics']
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"graph-sc: ACC={met['ACC']:.4f}, NMI={met['NMI']:.4f}, ARI={met['ARI']:.4f}, AMI={met['AMI']:.4f}, F1={met['F1']:.4f}")
        
    except Exception as e:
        print(f"\nERROR running graph-sc: {e}")
        import traceback
        traceback.print_exc()
        met = {'ACC': np.nan, 'NMI': np.nan, 'AMI': np.nan, 'ARI': np.nan, 'F1': np.nan}
        results = {'elapsed_time': np.nan}
    
    # Create safe output directory: graph-sc_Simulated_Results/task_X/
    task_dir = Path(args.output_dir) / f"task_{args.row_idx}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = task_dir / exp_config['output_file']
    
    # Prepare output row
    output_row = {
        'row_idx': args.row_idx,  
        'run_idx': run_idx,
        'rows': sim_params['rows'],
        'cols': sim_params['cols'],
        'num_modules': sim_params['num_modules'],
        'avg_genes_per_module': sim_params['avg_genes_per_module'],
        'avg_cells_per_module': sim_params['avg_cells_per_module'],
        'target_density': sim_params['target_density'],
        'module_density': sim_params['module_density'],
        'inter_module_density': sim_params['inter_module_density'],
        'inter_module_connection_probability': sim_params['inter_module_connection_probability'],
        'lambda_background': sim_params['lambda_background'],
        'lambda_module': sim_params['lambda_module'],
        'inter_module_lambda': sim_params['inter_module_lambda'],
        'graph-sc_ACC': met['ACC'],
        'graph-sc_NMI': met['NMI'],
        'graph-sc_ARI': met['ARI'],
        'graph-sc_AMI': met['AMI'],
        'graph-sc_F1': met['F1'],
        'graph-sc_runtime_minutes': results['elapsed_time'] / 60 if not np.isnan(results.get('elapsed_time', np.nan)) else np.nan,
    }

    output_df = pd.DataFrame([output_row])
    output_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()