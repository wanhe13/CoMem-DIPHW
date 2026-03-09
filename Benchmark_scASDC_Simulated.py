#!/usr/bin/env python
# coding: utf-8
"""
Benchmark Script for scASDC on Simulated Data
=============================================
Runs scASDC clustering on simulated scRNA-seq data.
"""

import argparse
import sys
import os
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import warnings
import scanpy as sc

warnings.filterwarnings("ignore")

# =============================================================================
# Path Setup & Imports
# =============================================================================
parent_directory = os.path.abspath('..')
scasdc_dir = os.path.join(parent_directory, 'CoMemEmpirical', 'scASDC')

if not os.path.exists(scasdc_dir):
    raise FileNotFoundError(f"scASDC directory not found at: {scasdc_dir}")

# Import scASDC modules
# Clean 'utils' to prevent conflicts
if 'utils' in sys.modules:
    del sys.modules['utils']

sys.path.insert(0, scasdc_dir)
import run_scASDC
from run_scASDC import train_scASDC
from utils.calcu_graph import construct_graph
from utils.utils import load_data
from utils.preprocess import normalize_1
sys.path.remove(scasdc_dir)

# Import simulation functions
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from Simulation_functions import *
from functions import *

# =============================================================================
# Experiment Configuration
# =============================================================================
EXPERIMENT_CONFIG = {
    'ngenes': {
        'results_file': 'results/ClusteringPerformance_ngenes_5metrics.csv',
        'output_file': 'scASDC_simulated_ngenes.csv',
    },
    'nmodules': {
        'results_file': 'results/ClusteringPerformance_nmodules_5metrics.csv',
        'output_file': 'scASDC_simulated_nmodules.csv',
    }
}

# =============================================================================
# scASDC Hyperparameters
# =============================================================================
SCASDC_PARAMS = {
    'lr': 1e-4,
    'n_z': 10,
    'top_k': 10,
    'max_epochs': 200,
    'high_genes': 3000,
}

# =============================================================================
# Helper Functions
# =============================================================================
def filter_zero_expression_community(expression_df_with_CommunityAssignment, thres=0):
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
    print("Generating simulated data...")
    modules = simulate_input_modules(
        sim_params['rows'], sim_params['cols'], sim_params['num_modules'],
        sim_params['avg_genes_per_module'], sim_params['avg_cells_per_module']
    )
    sparse_data = create_sparse_matrix_with_inter_module_variance(
        sim_params['rows'], sim_params['cols'], modules, sim_params['target_density'],
        sim_params['module_density'], sim_params['inter_module_density'],
        sim_params['inter_module_connection_probability'], sim_params['lambda_background'],
        sim_params['lambda_module'], sim_params['inter_module_lambda']
    )
    data_df_community = pd.DataFrame.sparse.from_spmatrix(sparse_data.T)
    community_assignments = modules_to_community_dict(modules, sim_params['cols'])
    data_df_community['community'] = list(community_assignments.values())
    data_df_community = data_df_community.T
    
    column_names = data_df_community.columns.tolist()
    shuffled_column_names = np.random.permutation(column_names)
    shuffled_df_community = data_df_community[shuffled_column_names]
    
    non_zero_df = filter_zero_expression_community(shuffled_df_community)
    raw_counts_array = np.array(non_zero_df.iloc[:-1].values)
    community_assignments = dict(zip(non_zero_df.columns, non_zero_df.loc['community']))
    
    node_list = list(non_zero_df.columns)
    gene_list = list(non_zero_df.index[:-1])
    num_clusters = sim_params['num_modules']
    
    sorted_dict = {k: community_assignments[k] for k in node_list}
    true_labels = list(sorted_dict.values())
    
    raw_counts_df = pd.DataFrame(raw_counts_array, index=gene_list, columns=node_list)
    
    print(f"  Data shape: {raw_counts_array.shape} (genes x cells)")
    print(f"  Number of clusters: {num_clusters}")
    
    return raw_counts_df, true_labels, num_clusters


def run_scasdc(raw_counts_df, true_labels, num_clusters, random_seed, row_idx, experiment_type):
    print("\nRunning scASDC...")
    start_time = time.time()
    
    dataname = f"sim_{experiment_type}_{row_idx}"
    
    # 1. Prepare data
    x = raw_counts_df.T.values
    y = np.array(true_labels)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 2. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=dataname)
    parser.add_argument('--k', type=int, default=SCASDC_PARAMS['top_k'])
    parser.add_argument('--lr', type=float, default=SCASDC_PARAMS['lr'])
    parser.add_argument('--n_clusters', default=num_clusters, type=int)
    parser.add_argument('--n_z', default=SCASDC_PARAMS['n_z'], type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--n_input', type=int, default=SCASDC_PARAMS['high_genes'])
    parser.add_argument('--max_epochs', type=int, default=SCASDC_PARAMS['max_epochs'])
    args = parser.parse_args([])
    
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"  Device: {device}")
    
    # 3. Preprocess
    x_processed = np.ceil(x).astype(int)
    adata = sc.AnnData(x_processed)
    adata.obs['Group'] = y_encoded
    
    high_genes = SCASDC_PARAMS['high_genes']
    if x_processed.shape[1] < high_genes:
        print(f"  WARNING: Total genes ({x_processed.shape[1]}) < high_genes ({high_genes})")
        high_genes = x_processed.shape[1]
    
    adata = normalize_1(adata, copy=True, highly_genes=high_genes,
                        size_factors=True, normalize_input=True, logtrans_input=True)
    count = adata.X
    print(f"  Shape after preprocessing: {count.shape}")
    
    # 4. Run Training
    cwd = os.getcwd()
    os.chdir(scasdc_dir)
    
    try:
        # Create necessary folders inside scASDC dir
        os.makedirs('../graph', exist_ok=True)
        os.makedirs('./pretrained_models', exist_ok=True)
        
        args.pretrain_path = f'./pretrained_models/{dataname}.pkl'
        
        print(f"  Constructing graph: {dataname}...")
        construct_graph(count, y_encoded, 'ncos', name=dataname, topk=SCASDC_PARAMS['top_k'])
        
        dataset = load_data(count, y_encoded)
        sf = adata.obs.size_factors
        
        # Inject args
        run_scASDC.args = args
        run_scASDC.device = device
        
        print(f"  Training for {SCASDC_PARAMS['max_epochs']} epochs...")
        acc, nmi, ari, ami, f1 = train_scASDC(dataset, count, sf)
        
    finally:
        os.chdir(cwd)
    
    elapsed_time = time.time() - start_time
    print(f"  scASDC completed in {elapsed_time/60:.1f} minutes")
    
    return {
        'metrics': {'ACC': acc, 'NMI': nmi, 'ARI': ari, 'AMI': ami, 'F1': f1},
        'elapsed_time': elapsed_time
    }


# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='scASDC Benchmark')
    parser.add_argument('--experiment', type=str, required=True, choices=['ngenes', 'nmodules'])
    parser.add_argument('--row_idx', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='scASDC_Simulated_Results')
    
    args = parser.parse_args()
    exp_config = EXPERIMENT_CONFIG[args.experiment]
    
    print("=" * 70)
    print(f"scASDC Benchmark - {args.experiment}")
    print(f"Row index: {args.row_idx}")
    print("=" * 70)
    
    if not os.path.exists(exp_config['results_file']):
        print(f"ERROR: Results file not found: {exp_config['results_file']}")
        sys.exit(1)
        
    params_df = pd.read_csv(exp_config['results_file'])
    if args.row_idx >= len(params_df):
        print(f"ERROR: Row index {args.row_idx} out of range")
        sys.exit(1)
        
    param_row = params_df.iloc[args.row_idx]
    run_idx = int(param_row.get('run_idx', args.row_idx))
    RANDOM_SEED = 42 + run_idx
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
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
    
    # Run
    raw_counts_df, true_labels, num_clusters = generate_simulated_data(sim_params, RANDOM_SEED)
    results = run_scasdc(raw_counts_df, true_labels, num_clusters, RANDOM_SEED, args.row_idx, args.experiment)
    met = results['metrics']
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"scASDC: ACC={met['ACC']:.4f}, NMI={met['NMI']:.4f}, ARI={met['ARI']:.4f}, F1={met['F1']:.4f}")
    
    # Save Results
    task_dir = Path(args.output_dir) / f"task_{args.row_idx}"
    task_dir.mkdir(parents=True, exist_ok=True)
    output_file = task_dir / exp_config['output_file']
    
    output_row = {
        'row_idx': args.row_idx,
        'experiment': args.experiment,
        'scASDC_ACC': met['ACC'],
        'scASDC_NMI': met['NMI'],
        'scASDC_ARI': met['ARI'],
        'scASDC_AMI': met['AMI'],
        'scASDC_F1': met['F1'],
        'scASDC_runtime_minutes': results['elapsed_time'] / 60,
        'rows': sim_params['rows'],
        'cols': sim_params['cols'],
        'num_modules': sim_params['num_modules'],
        'run_idx': run_idx
    }
    
    # Direct overwrite save
    output_df = pd.DataFrame([output_row])
    output_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()