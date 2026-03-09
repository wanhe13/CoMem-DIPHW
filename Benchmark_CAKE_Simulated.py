#!/usr/bin/env python
# coding: utf-8
"""
Unified CAKE Benchmark Script for Simulated Data
=================================================
Runs CAKE clustering on simulated scRNA-seq data for ngenes or nmodules experiments.
Uses optimized hyperparameters from scMixology benchmarks.

Usage:
    python Benchmark_CAKE_Simulated.py --experiment ngenes --row_idx 0
    python Benchmark_CAKE_Simulated.py --experiment nmodules --row_idx 5

Output:
    CAKE_simulated_ngenes.csv or CAKE_simulated_nmodules.csv
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import warnings
import h5py

warnings.filterwarnings('ignore')

# =============================================================================
# CAKE Hyperparameters (from Benchmark_ReviewerMethods)
# =============================================================================
CAKE_PARAMS = {
    'seed': 42,
    'num_genes': 2000,
    'epochs': 400,
    'batch_size': 128,
    'learning_rate': 0.0001,
    'latent_features': [2048, 1024, 256],
    'K': 4096,
    'm': 0.999,
    'T': 0.1,
    'p': 0.5,
    'lam': 0.7,
    'alpha': 0.2,
    'kd_alpha': 0.05,
    'kd_temperature': 3,
    'teacher_epochs': 100,
    'distill_epochs': 50,
    'n_neighbors': 20,
    'resolution': 1.0,
}

# =============================================================================
# Experiment Configuration
# =============================================================================
EXPERIMENT_CONFIG = {
    'ngenes': {
        'results_file': 'results/ClusteringPerformance_ngenes_5metrics.csv',
        'output_file': 'CAKE_simulated_ngenes.csv',
        'description': 'Variable number of genes experiment'
    },
    'nmodules': {
        'results_file': 'results/ClusteringPerformance_nmodules_5metrics.csv',
        'output_file': 'CAKE_simulated_nmodules.csv',
        'description': 'Variable number of modules experiment'
    }
}

# =============================================================================
# Helper Functions
# =============================================================================
@contextmanager
def working_directory(path):
    """Context manager for changing working directory."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def filter_zero_expression_community_fixed(df):
    """
    Filter out genes (rows) and cells (columns) with zero expression.
    Preserves the 'community' row.
    
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with genes as rows, cells as columns, and a 'community' row
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    # Separate the community row
    if 'community' in df.index:
        community_row = df.loc[['community']]
        data_df = df.drop('community')
    else:
        # Assume last row is community
        community_row = df.iloc[[-1]]
        data_df = df.iloc[:-1]
    
    # Convert to numeric for filtering
    data_numeric = data_df.apply(pd.to_numeric, errors='coerce')
    
    # Filter columns (cells) with zero total expression
    col_sums = data_numeric.sum(axis=0)
    non_zero_cols = col_sums[col_sums != 0].index
    filtered_df = data_numeric[non_zero_cols]
    
    # Filter rows (genes) with zero total expression
    row_sums = filtered_df.sum(axis=1)
    non_zero_rows = row_sums[row_sums != 0].index
    filtered_df = filtered_df.loc[non_zero_rows]
    
    # Filter community row to match remaining columns
    community_filtered = community_row[non_zero_cols]
    
    # Concatenate back (pandas 2.0+ compatible)
    result = pd.concat([filtered_df, community_filtered], axis=0)
    
    return result


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy and F1 score using the Hungarian algorithm.
    
    Parameters
    ----------
    y_true : Ground truth labels
    y_pred : Predicted cluster labels
        
    Returns
    -------
    tuple
        (accuracy, f1_macro)
    """
    from sklearn import metrics
    from sklearn.preprocessing import LabelEncoder
    from scipy.optimize import linear_sum_assignment
    
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
    
    # Build confusion matrix
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    
    # Hungarian algorithm (maximize overlap)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # Calculate accuracy
    acc = w[row_ind, col_ind].sum() / len(y_pred)
    
    # F1 score (remap predictions to match truth)
    map_dict = {r: c for r, c in zip(row_ind, col_ind)}
    new_predict = np.array([map_dict.get(x, x) for x in y_pred])
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    
    return acc, f1_macro


def compute_all_metrics(y_true, y_pred):
    """
    Compute all five clustering metrics: ACC, NMI, AMI, ARI, F1.
    
    
    Returns
    -------
    dict
        Dictionary with ACC, NMI, AMI, ARI, F1
    """
    from sklearn.preprocessing import LabelEncoder
    
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Encode string labels for sklearn metrics
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
        'NMI': normalized_mutual_info_score(y_true_encoded, y_pred_encoded, average_method='arithmetic'),
        'AMI': adjusted_mutual_info_score(y_true_encoded, y_pred_encoded, average_method='arithmetic'),
        'ARI': adjusted_rand_score(y_true_encoded, y_pred_encoded),
        'F1': f1
    }


def setup_imports():
    import scanpy as sc
    import torch
    
    parent_directory = os.path.abspath('..')
    if parent_directory not in sys.path:
        sys.path.append(parent_directory)
    
    import Simulation_functions
    import functions
    
    simulate_input_modules = getattr(Simulation_functions, 'simulate_input_modules', None)
    create_sparse_matrix_with_inter_module_variance = getattr(
        Simulation_functions, 'create_sparse_matrix_with_inter_module_variance', None
    )
    
    modules_to_community_dict = getattr(
        Simulation_functions, 'modules_to_community_dict', 
        getattr(functions, 'modules_to_community_dict', None)
    )
    
    # Import CAKE modules
    comem_empirical_dir = os.path.join(parent_directory, 'CoMemEmpirical')
    cake_dir = os.path.join(comem_empirical_dir, 'CAKE')
    
    sys.path.insert(0, cake_dir)
    from loaders import CellDatasetPseudoLabel, get_anchor
    from moco import Encoder, MoCo
    from pretrain import pretrain as cake_pretrain
    from cluster import get_pseudo_label
    from train_KD import DistillerLoss, get_prediction, train_distiller, train_teacher
    from utils import set_seed, yaml_config_hook
    sys.path.remove(cake_dir)
    
    return {
        'sc': sc,
        'torch': torch,
        'parent_directory': parent_directory,
        'comem_empirical_dir': comem_empirical_dir,
        'cake_dir': Path(cake_dir),
        'simulate_input_modules': simulate_input_modules,
        'create_sparse_matrix_with_inter_module_variance': create_sparse_matrix_with_inter_module_variance,
        'modules_to_community_dict': modules_to_community_dict,
        'CellDatasetPseudoLabel': CellDatasetPseudoLabel,
        'get_anchor': get_anchor,
        'Encoder': Encoder,
        'MoCo': MoCo,
        'cake_pretrain': cake_pretrain,
        'get_pseudo_label': get_pseudo_label,
        'DistillerLoss': DistillerLoss,
        'get_prediction': get_prediction,
        'train_distiller': train_distiller,
        'train_teacher': train_teacher,
        'set_seed': set_seed,
        'yaml_config_hook': yaml_config_hook,
    }


def generate_simulated_data(sim_params, imports):
    """
    Generate simulated scRNA-seq data.
    
    Parameters
    ----------
    sim_params : dict
        Simulation parameters from CSV row
    imports : dict
        Module imports
        
    Returns
    -------
    tuple
        (raw_counts_df, true_labels, node_list, num_clusters)
    """
    print("Generating simulated data...")
    
    # Generate modules
    modules = imports['simulate_input_modules'](
        sim_params['rows'], sim_params['cols'], sim_params['num_modules'],
        sim_params['avg_genes_per_module'], sim_params['avg_cells_per_module']
    )
    
    # Create sparse matrix with inter-module variance
    sparse_data = imports['create_sparse_matrix_with_inter_module_variance'](
        sim_params['rows'], sim_params['cols'], modules, sim_params['target_density'],
        sim_params['module_density'], sim_params['inter_module_density'],
        sim_params['inter_module_connection_probability'], sim_params['lambda_background'],
        sim_params['lambda_module'], sim_params['inter_module_lambda']
    )
    
    # Process data
    data_df_community = pd.DataFrame.sparse.from_spmatrix(sparse_data.T)
    community_assignments = imports['modules_to_community_dict'](modules, sim_params['cols'])
    data_df_community['community'] = list(community_assignments.values())
    data_df_community = data_df_community.T
    
    # Shuffle columns
    np.random.seed(CAKE_PARAMS['seed'])
    shuffled_column_names = np.random.permutation(data_df_community.columns.tolist())
    shuffled_df_community = data_df_community[shuffled_column_names]
    
    # Filter zero expression (use fixed version for pandas 2.0+ compatibility)
    non_zero_df = filter_zero_expression_community_fixed(shuffled_df_community)
    raw_counts_array = np.array(non_zero_df.iloc[:-1].values)
    community_assignments = dict(zip(non_zero_df.columns, non_zero_df.loc['community']))
    
    # Get node and gene lists
    node_list = list(non_zero_df.columns)
    gene_list = list(non_zero_df.index[:-1])
    num_clusters = sim_params['num_modules']
    
    # Get ground truth labels
    sorted_dict = {k: community_assignments[k] for k in node_list}
    true_labels = list(sorted_dict.values())
    
    # Create raw counts DataFrame
    raw_counts_df = pd.DataFrame(raw_counts_array, index=gene_list, columns=node_list)
    
    print(f"  Data shape: {raw_counts_array.shape} (genes x cells)")
    print(f"  Number of cells: {len(node_list)}")
    print(f"  Number of genes: {len(gene_list)}")
    print(f"  Number of clusters: {num_clusters}")
    
    return raw_counts_df, true_labels, node_list, num_clusters


def run_cake(raw_counts_df, true_labels, node_list, num_clusters, row_idx, imports):
    """
    Run CAKE clustering with optimized hyperparameters.
    
    Parameters
    ----------
    raw_counts_df : pd.DataFrame
        Raw count matrix (genes x cells)
    true_labels : list
        Ground truth cluster labels
    node_list : list
        Cell identifiers
    num_clusters : int
        Expected number of clusters
    row_idx : int
        Row index for naming
    imports : dict
        Module imports
        
    Returns
    -------
    dict
        Dictionary with predictions and metrics
    """
    sc = imports['sc']
    torch = imports['torch']
    cake_dir = imports['cake_dir']
    
    print("\nRunning CAKE with optimized hyperparameters...")
    start_time = time.time()
    
    seq_data_name = f"sim_cake_row_{row_idx}"
    
    # Preprocess: normalize, log-transform, select HVGs
    adata_cake = sc.AnnData(raw_counts_df.T.copy())
    sc.pp.normalize_total(adata_cake, target_sum=1e4)
    sc.pp.log1p(adata_cake)
    
    # Handle case where simulated data has fewer genes than target HVG count
    total_genes = adata_cake.n_vars
    target_hvg = CAKE_PARAMS['num_genes']
    
    if total_genes < target_hvg:
        print(f"  WARNING: Total genes ({total_genes}) < target HVG ({target_hvg})")
        print(f"  Using all {total_genes} genes instead of HVG selection")
        # Skip HVG selection entirely - use all genes
        n_hvg = total_genes
    else:
        n_hvg = target_hvg
        sc.pp.highly_variable_genes(adata_cake, n_top_genes=n_hvg)
        adata_cake = adata_cake[:, adata_cake.var.highly_variable].copy()
    
    print(f"  Data shape after preprocessing: {adata_cake.shape}")
    
    # Generate initial pseudo-labels via spectral clustering
    n_pcs = max(1, min(50, adata_cake.n_vars - 1))
    sc.pp.pca(adata_cake, n_comps=n_pcs, random_state=CAKE_PARAMS['seed'])
    
    spectral = SpectralClustering(
        n_clusters=num_clusters, 
        random_state=CAKE_PARAMS['seed'], 
        n_init=10
    )
    spectral_labels = spectral.fit_predict(adata_cake.obsm['X_pca'])
    print(f"  Spectral initialization: {len(np.unique(spectral_labels))} clusters")
    
    # Prepare CAKE data directory
    dest_dir = cake_dir / "data" / "h5_nested_dir" / seq_data_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    h5_path = dest_dir / "data.h5"
    
    # Convert labels to integers
    def _to_int_label(lbl):
        if isinstance(lbl, str) and lbl.upper() == 'NA':
            return 100
        if isinstance(lbl, (int, np.integer)) and lbl < 0:
            return 100
        return int(lbl)
    
    true_labels_for_h5 = np.array([_to_int_label(lbl) for lbl in true_labels], dtype=np.int32)
    
    # Prepare data for HDF5
    expr = adata_cake.X.astype(np.float32)
    cell_names = adata_cake.obs_names.astype(str).to_numpy().astype("S")
    gene_names = adata_cake.var_names.astype(str).to_numpy().astype("S")
    spectral_str = np.array([f"Cluster_{i}" for i in spectral_labels]).astype("S")
    
    # Write HDF5 file
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("exprs", data=expr)
        obs_group = f.create_group("obs")
        obs_group.create_dataset("cell_type1", data=spectral_str)
        obs_group.create_dataset("Group", data=spectral_labels.astype(np.int32))
        obs_group.create_dataset("true_group", data=true_labels_for_h5)
        f.create_dataset("obs_names", data=cell_names)
        var_group = f.create_group("var")
        var_group.create_dataset("gene_id", data=gene_names)
        f.create_dataset("var_names", data=gene_names)
        f.create_group("uns")
    
    # Write YAML config with optimized parameters
    config_dir = cake_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    model_path = (cake_dir / "states" / seq_data_name).as_posix()
    root_dir = (cake_dir / "data").as_posix()
    
    latent_str = str(CAKE_PARAMS['latent_features'])
    config_content = f"""seed: {CAKE_PARAMS['seed']}
workers: 0

K: {CAKE_PARAMS['K']}
max_element: {len(node_list)}
m: {CAKE_PARAMS['m']}
n: {CAKE_PARAMS['n_neighbors']}
c: 1
T: {CAKE_PARAMS['T']}
p: {CAKE_PARAMS['p']}
lam: {CAKE_PARAMS['lam']}
alpha: {CAKE_PARAMS['alpha']}
kd_alpha: {CAKE_PARAMS['kd_alpha']}
kd_temperature: {CAKE_PARAMS['kd_temperature']}
learning_rate: {CAKE_PARAMS['learning_rate']}
num_genes: {adata_cake.shape[1]}
latent_feature: {latent_str}

epochs: {CAKE_PARAMS['epochs']}
classnum: {num_clusters}
start_epoch: 1
batch_size: {CAKE_PARAMS['batch_size']}
resolution: {CAKE_PARAMS['resolution']}
teacher_epochs: {CAKE_PARAMS['teacher_epochs']}
distill_epochs: {CAKE_PARAMS['distill_epochs']}

reload: False
flag: aug_nn
model_path: {model_path}
data_type: h5_nested
dataset_name: {seq_data_name}
root_dir: {root_dir}
"""
    
    config_path = config_dir / f"config_{seq_data_name}.yaml"
    config_path.write_text(config_content)
    
    print(f"  Config: epochs={CAKE_PARAMS['epochs']}, lr={CAKE_PARAMS['learning_rate']}, "
          f"batch_size={CAKE_PARAMS['batch_size']}")
    
    # Run CAKE pipeline
    with working_directory(cake_dir):
        cfg = imports['yaml_config_hook'](str(config_path))
        cfg.setdefault("teacher_epochs", CAKE_PARAMS['teacher_epochs'])
        cfg.setdefault("distill_epochs", CAKE_PARAMS['distill_epochs'])
        cfg.setdefault("workers", 0)
        args = SimpleNamespace(**cfg)
        args.config_path = str(config_path)
        
        imports['set_seed'](args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        
        # Phase 1: MoCo pretraining
        print("  Phase 1: MoCo pretraining...")
        moco_model, _ = imports['cake_pretrain'](args, device=device)
        
        # Phase 2: Get pseudo labels
        print("  Phase 2: Getting pseudo labels...")
        adata_embedding, adata, y_encoded, leiden_pred, kmeans_pred, val_loader = \
            imports['get_pseudo_label'](args, moco_model, device=device)
        
        # Convert leiden predictions
        try:
            leiden_array = np.array(list(map(int, leiden_pred)))
        except ValueError:
            leiden_array = np.array([int(float(x)) for x in leiden_pred])
        
        pretrain_metrics = compute_all_metrics(true_labels_for_h5, leiden_array)
        print(f"  MoCo+Leiden metrics: ARI={pretrain_metrics['ARI']:.4f}, "
              f"NMI={pretrain_metrics['NMI']:.4f}")
        
        # Phase 3: Anchor selection
        print("  Phase 3: Anchor selection...")
        adata, adata_embedding = imports['get_anchor'](
            adata, adata_embedding,
            pseudo_label='leiden',
            seed=args.seed,
            k=30,
            percent=0.5,
        )
        
        train_adata = adata[adata.obs.leiden_density_status == 'low', :].copy()
        test_adata = adata[adata.obs.leiden_density_status == 'high', :].copy()
        
        train_dataset = imports['CellDatasetPseudoLabel'](
            train_adata, pseudo_label='leiden', oversample_flag=True, seed=args.seed
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        
        # Phase 4: Teacher-student training
        print("  Phase 4: Teacher training...")
        pseudo_cluster_count = len(np.unique(leiden_pred))
        
        teacher = imports['Encoder'](
            in_features=adata.shape[1],
            num_cluster=pseudo_cluster_count,
            latent_features=args.latent_feature,
            device=device,
            p=args.p,
        ).to(device)
        
        student = imports['Encoder'](
            in_features=adata.shape[1],
            num_cluster=pseudo_cluster_count,
            latent_features=args.latent_feature,
            device=device,
            p=args.p,
        ).to(device)
        
        # Initialize teacher from checkpoint
        for name, param in teacher.named_parameters():
            if name not in {"fc.weight", "fc.bias"}:
                param.requires_grad = False
        teacher.fc.weight.data.normal_(mean=0.0, std=0.01)
        teacher.fc.bias.data.zero_()
        
        checkpoint_path = Path(args.model_path) / f"seed_{args.seed}" / f"checkpoint_{args.epochs}.tar"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CAKE checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['net']
        for key in list(state_dict.keys()):
            if key.startswith("encoder_k.") and not key.startswith("encoder_k.fc"):
                state_dict[key[len("encoder_k."):]] = state_dict[key]
            del state_dict[key]
        teacher.load_state_dict(state_dict, strict=False)
        student.load_state_dict(teacher.state_dict(), strict=False)
        
        # Train teacher
        teacher_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, teacher.parameters()),
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        teacher_criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(1, CAKE_PARAMS['teacher_epochs'] + 1):
            loss = imports['train_teacher'](
                train_loader, teacher, teacher_criterion, teacher_optimizer, device, epoch
            )
            if epoch % 10 == 0:
                print(f"    Teacher epoch {epoch}/{CAKE_PARAMS['teacher_epochs']}, loss={loss:.4f}")
        
        # Phase 5: Knowledge distillation
        print("  Phase 5: Knowledge distillation...")
        distiller_loss = imports['DistillerLoss'](
            alpha=args.kd_alpha, 
            temperature=args.kd_temperature
        )
        distiller_optimizer = torch.optim.Adam(
            student.parameters(), 
            lr=args.learning_rate, 
            weight_decay=0.0
        )
        
        for param in teacher.parameters():
            param.requires_grad = False
        
        for epoch in range(1, CAKE_PARAMS['distill_epochs'] + 1):
            loss = imports['train_distiller'](
                train_loader, student, teacher, distiller_loss, distiller_optimizer, device
            )
            if epoch % 10 == 0:
                print(f"    Student epoch {epoch}/{CAKE_PARAMS['distill_epochs']}, loss={loss:.4f}")
        
        # Get final predictions
        student.eval()
        predicted_labels = np.array(
            imports['get_prediction'](student, device, val_loader), 
            dtype=np.int32
        )
    
    elapsed_time = time.time() - start_time
    print(f"  CAKE completed in {elapsed_time/60:.1f} minutes")
    
    # Compute all metrics
    metrics = compute_all_metrics(true_labels, predicted_labels)
    
    return {
        'predictions': predicted_labels,
        'metrics': metrics,
        'elapsed_time': elapsed_time
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CAKE Benchmark for Simulated scRNA-seq Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python Benchmark_CAKE_Simulated.py --experiment ngenes --row_idx 0
    python Benchmark_CAKE_Simulated.py --experiment nmodules --row_idx 5
    python Benchmark_CAKE_Simulated.py --experiment ngenes --row_idx 0 --append
        """
    )
    parser.add_argument(
        '--experiment', 
        type=str, 
        required=True,
        choices=['ngenes', 'nmodules'],
        help='Experiment type: ngenes or nmodules'
    )
    parser.add_argument(
        '--row_idx', 
        type=int, 
        required=True,
        help='Row index from the simulation parameters CSV'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing output CSV instead of creating new'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='CAKE_Simulated_Results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Get experiment configuration
    exp_config = EXPERIMENT_CONFIG[args.experiment]
    
    print("=" * 70)
    print(f"CAKE Benchmark - {exp_config['description']}")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Row index: {args.row_idx}")
    print(f"Results file: {exp_config['results_file']}")
    print(f"Output file: {exp_config['output_file']}")
    print("=" * 70)
    
    # Setup imports
    imports = setup_imports()
    
    # Set random seeds
    np.random.seed(CAKE_PARAMS['seed'])
    imports['torch'].manual_seed(CAKE_PARAMS['seed'])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load simulation parameters
    if not os.path.exists(exp_config['results_file']):
        print(f"ERROR: Results file not found: {exp_config['results_file']}")
        sys.exit(1)
    
    params_df = pd.read_csv(exp_config['results_file'])
    
    if args.row_idx >= len(params_df):
        print(f"ERROR: Row index {args.row_idx} out of range (max: {len(params_df)-1})")
        sys.exit(1)
    
    param_row = params_df.iloc[args.row_idx]
    
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
    
    # Generate simulated data
    raw_counts_df, true_labels, node_list, num_clusters = generate_simulated_data(
        sim_params, imports
    )
    
    # Run CAKE
    try:
        results = run_cake(
            raw_counts_df, true_labels, node_list, num_clusters,
            args.row_idx, imports
        )
        
        metrics = results['metrics']
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"ACC: {metrics['ACC']:.4f}")
        print(f"NMI: {metrics['NMI']:.4f}")
        print(f"AMI: {metrics['AMI']:.4f}") 
        print(f"ARI: {metrics['ARI']:.4f}")
        print(f"F1:  {metrics['F1']:.4f}")
        print(f"Runtime: {results['elapsed_time']/60:.1f} minutes")
        
    except Exception as e:
        print(f"\nERROR running CAKE: {e}")
        import traceback
        traceback.print_exc()
        metrics = {'ACC': np.nan, 'NMI': np.nan, 'AMI': np.nan, 'ARI': np.nan, 'F1': np.nan}
        results = {'elapsed_time': np.nan}
    
    # Prepare output row
    output_row = {
        'row_idx': args.row_idx,
        'experiment': args.experiment,
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
        'CAKE_ACC': metrics['ACC'],
        'CAKE_NMI': metrics['NMI'],
        'CAKE_AMI': metrics['AMI'], 
        'CAKE_ARI': metrics['ARI'],
        'CAKE_F1': metrics['F1'],
        'runtime_minutes': results['elapsed_time'] / 60 if not np.isnan(results.get('elapsed_time', np.nan)) else np.nan,
        'seed': CAKE_PARAMS['seed'],
        'epochs': CAKE_PARAMS['epochs'],
        'learning_rate': CAKE_PARAMS['learning_rate'],
        'batch_size': CAKE_PARAMS['batch_size'],
    }
    
    # Save results
    output_file = output_dir / exp_config['output_file']
    
    if args.append and output_file.exists():
        existing_df = pd.read_csv(output_file)
        # Check if row already exists
        if args.row_idx in existing_df['row_idx'].values:
            # Update existing row
            existing_df.loc[existing_df['row_idx'] == args.row_idx, output_row.keys()] = output_row.values()
            output_df = existing_df
        else:
            # Append new row
            output_df = pd.concat([existing_df, pd.DataFrame([output_row])], ignore_index=True)
    else:
        output_df = pd.DataFrame([output_row])
    
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()