import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import itertools
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr,kendalltau
from joblib import Parallel, delayed


def filter_zero_expression(matrix,thres=0):
    """
    Filter out genes (rows) and cells (columns) with zero expression.

    :param matrix: A sparse matrix
    :return: Filtered sparse matrix
    """
    # Convert to dense matrix for processing
    dense_matrix = matrix.toarray()

    # Filter out genes (rows) with zero expression across all cells
    gene_expression_sum = np.sum(dense_matrix, axis=1)
    genes_to_keep = gene_expression_sum > thres
    filtered_matrix = dense_matrix[genes_to_keep]

    # Filter out cells (columns) with zero expression across all genes
    cell_expression_sum = np.sum(filtered_matrix, axis=0)
    cells_to_keep = cell_expression_sum > thres
    filtered_matrix = filtered_matrix[:, cells_to_keep]

    return sp.csr_matrix(filtered_matrix)

'''
def normalize_matrix_cpm(matrix):
    """
    Normalize the matrix using Counts Per Million (CPM) normalization.

    :param matrix: A sparse matrix
    :return: Normalized matrix
    """
    dense_matrix = matrix.toarray()
    # Sum across rows (genes) for each column (cell)
    column_sums = np.sum(dense_matrix, axis=0)
    column_sums[column_sums == 0] = 1
    normalized_matrix = 1e6 * dense_matrix / column_sums
    return sp.csr_matrix(normalized_matrix)

'''

def normalize_sparsematrix_cpm(matrix):
    """
    Normalize the matrix using Counts Per Million (CPM) normalization.

    :param matrix: A sparse matrix
    :return: Normalized matrix
    """
    dense_matrix = matrix.toarray()
    column_sums = np.sum(dense_matrix, axis=0)
    column_sums[column_sums == 0] = 1
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
    column_sums[column_sums == 0] = 1
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


def compute_matrix_density(sparse_matrix):
    """
    Compute the density of a sparse matrix.

    :param sparse_matrix: A scipy sparse matrix
    :return: Density of the matrix
    """
    non_zero_elements = sparse_matrix.nnz

    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]

    density = non_zero_elements / total_elements

    return density




def modules_to_community_dict(modules, total_columns):
    """
    Convert modules to community membership dictionary.

    :param modules: List of module definitions as [start_row, end_row), [start_column, end_column)
    :param total_columns: Total number of columns (cells)
    :return: Dictionary mapping column indices to module/community membership (or "NA" if not a member)
    """
    community_membership = {}
    
    for col_index in range(total_columns):
        for module_index, (start_row, end_row, start_column, end_column) in enumerate(modules):
            if start_column <= col_index < end_column:
                community_membership[col_index] = module_index
                break
        else:
            community_membership[col_index] = "NA"
    
    return community_membership




def create_graph_from_sparse_matrix_HT(matrix, threshold_percentile=99):
    """
    Create a graph from a sparse matrix based on correlation threshold.

    :param matrix: The sparse matrix
    :param threshold: Correlation threshold to create an edge
    :return: A graph (NetworkX object)
    """
    df = pd.DataFrame(matrix.toarray())
    correlations = df.corr().abs()

    threshold = np.percentile(correlations, threshold_percentile)

    graph = nx.Graph()
    graph.add_nodes_from(range(correlations.shape[0]))

    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[1]):
            if correlations.iloc[i, j] >= threshold:
                graph.add_edge(i, j, weight=correlations.iloc[i, j])

    graph_weighted = nx.Graph()
    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[1]):
            graph_weighted.add_edge(i, j, weight=correlations.iloc[i, j])
                
    return graph,graph_weighted

'''
def create_correlation_network(matrix, threshold_percentile=99):
    """
    Create a graph from a sparse matrix based on correlation threshold.

    :param matrix: Numpy array gene by cells
    :param threshold: Correlation threshold to filter edges
    :return: Filtered numpy array correlation network
    """
    cell_coexpression = abs(np.corrcoef(matrix.T))
    cell_coexpression_flat=cell_coexpression.flatten()
    
    cell_coexpression_weighted=deepcopy(cell_coexpression)
    
    threshold = np.percentile(cell_coexpression_flat, threshold_percentile)
    cell_coexpression[cell_coexpression<threshold]=0
    cell_coexpression_filtered=cell_coexpression    
    return cell_coexpression_filtered,cell_coexpression_weighted
'''






def create_correlation_network(matrix, threshold_percentile=99, method='pearson'):
    # assume matrix columns are the cells
    if method == 'pearson':
        cell_coexpression = abs(np.corrcoef(matrix.T))
    elif method == 'spearman':
        corr, _ = spearmanr(matrix)
        cell_coexpression = abs(corr)
    elif method == 'cosine':
        distance_matrix = pdist(matrix.T, metric='cosine')
        cell_coexpression = 1 - squareform(distance_matrix)
    elif method == 'kendall':
        def compute_kendall(i, j):
            tau, _ = kendalltau(matrix[:, i], matrix[:, j])
            return tau

        num_cells = matrix.shape[1]
        cell_coexpression = np.zeros((num_cells, num_cells))

        results = Parallel(n_jobs=-1)(
            delayed(compute_kendall)(i, j) for i in range(num_cells) for j in range(i, num_cells)
        )

        index = 0
        for i in range(num_cells):
            for j in range(i, num_cells):
                tau = results[index]
                cell_coexpression[i, j] = abs(tau)
                cell_coexpression[j, i] = abs(tau)
                index += 1

    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    cell_coexpression_flat = cell_coexpression.flatten()
    cell_coexpression_weighted = deepcopy(cell_coexpression)

    threshold = np.percentile(cell_coexpression_flat, threshold_percentile)
    cell_coexpression[cell_coexpression < threshold] = 0
    cell_coexpression_filtered = cell_coexpression

    return cell_coexpression_filtered, cell_coexpression_weighted



def create_sparse_matrix_with_inter_module_variance(rows, cols, modules, target_density, module_density, inter_module_density, inter_module_connection_probability, lambda_background, lambda_module, inter_module_lambda):
    """
    Create a sparse matrix with predefined modules, specified overall, module density, and inter-module variance.

    :param rows: Number of rows in the matrix
    :param cols: Number of columns in the matrix
    :param modules: List of tuples, each defining a module (start_row, end_row, start_col, end_col)
    :param target_density: Target density for the sparse matrix
    :param module_density: Target density for the modules within the matrix
    :param inter_module_density: Target density for inter-module connections
    :param inter_module_connection_probability: Probability of inter-module connections
    :param lambda_background: Lambda for the Poisson distribution outside modules
    :param lambda_module: Lambda for the Poisson distribution inside modules
    :param inter_module_lambda: Lambda for the Poisson distribution between modules
    :return: A sparse matrix with the specified densities and inter-module variance
    """
    matrix = np.zeros((rows, cols))

    # Embed modules first
    for start_row, end_row, start_col, end_col in modules:
        module_rows = end_row - start_row
        module_cols = end_col - start_col
        module_size = module_rows * module_cols

        # Calculate the number of non-zero elements for the module
        module_non_zero_elements = int(module_size * module_density)
        
        # Generate signals values for the module
        module_values = np.random.poisson(lambda_module, module_non_zero_elements)
        
        # Choose random indices for the non-zero elements within the module
        module_indices = np.random.choice(module_size, module_non_zero_elements, replace=False)
        
        # Create a flat module matrix, insert values, then reshape
        module_matrix = np.zeros(module_size)
        np.put(module_matrix, module_indices, module_values)
        module_matrix = module_matrix.reshape((module_rows, module_cols))

        # Place the module in the main matrix
        matrix[start_row:end_row, start_col:end_col] = module_matrix

    # Calculate remaining number of non-zero elements for the background
    total_elements = rows * cols
    total_module_elements = sum((end_row - start_row) * (end_col - start_col) for start_row, end_row, start_col, end_col in modules)
    background_elements = total_elements - total_module_elements
    background_non_zero_elements = int(background_elements * target_density) - np.count_nonzero(matrix)

    # Generate non-zero values for the background
    if background_non_zero_elements > 0:
        background_values = np.random.poisson(lambda_background, background_non_zero_elements)

        # Select random indices for the background, excluding module indices
        flat_matrix = matrix.flatten()
        zero_indices = np.where(flat_matrix == 0)[0]
        background_indices = np.random.choice(zero_indices, background_non_zero_elements, replace=False)

        # Place the background values in the matrix
        np.put(flat_matrix, background_indices, background_values)
        matrix = flat_matrix.reshape((rows, cols))

    # Process inter-module variance
    for (module1, module2) in itertools.combinations(modules, 2):
        if np.random.rand() < inter_module_connection_probability:  # Only fill some inter-modules based on the probability
            start_row_1, end_row_1, start_col_1, end_col_1 = module1
            start_row_2, end_row_2, start_col_2, end_col_2 = module2

            # Calculate inter-module indices and size
            inter_module_rows = end_row_1 - start_row_1
            inter_module_cols = end_col_2 - start_col_2
            inter_module_size = inter_module_rows * inter_module_cols
            
            # Calculate the number of non-zero elements for the inter-module connections
            inter_module_non_zero_elements = int(inter_module_size * inter_module_density)

            # Generate non-zero values for inter-module connections
            inter_module_values = np.random.poisson(inter_module_lambda, inter_module_non_zero_elements)
            
            # Choose random indices for the non-zero elements within the inter-module connections
            inter_module_indices = np.random.choice(inter_module_size, inter_module_non_zero_elements, replace=False)

            # Create a flat inter-module matrix, insert values, then reshape and place in the main matrix
            inter_module_matrix = np.zeros(inter_module_size)
            np.put(inter_module_matrix, inter_module_indices, inter_module_values)
            inter_module_matrix = inter_module_matrix.reshape((inter_module_rows, inter_module_cols))
            matrix[start_row_1:end_row_1, start_col_2:end_col_2] = inter_module_matrix
            
    for (module2, module1) in itertools.combinations(modules, 2):
        if np.random.rand() < inter_module_connection_probability:  # Only fill some inter-modules based on the probability
            start_row_1, end_row_1, start_col_1, end_col_1 = module1
            start_row_2, end_row_2, start_col_2, end_col_2 = module2

            # Calculate inter-module indices and size
            inter_module_rows = end_row_1 - start_row_1
            inter_module_cols = end_col_2 - start_col_2
            inter_module_size = inter_module_rows * inter_module_cols
            
            # Calculate the number of non-zero elements for the inter-module connections
            inter_module_non_zero_elements = int(inter_module_size * inter_module_density)

            # Generate non-zero values for inter-module connections
            inter_module_values = np.random.poisson(inter_module_lambda, inter_module_non_zero_elements)
            
            # Choose random indices for the non-zero elements within the inter-module connections
            inter_module_indices = np.random.choice(inter_module_size, inter_module_non_zero_elements, replace=False)

            # Create a flat inter-module matrix, insert values, then reshape and place in the main matrix
            inter_module_matrix = np.zeros(inter_module_size)
            np.put(inter_module_matrix, inter_module_indices, inter_module_values)
            inter_module_matrix = inter_module_matrix.reshape((inter_module_rows, inter_module_cols))
            matrix[start_row_1:end_row_1, start_col_2:end_col_2] = inter_module_matrix
            
    return sp.csr_matrix(matrix)






def simulate_input_modules(num_rows, num_cols, num_modules, avg_genes_per_module,avg_cells_per_module):
    """
    Simulate non-overlapping input modules for the sparse matrix function with variable sizes.

    :param num_rows: Total number of rows in the matrix (genes)
    :param num_cols: Total number of columns in the matrix (cells)
    :param num_modules: Number of modules to generate
    :param avg_cells_per_module: Average number of cells (columns) per module
    :param avg_genes_per_module: Average number of genes (rows) per module
    :return: List of tuples, each defining a module (start_row, end_row, start_col, end_col)
    """
    modules = []

    # Create non-overlapping row segments
    row_segments = np.linspace(0, num_rows, num=num_modules + 1, dtype=int)

    # Create non-overlapping column segments
    col_segments = np.linspace(0, num_cols, num=num_modules + 1, dtype=int)

    for i in range(num_modules):
        start_row = row_segments[i]
        end_row = row_segments[i + 1]

        start_col = col_segments[i]
        end_col = col_segments[i + 1]

        # Calculate the expected module sizes based on Poisson distribution
        expected_row_size = np.random.poisson(avg_genes_per_module)
        expected_col_size = np.random.poisson(avg_cells_per_module)

        # Ensure there is enough interval range for genes in this row
        end_row = min(start_row + expected_row_size, end_row)
        # Ensure there is enough interval range for cells in this column
        end_col = min(start_col + expected_col_size, end_col)

        modules.append((start_row, end_row, start_col, end_col))

    return modules
