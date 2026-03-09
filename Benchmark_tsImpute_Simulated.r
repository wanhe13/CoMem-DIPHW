#!/usr/bin/env Rscript
# ==============================================================================
# Benchmark Script for tsImpute on Simulated Data
# ==============================================================================
# Runs tsImpute + KMeans clustering on simulated scRNA-seq data.
# Calculates all 5 metrics: ACC, NMI, AMI, ARI, F1
# Each row_idx saves to a separate file to avoid race conditions.
#
# Usage:
#   Rscript Benchmark_tsImpute_Simulated.R <row_idx> <experiment_type> <params_csv>
#   Rscript Benchmark_tsImpute_Simulated.R 0 ngenes results/ClusteringPerformance_ngenes_5metrics.csv
# ==============================================================================

options(Seurat.object.assay.version = "v3")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RUNNING_LOCALLY <- FALSE

if (RUNNING_LOCALLY) {
  row_idx    <- 0
  exp_type   <- "ngenes"
  params_csv <- "results/ClusteringPerformance_ngenes_5metrics.csv"
} else {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 3) stop("Usage: script.R <row_idx> <exp_type> <params_csv>")
  row_idx    <- as.integer(args[1])
  exp_type   <- args[2]
  params_csv <- args[3]
}

library(tsImpute)
library(aricode)  
library(clue)    
library(future)   

plan("sequential")
options(future.globals.maxSize = 8000 * 1024^2)

cluster_accuracy <- function(y_true, y_pred) {
  # Calculate clustering accuracy using Hungarian algorithm
  
  y_true <- as.integer(as.factor(y_true))
  y_pred <- as.integer(as.factor(y_pred))
  
  D <- max(max(y_true), max(y_pred))
  w <- matrix(0, nrow = D, ncol = D)
  
  for (i in seq_along(y_pred)) {
    w[y_pred[i], y_true[i]] <- w[y_pred[i], y_true[i]] + 1
  }
  
  assignment <- solve_LSAP(max(w) - w)
  
  acc <- sum(w[cbind(seq_len(D), assignment)]) / length(y_pred)
  
  return(acc)
}

compute_f1_score <- function(y_true, y_pred) {
  # Calculate F1 score with Hungarian matching
  
  y_true_factor <- as.factor(y_true)
  y_pred_factor <- as.factor(y_pred)
  
  y_true_int <- as.integer(y_true_factor)
  y_pred_int <- as.integer(y_pred_factor)
  
  D <- max(max(y_true_int), max(y_pred_int))
  w <- matrix(0, nrow = D, ncol = D)
  
  for (i in seq_along(y_pred_int)) {
    w[y_pred_int[i], y_true_int[i]] <- w[y_pred_int[i], y_true_int[i]] + 1
  }
  
  # Hungarian algorithm
  assignment <- solve_LSAP(max(w) - w)
  
  mapping <- setNames(as.integer(assignment), seq_len(D))
  y_pred_mapped <- mapping[y_pred_int]
  
  classes <- unique(y_true_int)
  f1_scores <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    cls <- classes[i]
    tp <- sum(y_true_int == cls & y_pred_mapped == cls)
    fp <- sum(y_true_int != cls & y_pred_mapped == cls)
    fn <- sum(y_true_int == cls & y_pred_mapped != cls)
    
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1_scores[i] <- ifelse(precision + recall > 0, 
                           2 * precision * recall / (precision + recall), 0)
  }
  
  return(mean(f1_scores))
}

compute_all_metrics <- function(y_true, y_pred) {
  # Compute all 5 clustering metrics: ACC, NMI, AMI, ARI, F1
  
  # Handle NA labels (convert to "Unassigned" string)
  y_true <- as.character(y_true)
  y_true[is.na(y_true) | y_true == "NA"] <- "Unassigned"
  y_pred <- as.character(y_pred)
  
  acc <- cluster_accuracy(y_true, y_pred)
  nmi <- NMI(y_true, y_pred, variant = "sum")  # arithmetic mean variant
  ami <- AMI(y_true, y_pred)
  ari <- ARI(y_true, y_pred)
  f1  <- compute_f1_score(y_true, y_pred)
  
  return(list(ACC = acc, NMI = nmi, AMI = ami, ARI = ari, F1 = f1))
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

start_time <- Sys.time()

# ------------------------------------------------------------------------------
# 1. LOAD SIMULATION PARAMETERS
# ------------------------------------------------------------------------------
if (!file.exists(params_csv)) {
  stop(sprintf("ERROR: Params file not found: %s", params_csv))
}

params_df <- read.csv(params_csv)

if (row_idx >= nrow(params_df)) {
  stop(sprintf("ERROR: Row index %d out of range (max: %d)", row_idx, nrow(params_df) - 1))
}

param_row <- params_df[row_idx + 1, ]  # R is 1-indexed

# Get run_idx for random seed (matches Python: RANDOM_SEED = 42 + run_idx)
run_idx <- ifelse("run_idx" %in% names(param_row), param_row$run_idx, row_idx)
RANDOM_SEED <- 42 + run_idx

set.seed(RANDOM_SEED)

# Extract simulation parameters
sim_params <- list(
  rows = as.integer(param_row$rows),
  cols = as.integer(param_row$cols),
  num_modules = as.integer(param_row$num_modules),
  avg_genes_per_module = as.integer(param_row$avg_genes_per_module),
  avg_cells_per_module = as.integer(param_row$avg_cells_per_module),
  target_density = as.numeric(param_row$target_density),
  module_density = as.numeric(param_row$module_density),
  inter_module_density = as.numeric(param_row$inter_module_density),
  inter_module_connection_probability = as.numeric(param_row$inter_module_connection_probability),
  lambda_background = as.integer(param_row$lambda_background),
  lambda_module = as.integer(param_row$lambda_module),
  inter_module_lambda = as.integer(param_row$inter_module_lambda)
)

cat(sprintf("\nSimulation parameters:\n"))
cat(sprintf("  Rows (genes): %d\n", sim_params$rows))
cat(sprintf("  Cols (cells): %d\n", sim_params$cols))
cat(sprintf("  Modules: %d\n", sim_params$num_modules))
cat(sprintf("  avg_genes_per_module: %d\n", sim_params$avg_genes_per_module))

n_clusters <- sim_params$num_modules

# ------------------------------------------------------------------------------
# 2. LOAD DATA (pre-generated by Python)
# ------------------------------------------------------------------------------
data_dir <- file.path(paste0("data_", exp_type), paste0("sim_", row_idx))
counts_file <- file.path(data_dir, "counts.csv")
labels_file <- file.path(data_dir, "labels.csv")

if (!file.exists(counts_file)) {
  stop(sprintf("ERROR: Counts file not found: %s", counts_file))
}
if (!file.exists(labels_file)) {
  stop(sprintf("ERROR: Labels file not found: %s", labels_file))
}

cat(sprintf("\nLoading data from: %s\n", data_dir))
counts <- read.csv(counts_file, row.names = 1, check.names = FALSE)
labels <- read.csv(labels_file, stringsAsFactors = FALSE)

X <- as.matrix(counts)  # genes x cells
true_labels <- labels$true_label

cat(sprintf("  Data shape: %d genes x %d cells\n", nrow(X), ncol(X)))
cat(sprintf("  Number of clusters: %d\n", n_clusters))

if (nrow(X) == 0 || ncol(X) == 0) {
  stop("ERROR: Empty expression matrix!")
}

# ------------------------------------------------------------------------------
# 3. RUN tsImpute
# ------------------------------------------------------------------------------
imputed_data <- tsimpute(X, seed = RANDOM_SEED)

# ------------------------------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------------------------------

# CPM normalization 
lib_sizes <- colSums(imputed_data)
lib_sizes[lib_sizes == 0] <- 1  # Avoid division by zero
norm_data <- 1e6 * sweep(imputed_data, 2, lib_sizes, FUN = "/")

# Log-transform 
log_data <- log2(norm_data + 1)

# ------------------------------------------------------------------------------
# 5. PCA 
# ------------------------------------------------------------------------------
n_pcs <- min(30, ncol(log_data) - 1)
# prcomp expects rows=samples, cols=features -> transpose
pca_res <- prcomp(t(log_data), center = TRUE, scale. = FALSE, rank. = n_pcs)
pca_embeddings <- pca_res$x


# ------------------------------------------------------------------------------
# 6. K-MEANS CLUSTERING
# ------------------------------------------------------------------------------
kmeans_result <- kmeans(pca_embeddings, centers = n_clusters, nstart = 10, iter.max = 300)
pred_labels <- kmeans_result$cluster

# ------------------------------------------------------------------------------
# 7. COMPUTE ALL 5 METRICS
# ------------------------------------------------------------------------------
metrics <- compute_all_metrics(true_labels, pred_labels)

elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

cat("\n======================================================================\n")
cat("RESULTS\n")
cat("======================================================================\n")
cat(sprintf("tsImpute: ACC=%.4f, NMI=%.4f, ARI=%.4f, AMI=%.4f, F1=%.4f\n",
            metrics$ACC, metrics$NMI, metrics$ARI, metrics$AMI, metrics$F1))
cat(sprintf("Runtime: %.1f minutes\n", elapsed_time))

# ------------------------------------------------------------------------------
# 8. SAVE RESULTS 
# ------------------------------------------------------------------------------

# Create output directory per task
output_dir <- "tsImpute_Simulated_Results"
task_dir <- file.path(output_dir, paste0("task_", row_idx))
dir.create(task_dir, recursive = TRUE, showWarnings = FALSE)

output_file <- file.path(task_dir, paste0("tsImpute_simulated_", exp_type, ".csv"))

# Prepare output row (matching other benchmark scripts for easy merging)
output_row <- data.frame(
  row_idx = row_idx,
  run_idx = run_idx,
  rows = sim_params$rows,
  cols = sim_params$cols,
  num_modules = sim_params$num_modules,
  avg_genes_per_module = sim_params$avg_genes_per_module,
  avg_cells_per_module = sim_params$avg_cells_per_module,
  target_density = sim_params$target_density,
  module_density = sim_params$module_density,
  inter_module_density = sim_params$inter_module_density,
  inter_module_connection_probability = sim_params$inter_module_connection_probability,
  lambda_background = sim_params$lambda_background,
  lambda_module = sim_params$lambda_module,
  inter_module_lambda = sim_params$inter_module_lambda,
  tsImpute_ACC = metrics$ACC,
  tsImpute_NMI = metrics$NMI,
  tsImpute_ARI = metrics$ARI,
  tsImpute_AMI = metrics$AMI,
  tsImpute_F1 = metrics$F1,
  tsImpute_runtime_minutes = elapsed_time,
  stringsAsFactors = FALSE
)

# Save to individual file (safe for parallel execution)
write.csv(output_row, output_file, row.names = FALSE)
