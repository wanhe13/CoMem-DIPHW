#!/usr/bin/env Rscript
# ============================================================================
# TsImpute Benchmark on scMixology 3cl Dataset 
# ============================================================================

setwd("/projects/radlab/Wan//NSF/CoMemEmpirical/")  
options(Seurat.object.assay.version = "v3")

suppressPackageStartupMessages({
  library(tsImpute)
  library(aricode)
  library(Seurat)
  library(mclust)
})

calc_acc <- function(true_labels, clusters) {
  error_rate <- mclust::classError(clusters, true_labels)$errorRate
  return(1 - error_rate)
}

calc_f1 <- function(true_labels, clusters) {
  mapping <- mclust::classError(clusters, true_labels)$map
  predicted_labels <- mapping[as.character(clusters)]
  
  classes <- unique(true_labels)
  f1_scores <- sapply(classes, function(cls) {
    tp <- sum(predicted_labels == cls & true_labels == cls, na.rm = TRUE)
    fp <- sum(predicted_labels == cls & true_labels != cls, na.rm = TRUE)
    fn <- sum(predicted_labels != cls & true_labels == cls, na.rm = TRUE)
    
    precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
    recall    <- if (tp + fn == 0) 0 else tp / (tp + fn)
    
    if (precision + recall == 0) 0 else 2 * (precision * recall) / (precision + recall)
  })
  return(mean(f1_scores))
}

cluster_kmeans <- function(data, k = 3, n_hvg = 500, n_pcs = 10, seed = 1) {
  set.seed(seed)
  
  # Seurat preprocessing
  seurat_obj <- CreateSeuratObject(counts = data)
  seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
  seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = n_hvg, verbose = FALSE)
  seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
  seurat_obj <- RunPCA(seurat_obj, npcs = n_pcs, verbose = FALSE)
  
  # K-means on PCA
  pca_coords <- Embeddings(seurat_obj, "pca")[, 1:n_pcs]
  kmeans_result <- kmeans(pca_coords, centers = k, nstart = 25)
  
  return(kmeans_result$cluster)
}

expr_matrix <- as.matrix(read.csv("data/sce_sc_10x_qc_counts_matrix.csv", row.names = 1))
ground_truth_df <- read.csv("data/groundtruth.csv")
true_labels <- ground_truth_df[, 2]

cat(sprintf("Loaded 3cl data: %d genes x %d cells\n", nrow(expr_matrix), ncol(expr_matrix)))

K_CLUSTERS <- 3
N_RUNS <- 5

all_runs_list <- list()


for (i in 1:N_RUNS) {
  cat(sprintf("\n=== RUN %d/%d (Seed: %d) ===\n", i, N_RUNS, i))
  
  # 1. Imputation
  set.seed(i) 
  imputed_data <- tsimpute(expr_matrix, seed = i)
  
  # 2. Clustering
  cat("Clustering Original data...\n")
  clusters_original <- cluster_kmeans(expr_matrix, k = K_CLUSTERS, seed = i)
  
  cat("Clustering tsImpute data...\n")
  clusters_tsimpute <- cluster_kmeans(imputed_data, k = K_CLUSTERS, seed = i)
  
  # 3. Metrics
  run_results <- data.frame(
    Run = i,
    Method = c("Original", "tsImpute"),
    Dataset = "3cl", 
    
    ARI = c(adjustedRandIndex(true_labels, clusters_original),
            adjustedRandIndex(true_labels, clusters_tsimpute)),
    
    NMI = c(NMI(true_labels, clusters_original),
            NMI(true_labels, clusters_tsimpute)),
    
    AMI = c(AMI(true_labels, clusters_original),
            AMI(true_labels, clusters_tsimpute)),
    
    ACC = c(calc_acc(true_labels, clusters_original),
            calc_acc(true_labels, clusters_tsimpute)),
    
    F1  = c(calc_f1(true_labels, clusters_original),
            calc_f1(true_labels, clusters_tsimpute))
  )
  
  print(run_results)
  all_runs_list[[i]] <- run_results
}

# Combine all runs
final_results_df <- do.call(rbind, all_runs_list)
# ============================================================================
# Summary Statistics
# ============================================================================
cat("\n\n=== GRAND SUMMARY (Mean +/- SD) ===\n")
summary_stats <- aggregate(. ~ Method, data = final_results_df[, -1], 
                           FUN = function(x) sprintf("%.4f (+/- %.4f)", mean(x), sd(x)))
print(summary_stats)

# ============================================================================
# Save Results
# ============================================================================
dir.create("results", showWarnings = FALSE)

# Save full results (all runs)
write.csv(final_results_df, "results/TsImpute_scMixology3cl_MultiRun.csv", row.names = FALSE)

# Save summary
write.csv(summary_stats, "results/TsImpute_scMixology3cl_Summary.csv", row.names = FALSE)

cat("\nResults saved to 'results/TsImpute_scMixology3cl_MultiRun.csv'\n")