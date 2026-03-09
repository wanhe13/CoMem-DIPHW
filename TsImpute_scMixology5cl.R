setwd("/projects/radlab/Wan//NSF/CoMemEmpirical/")  
options(Seurat.object.assay.version = "v3")

library(tsImpute)
library(aricode)
library(Seurat)
library(mclust)  
calc_acc <- function(true_labels, clusters) {
  error_rate <- mclust::classError(clusters, true_labels)$errorRate
  return(1 - error_rate)
}


calc_f1 <- function(true_labels, clusters) {
  true_labels <- as.integer(as.factor(true_labels))
  clusters <- as.integer(as.factor(clusters))
  
  D <- max(max(true_labels), max(clusters))
  w <- matrix(0, nrow = D, ncol = D)
  
  for (i in seq_along(clusters)) {
    w[clusters[i], true_labels[i]] <- w[clusters[i], true_labels[i]] + 1
  }
  
  assignment <- clue::solve_LSAP(max(w) - w)
  
  mapping <- setNames(as.integer(assignment), seq_len(D))
  clusters_mapped <- mapping[clusters]
  
  classes <- unique(true_labels)
  f1_scores <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    cls <- classes[i]
    tp <- sum(true_labels == cls & clusters_mapped == cls)
    fp <- sum(true_labels != cls & clusters_mapped == cls)
    fn <- sum(true_labels == cls & clusters_mapped != cls)
    
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1_scores[i] <- ifelse(precision + recall > 0, 
                           2 * precision * recall / (precision + recall), 0)
  }
  
  return(mean(f1_scores))
}


cluster_kmeans <- function(data, k = 5, n_hvg = 500, n_pcs = 10, seed = 1) {
  set.seed(seed)
  
  seurat_obj <- CreateSeuratObject(counts = data)
  seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
  seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = n_hvg, verbose = FALSE)
  seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
  seurat_obj <- RunPCA(seurat_obj, npcs = n_pcs, verbose = FALSE)
  
  pca_coords <- Embeddings(seurat_obj, "pca")[, 1:n_pcs]
  kmeans_result <- kmeans(pca_coords, centers = k, nstart = 25)
  
  return(kmeans_result$cluster)
}

expr_matrix <- as.matrix(read.csv("data/sce_sc_10x_5cl_qc_counts_matrix.csv", row.names = 1))
ground_truth_df <- read.csv("data/groundtruth_5cl.csv")
true_labels <- ground_truth_df[, 2]

K_CLUSTERS <- 5
N_RUNS <- 5

all_runs_list <- list()


for (i in 1:N_RUNS) {
  cat(sprintf("\n=== RUN %d/%d (Seed: %d) ===\n", i, N_RUNS, i))
  
  # 1. Imputation (Vary seed to test stability)
  set.seed(i) 
  imputed_data <- tsimpute(expr_matrix, seed = i)
  
  # 2. Clustering Original Data
  cat("Clustering Original data...\n")
  clusters_original <- cluster_kmeans(expr_matrix, k = K_CLUSTERS, seed = i)
  
  # 3. Clustering Imputed Data
  cat("Clustering tsImpute data...\n")
  clusters_tsimpute <- cluster_kmeans(imputed_data, k = K_CLUSTERS, seed = i)
  
  # 4. Calculate Metrics for this Run
  run_results <- data.frame(
    Run = i,
    Method = c("Original", "tsImpute"),
    
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
write.csv(final_results_df, "results/TsImpute_scMixology5cl_MultiRun.csv", row.names = FALSE)

# Save summary
write.csv(summary_stats, "results/TsImpute_scMixology5cl_Summary.csv", row.names = FALSE)

cat("\nResults saved to 'results/TsImpute_scMixology5cl_MultiRun.csv'\n")