library(parallel)
library(doParallel)
library(foreach)

cat("============================================================\n")
cat("R Parallel Computing Demo - Random Forest Cross-Validation\n")
cat("============================================================\n\n")

# Detect number of cores
num_cores <- detectCores()
slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK")

cat("[INFO] CPU Detection:\n")
cat(sprintf("  - Total CPUs available: %d\n", num_cores))
if (slurm_cpus != "") {
    cat(sprintf("  - SLURM allocated CPUs: %s\n", slurm_cpus))
    num_cores <- as.integer(slurm_cpus)
    cat(sprintf("  - Using SLURM allocation: %d cores\n", num_cores))
} else {
    cat("  - No SLURM allocation detected\n")
    cat("  - Using all available cores\n")
}

# Generate synthetic dataset
cat("\n[STEP 1/4] Generating synthetic dataset...\n")
set.seed(42)
n_samples <- 50000
n_features <- 30

X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
y <- factor(ifelse(rowSums(X[, 1:5]) + rnorm(n_samples, 0, 2) > 0, "A", "B"))

cat(sprintf("  - Dataset: %d samples, %d features\n", n_samples, n_features))
cat(sprintf("  - Classes: %s\n", paste(table(y), collapse=" / ")))

# Load randomForest
library(randomForest)

cat("\n[STEP 2/4] Setting up parallel backend...\n")
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat(sprintf("  - Parallel cluster created with %d workers\n", num_cores))

# Define cross-validation parameters
n_folds <- 10
n_repeats <- 5
total_iterations <- n_folds * n_repeats

cat("\n[STEP 3/4] Running parallel cross-validation...\n")
cat(sprintf("  - Cross-validation folds: %d\n", n_folds))
cat(sprintf("  - Repeats: %d\n", n_repeats))
cat(sprintf("  - Total iterations: %d\n", total_iterations))
cat(sprintf("  - Running on %d cores in parallel\n", num_cores))
cat("------------------------------------------------------------\n")

start_time <- Sys.time()

# Parallel cross-validation using foreach
results <- foreach(i = 1:total_iterations,
                   .combine = rbind,
                   .packages = c("randomForest")) %dopar% {

    # Create random train/test split
    set.seed(i)
    train_idx <- sample(1:n_samples, size = 0.8 * n_samples)
    test_idx <- setdiff(1:n_samples, train_idx)

    # Train Random Forest
    rf_model <- randomForest(X[train_idx, ], y[train_idx],
                             ntree = 100, mtry = sqrt(n_features))

    # Predict on test set
    predictions <- predict(rf_model, X[test_idx, ])
    accuracy <- sum(predictions == y[test_idx]) / length(test_idx)

    # Return results
    data.frame(iteration = i, accuracy = accuracy)
}

end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Stop cluster
stopCluster(cl)

cat("------------------------------------------------------------\n")
cat("\n[STEP 4/4] Results:\n")
cat(sprintf("  - Total time: %.2f minutes (%.1f seconds)\n",
            elapsed_time/60, elapsed_time))
cat(sprintf("  - Average time per iteration: %.2f seconds\n",
            elapsed_time/total_iterations))
cat(sprintf("  - Mean accuracy: %.4f\n", mean(results$accuracy)))
cat(sprintf("  - SD accuracy: %.4f\n", sd(results$accuracy)))
cat(sprintf("  - Min accuracy: %.4f\n", min(results$accuracy)))
cat(sprintf("  - Max accuracy: %.4f\n", max(results$accuracy)))

# Performance metrics
cat("\n[PERFORMANCE METRICS]\n")
cat(sprintf("  - Number of cores used: %d\n", num_cores))
cat(sprintf("  - Wall-clock time: %.1f seconds\n", elapsed_time))
cat(sprintf("  - Estimated speedup vs serial: %.1fx\n",
            total_iterations * (elapsed_time/total_iterations) / elapsed_time))

# Save results
write.csv(results, "cv_results.csv", row.names = FALSE)
saveRDS(results, "cv_results.rds")

cat("\n[SAVING] Results saved:\n")
cat("  - cv_results.csv\n")
cat("  - cv_results.rds\n")

cat("\n============================================================\n")
cat("ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat("============================================================\n")