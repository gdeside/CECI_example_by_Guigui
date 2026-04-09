# ============================================================
# FILE: out.R
# PURPOSE: Aggregate results from all tasks
# RUN THIS AFTER all tasks have completed
# ============================================================

cat("============================================================\n")
cat("Results Aggregation - CECI Cluster Simulations\n")
cat("============================================================\n\n")

# ---- 1. LOAD PARAMETERS ----
source("source.R")

# ---- 2. COLLECT RESULTS FROM ALL TASKS ----
cat("[STEP 1] Collecting results from all tasks...\n\n")

all_results <- list()
missing_tasks <- c()

for (task_id in 0:(n_tasks - 1)) {
    filename <- paste0("Results/result_", task_id, ".rds")
    
    if (file.exists(filename)) {
        all_results[[task_id + 1]] <- readRDS(filename)
        cat(sprintf("  Task %2d: OK\n", task_id))
    } else {
        cat(sprintf("  Task %2d: MISSING!\n", task_id))
        missing_tasks <- c(missing_tasks, task_id)
    }
}

# Combine all results into one data frame
results_df <- do.call(rbind, all_results)

cat(sprintf("\n[COLLECTION SUMMARY]\n"))
cat(sprintf("  - Total simulations collected: %d (expected: %d)\n", 
            nrow(results_df), M))
cat(sprintf("  - Missing tasks: %d\n", length(missing_tasks)))

if (length(missing_tasks) > 0) {
    cat(sprintf("  - Missing task IDs: %s\n", paste(missing_tasks, collapse = ", ")))
    cat("\n  WARNING: Not all tasks completed successfully!\n")
}

# ---- 3. COMPUTE STATISTICS ----
cat("\n[STEP 2] Computing statistics...\n\n")

cat("============================================================\n")
cat("SIMULATION RESULTS\n")
cat("============================================================\n\n")

# MSE statistics
cat("[Mean Squared Error]\n")
cat(sprintf("  Mean:   %.4f\n", mean(results_df$mse)))
cat(sprintf("  SD:     %.4f\n", sd(results_df$mse)))
cat(sprintf("  Median: %.4f\n", median(results_df$mse)))
cat(sprintf("  Min:    %.4f\n", min(results_df$mse)))
cat(sprintf("  Max:    %.4f\n", max(results_df$mse)))

# Zero identification
cat(sprintf("\n[Correct Zero Identification]\n"))
cat(sprintf("  Mean:   %.2f / %d (%.1f%%)\n", 
            mean(results_df$correct_zeros), 
            n_features - n_informative,
            100 * mean(results_df$correct_zeros) / (n_features - n_informative)))
cat(sprintf("  SD:     %.2f\n", sd(results_df$correct_zeros)))
cat(sprintf("  Median: %.0f\n", median(results_df$correct_zeros)))

# Non-zero identification
cat(sprintf("\n[Correct Non-Zero Identification]\n"))
cat(sprintf("  Mean:   %.2f / %d (%.1f%%)\n", 
            mean(results_df$correct_nonzeros), 
            n_informative,
            100 * mean(results_df$correct_nonzeros) / n_informative))
cat(sprintf("  SD:     %.2f\n", sd(results_df$correct_nonzeros)))
cat(sprintf("  Median: %.0f\n", median(results_df$correct_nonzeros)))

# ---- 4. CREATE SUMMARY TABLE ----
summary_stats <- data.frame(
    Metric = c("MSE", "Correct Zeros", "Correct Non-Zeros"),
    Mean = c(mean(results_df$mse), 
             mean(results_df$correct_zeros),
             mean(results_df$correct_nonzeros)),
    SD = c(sd(results_df$mse),
           sd(results_df$correct_zeros),
           sd(results_df$correct_nonzeros)),
    Min = c(min(results_df$mse),
            min(results_df$correct_zeros),
            min(results_df$correct_nonzeros)),
    Max = c(max(results_df$mse),
            max(results_df$correct_zeros),
            max(results_df$correct_nonzeros))
)

# ---- 5. SAVE AGGREGATED RESULTS ----
cat("\n[STEP 3] Saving results...\n\n")

# Save full results
save(results_df, summary_stats, 
     file = "final_results.RData")
cat("  - Saved: final_results.RData\n")

# Save summary as CSV
write.csv(summary_stats, "summary_statistics.csv", row.names = FALSE)
cat("  - Saved: summary_statistics.csv\n")

# Save individual results as CSV for easy viewing
write.csv(results_df, "all_simulations.csv", row.names = FALSE)
cat("  - Saved: all_simulations.csv\n")

cat("\n============================================================\n")
cat("AGGREGATION COMPLETED SUCCESSFULLY!\n")
cat("============================================================\n")