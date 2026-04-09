# ============================================================
# FILE: run.R (ROBUST VERSION WITH AUTO FALLBACK)
# ============================================================

library(foreach, quietly = TRUE)
library(parallel, quietly = TRUE)
library(doParallel, quietly = TRUE)

cat("============================================================\n")
cat("Parallel R Simulation - CECI Cluster\n")
cat("============================================================\n\n")

# ---- LOAD PARAMETERS ----
source("source.R")

# ---- GET TASK INFO ----
array_id <- as.numeric(Sys.getenv('SLURM_ARRAY_TASK_ID'))
cpus_per_task <- as.numeric(Sys.getenv('SLURM_CPUS_PER_TASK'))

cat(sprintf("\n[TASK INFO]\n"))
cat(sprintf("  - Task ID: %d\n", array_id))
cat(sprintf("  - CPUs allocated: %d\n", cpus_per_task))

# ---- DETERMINE ITERATIONS ----
start_iter <- array_id * M_per_task + 1
end_iter <- (array_id + 1) * M_per_task
iterations_for_this_task <- start_iter:end_iter

cat(sprintf("  - Iterations assigned: %d to %d (%d total)\n", 
            start_iter, end_iter, M_per_task))

# ---- TRY PARALLEL FIRST ----
cat(sprintf("\n[PARALLEL ATTEMPT]\n"))

cl <- makeCluster(cpus_per_task)
registerDoParallel(cl)

invisible(clusterCall(cl, function() { library(glmnet) }))

cat("  - Parallel cluster ready\n")
cat("  - Starting parallel execution...\n")

start_time <- Sys.time()
parallel_failed <- FALSE

results <- tryCatch({
  foreach(iter = iterations_for_this_task, 
          .combine = rbind,
          .packages = c("glmnet")) %dopar% {
            
            result <- run_one_simulation(iter)
            data.frame(
              iteration = result$iteration,
              mse = result$mse,
              correct_zeros = result$correct_zeros,
              correct_nonzeros = result$correct_nonzeros
            )
          }
}, error = function(e) {
  parallel_failed <<- TRUE
  cat(sprintf("\n  ⚠ PARALLEL FAILED: %s\n\n", toString(e)))
  return(NULL)
})

stopCluster(cl)

# ---- FALLBACK TO SEQUENTIAL IF NEEDED ----
if (parallel_failed) {
  cat("[SEQUENTIAL FALLBACK]\n")
  cat("  - Running sequentially (slower but reliable)...\n")
  
  results_list <- list()
  for (i in seq_along(iterations_for_this_task)) {
    iter <- iterations_for_this_task[i]
    
    if (i %% 5 == 0) {
      cat(sprintf("    Progress: %d/%d (%.0f%%)\n", 
                  i, length(iterations_for_this_task), 
                  100 * i / length(iterations_for_this_task)))
    }
    
    result <- run_one_simulation(iter)
    results_list[[i]] <- data.frame(
      iteration = result$iteration,
      mse = result$mse,
      correct_zeros = result$correct_zeros,
      correct_nonzeros = result$correct_nonzeros
    )
  }
  
  results <- do.call(rbind, results_list)
  cat("  - Sequential execution completed\n")
}

elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# ---- REPORT RESULTS ----
cat(sprintf("\n[EXECUTION SUMMARY]\n"))
if (parallel_failed) {
  cat("  - Mode: SEQUENTIAL (parallel failed)\n")
} else {
  cat("  - Mode: PARALLEL\n")
}
cat(sprintf("  - Total time: %.2f seconds\n", elapsed_time))
cat(sprintf("  - Time per iteration: %.2f seconds\n", 
            elapsed_time / M_per_task))

# ---- SAVE RESULTS ----
output_file <- paste0("Results/result_", array_id, ".rds")
saveRDS(results, file = output_file)

cat(sprintf("\n[RESULTS SAVED]\n"))
cat(sprintf("  - File: %s\n", output_file))
cat(sprintf("  - Rows: %d\n", nrow(results)))

cat("\n============================================================\n")
cat(sprintf("Task %d completed successfully!\n", array_id))
cat("============================================================\n")