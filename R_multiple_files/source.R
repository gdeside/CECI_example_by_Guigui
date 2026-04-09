# ============================================================
# FILE: source.R (WITH SAFER SEED)
# ============================================================

# ---- 1. SIMULATION PARAMETERS ----
n_tasks <- 50
M <- 1000
M_per_task <- M / n_tasks

n_samples <- 500
n_features <- 100
n_informative <- 10

# ---- 2. THE CORE FUNCTION ----
run_one_simulation <- function(iter_id) {
  # Use seed offset to avoid problematic low seeds
  set.seed(iter_id + 12345)
  
  # Generate data
  X <- matrix(rnorm(n_samples * n_features), 
              nrow = n_samples, ncol = n_features)
  
  # True coefficients
  beta_true <- c(rnorm(n_informative, mean = 2, sd = 0.5),
                 rep(0, n_features - n_informative))
  
  # Generate response
  y <- X %*% beta_true + rnorm(n_samples)
  
  # Fit model
  library(glmnet)
  fit <- cv.glmnet(X, y, alpha = 0)
  beta_est <- as.vector(coef(fit, s = "lambda.min"))[-1]
  
  # Calculate metrics
  mse <- mean((beta_est - beta_true)^2)
  n_correct_zeros <- sum((beta_true == 0) & (abs(beta_est) < 0.1))
  n_correct_nonzeros <- sum((beta_true != 0) & (abs(beta_est) > 0.1))
  
  # Return results
  return(list(
    iteration = iter_id,
    mse = mse,
    correct_zeros = n_correct_zeros,
    correct_nonzeros = n_correct_nonzeros
  ))
}

cat("source.R loaded successfully\n")
cat(sprintf("Total simulations: %d\n", M))
cat(sprintf("Simulations per task: %d\n", M_per_task))
cat(sprintf("Sample size: %d, Features: %d, Informative: %d\n", 
            n_samples, n_features, n_informative))