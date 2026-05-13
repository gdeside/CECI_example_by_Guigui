# R Single Job Example

Demonstrates parallel computing in R using a single SLURM job on the CECI cluster.

## What it does

Runs a **Random Forest repeated cross-validation** experiment:

- 50,000 samples × 30 features (synthetic dataset)
- 10-fold CV × 5 repeats = 50 total iterations
- Uses `parallel::makeCluster` and `doParallel` to distribute folds across CPUs

## Files

| File | Description |
|------|-------------|
| `example_R.R` | Random Forest CV script |
| `example_R.sbatch` | SLURM submission script |

## SLURM resources

| Parameter | Value |
|-----------|-------|
| CPUs | 20 |
| Memory | 4000 MB/CPU (≈ 80 GB total) |
| Time limit | 30 min |
| Module | R (default cluster version) |

## How to run

```bash
# Install required R packages once (interactive session or dedicated install job):
# install.packages(c("randomForest", "doParallel", "foreach", "caret"))

# Submit the job
sbatch example_R.sbatch
```

## Output

- `cv_results.csv` — cross-validation results in CSV format
- `cv_results.rds` — same results as an R object (for downstream analysis)
