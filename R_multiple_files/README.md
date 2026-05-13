# R Array Job Example

Demonstrates SLURM **array jobs** in R: splitting a large simulation study into many independent tasks that run in parallel across multiple nodes.

## What it does

Runs an **elastic net regression simulation study**:

- 1000 total simulations split across 50 SLURM array tasks (20 simulations each)
- Each simulation: 500 samples × 100 features (10 informative), fitted with `glmnet`
- Metrics collected: MSE, correct zero identification, correct non-zero identification

## Files

| File | Description |
|------|-------------|
| `source.R` | Shared parameters and `run_one_simulation()` function |
| `run.R` | Per-task script — reads `SLURM_ARRAY_TASK_ID`, runs its 20 simulations, saves `Results/result_[id].rds` |
| `out.R` | Aggregation script — run **after** all tasks finish to collect and summarise results |
| `submit.sbatch` | SLURM array job submission script |

## Workflow

```
submit.sbatch  →  run.R × 50 tasks (parallel)  →  Results/result_*.rds
                                                         ↓
                                                      out.R  →  final_results.RData
                                                                 summary_statistics.csv
                                                                 all_simulations.csv
```

## SLURM resources (per task)

| Parameter | Value |
|-----------|-------|
| Array tasks | 0–49 (50 tasks) |
| CPUs per task | 4 |
| Memory | 4000 MB/CPU |
| Time limit | 30 min |
| Module | R (default cluster version) |

## How to run

```bash
# 1. Create the Results directory
mkdir -p Results log

# 2. Submit the array job
sbatch submit.sbatch

# 3. Once all tasks are done, aggregate results
Rscript out.R
```

## Output

- `Results/result_[0-49].rds` — per-task raw results
- `final_results.RData` — all results in a single R object
- `summary_statistics.csv` — aggregated statistics
- `all_simulations.csv` — full simulation table
