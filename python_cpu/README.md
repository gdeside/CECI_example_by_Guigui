# Python CPU Example

Demonstrates multi-core CPU parallelism on the CECI cluster using Python and scikit-learn.

## What it does

Runs a **Random Forest grid search** with 5-fold cross-validation over 8 hyperparameter combinations (40 total fits). The script detects the number of CPUs allocated by SLURM and uses all of them (`n_jobs=-1`).

## Files

| File | Description |
|------|-------------|
| `example_CPU.py` | Grid search script using scikit-learn |
| `example_CPU.sbatch` | SLURM submission script |

## SLURM resources

| Parameter | Value |
|-----------|-------|
| CPUs | 25 |
| Memory | 400 MB/CPU (≈ 10 GB total) |
| Time limit | 30 min |
| Module | Python/3.11.3-GCCcore-12.3.0 |

## How to run

```bash
# Activate your virtual environment first (created once):
module load releases/2023a
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv env
source env/bin/activate
pip install scikit-learn

# Submit the job
sbatch example_CPU.sbatch
```

## Output

Results are written to `results.txt` in the working directory.
