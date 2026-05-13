# CECI Example by Guigui

Practical examples for running **CPU**, **GPU**, and **R** workloads on the [CECI](https://www.ceci-hpc.be/) cluster (Consortium des Équipements de Calcul Intensif — Belgian HPC infrastructure). Each example is self-contained with its own SLURM submission script and README.

## Repository structure

```
CECI_example_by_Guigui/
├── python_cpu/          # Multi-core CPU job with scikit-learn (Random Forest grid search)
├── python_gpu/          # GPU job with PyTorch (CNN on CIFAR-10)
├── R_single_job/        # Parallel R job (Random Forest cross-validation)
├── R_multiple_files/    # SLURM array job in R (elastic net simulation study)
├── job_helpers/         # Python helpers to programmatically submit SLURM jobs
└── Slides/              # Presentation slides (LaTeX/PDF)
```

## Quick overview

| Folder | Language | Parallelism | Use case |
|--------|----------|-------------|----------|
| `python_cpu` | Python | Multi-core (`n_jobs=-1`) | Hyperparameter search |
| `python_gpu` | Python | GPU (PyTorch) | Deep learning |
| `R_single_job` | R | Multi-core (`doParallel`) | Cross-validation |
| `R_multiple_files` | R | SLURM array (50 tasks) | Large simulation study |
| `job_helpers` | Python | — | Programmatic job submission |

## Getting started

1. **Clone the repository** on the cluster:
   ```bash
   git clone https://github.com/gdeside/ceci_example_by_guigui.git
   cd ceci_example_by_guigui
   ```

2. **Navigate to the example** you want to run and follow its `README.md`.

3. **Submit the job** with `sbatch <script>.sbatch`.

Each folder's README describes the required modules, expected outputs, and any setup steps (e.g. creating a Python virtual environment or installing R packages).

## Prerequisites

- Access to a CECI cluster (login via SSH)
- Basic familiarity with SLURM (`sbatch`, `squeue`, `scancel`)
- Python ≥ 3.11 or R, depending on the example

## Slides

The `Slides/` folder contains a presentation that walks through the examples and explains the CECI cluster setup.
