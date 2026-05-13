# Python GPU Example

Demonstrates GPU-accelerated deep learning on the CECI cluster using PyTorch.

## What it does

Trains a **Convolutional Neural Network (CNN)** on the CIFAR-10 image dataset for 20 epochs. The script auto-detects whether a GPU is available and adjusts accordingly.

### Architecture

- 3 convolutional layers with batch normalisation and max pooling
- 2 fully connected layers
- Input: 32×32 RGB images (10 classes)

## Files

| File | Description |
|------|-------------|
| `example_GPU.py` | CNN training script using PyTorch |
| `example_GPU.sbatch` | SLURM submission script |

## SLURM resources

| Parameter | Value |
|-----------|-------|
| Partition | `gpu` |
| GPUs | 1 |
| CPUs | 4 |
| Memory | 16 GB |
| Time limit | 30 min |
| Module | Python/3.11.3-GCCcore-12.3.0 |

## How to run

```bash
# Activate your virtual environment first (created once):
module load releases/2023a
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv env
source env/bin/activate
pip install torch torchvision

# Submit the job
sbatch example_GPU.sbatch
```

## Output

Training metrics per epoch are written to `training_results.txt`.
