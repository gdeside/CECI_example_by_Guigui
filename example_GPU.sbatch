#!/bin/bash
#SBATCH --job-name=Example_GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=example_GPU.txt

module load releases/2023a

module load Python/3.11.3-GCCcore-12.3.0

source env/bin/activate

python3 example_GPU.py