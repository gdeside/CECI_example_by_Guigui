#!/bin/bash
#SBATCH --job-name=Example_CPU
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=400
#SBATCH --mail-user=guillaume.deside@uclouvain.be
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --output=example_CPU.txt


module load releases/2023a

module load Python/3.11.3-GCCcore-12.3.0

source env/bin/activate

python3 example_CPU.py