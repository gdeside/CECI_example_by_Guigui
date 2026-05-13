#!/bin/bash
#SBATCH --job-name=Example_R
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=guillaume.deside@uclouvain.be
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --output=example_R.txt


module load R

Rscript example_R.R