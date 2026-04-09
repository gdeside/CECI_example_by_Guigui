#!/bin/bash
#SBATCH --array=0-49
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guillaume.deside@uclouvain.be
#SBATCH --job-name=R_Parallel_Sim
#SBATCH --output=log/Output_%a.txt
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=4

module load R

Rscript run.R