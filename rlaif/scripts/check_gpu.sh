#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=15:00:00
#SBATCH --partition=gpupart_v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
nvidia-smi