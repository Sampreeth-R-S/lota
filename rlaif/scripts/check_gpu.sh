#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output_ultrafeedback_noninstruct_8b.txt
#SBATCH --error=error_ultrafeedback_noninstruct_8b.txt
#SBATCH --time=1-12:00:00
#SBATCH --partition=dgx2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
nvidia-smi
kill -9 1207062
kill -9 1207063