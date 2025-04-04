#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output_eval_ARC-Challenge_noninstruct_trained.txt
#SBATCH --error=error_eval_ARC-Challenge_noninstruct_trained.txt
#SBATCH --time=15:00:00
#SBATCH --partition=dgx2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
ls /raid/pabitracs/
source ~/miniconda3/bin/activate base
ls /raid/pabitracs/Sampreeth/lota/rlaif/scripts/fine_tuned_model
export CUDA_LAUNCH_BLOCKING=1
pip install huggingface-hub
huggingface-cli login --token "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"
export WANDB_API_KEY="51bee7d3cdc3e9154459f935c5e4ea81ad2ef813"
export WANDB_MODE=offline

python ../eval_model_all.py --model "meta-llama/Meta-Llama-3-8B" --datasets "ARC-Challenge"
