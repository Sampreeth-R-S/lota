#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=15:00:00
#SBATCH --partition=gpupart_v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
source activate /home/du1/21CS30038/.conda/envs/pytorch_env
export CUDA_LAUNCH_BLOCKING=1
pip install huggingface-hub
huggingface-cli login --token "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"
export WANDB_API_KEY="51bee7d3cdc3e9154459f935c5e4ea81ad2ef813"


python ../convert_policy_to_hf_lora.py --lora_rank 8 --model_path "meta-llama/Llama-3.2-1B" --policy_path "/home/du1/21CS30038/lota/rlaif/scripts/outputs/epoch-0/policy.pt" --save_path ./epoch3/