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
#SBATCH --cpus-per-task=1
#SBATCH --mem=100gb
source ~/miniconda3/bin/activate base
nvidia-smi
export CUDA_LAUNCH_BLOCKING=1
pip install huggingface-hub
huggingface-cli login --token "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"
export WANDB_API_KEY="51bee7d3cdc3e9154459f935c5e4ea81ad2ef813"
export WANDB_MODE=offline

archive="${1:-meta-llama/Meta-Llama-3-8B}"
lr="${2:-1e-5}"
dataset_name="${3:-ultrafeedback}"
data_fraction="${4:-1.0}"
n_epochs="${5:-3}"
mask_dir="${6:-none}"
sparsity_ratio="${7:-0.0}"
batch_size="${8:-8}"
model="${9:-llama3}"
grad_norm="${10:-10}"
lora_rank="${11:-8}"
lora_alpha="${12:-32}"
freeze_odd_layers="${13:-false}"
freeze_even_layers="${14:-false}"

model_archive=${archive}

# Replace commas with underscores to avoid issues with file paths or names
sanitized_dataset_name=$(echo $dataset_name | tr ',' '_')
exp_name="${sanitized_dataset_name}_${archive}_${grad_norm}_${lr}_${batch_size}_${data_fraction}_${mask_dir}_${sparsity_ratio}_LORA_${lora_rank}_${lora_alpha}"
# exp_name="${sanitized_dataset_name}_${archive}_${grad_norm}_${lr}_${batch_size}_${data_fraction}_${mask_dir}_${sparsity_ratio}_DORA"
trainer_type="BasicTrainer"

python -u ../train_single_gpu_lora.py do_first_eval=False \
        mask_path=./masks/$mask_dir/${sparsity_ratio}_mask.pt \
        loss=sft \
        model=${model} \
        model.archive=${model_archive} \
        datasets=[${dataset_name}] \
        exp_name=${exp_name} \
        eval_batch_size=8 \
        sample_during_eval=false \
        lr=$lr \
        trainer=$trainer_type \
        activation_checkpointing=True \
        data_fraction=$data_fraction \
        save_every=epoch_$n_epochs \
        eval_every=100000 \
        n_epochs=$n_epochs \
        batch_size=$batch_size \
        gradient_accumulation_steps=1 \
        optimizer=RMSprop \
        grad_norm_strategy=even \
        max_grad_norm=$grad_norm \
        lora_rank=$lora_rank \
        lora_alpha=$lora_alpha \
        freeze_odd_layers=$freeze_odd_layers \
        freeze_even_layers=$freeze_even_layers

# python ../convert_policy_to_hf_lora.py --lora_rank ${lora_rank} --model_path ${model_archive} --policy_path ${model_save_path}/epoch-$n_epochs/policy.pt --save_path ${model_save_path}/epoch-$n_epochs/

# python ../eval_model_all.py --model "${model_save_path}/epoch-$n_epochs/" --datasets "${dataset_name}"
# python ../eval_model_all.py --model "/root/Tests/DLTH_LoTA/output/epoch-2/" --datasets "gsm8k"

# python ../eval_model_all.py --model "meta-llama/Llama-3.2-1B" --datasets "gsm8k"
# python ../eval_model_all.py --model "meta-llama/Llama-3.2-1B-Instruct" --datasets "gsm8k"


# python ../generate_samples.py --prompt_set alpaca_eval --temperatures 0.7 --model_name ${exp_name} --model_path ${model_save_path}/epoch-$n_epochs/

# python ../cleanup.py --model ${model_save_path}/epoch-$n_epochs/
