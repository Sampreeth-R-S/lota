import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist

import re
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator, get_dataset
# from filter_dataset_2 import get_batch_iterator, get_dataset
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
import subprocess
from typing import Optional, Dict, List, Union, Tuple

def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False,
             importance_correction: bool = False,
             ipo: bool = False,
             robust_eps: float = 0.0) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    
    
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    if ipo:
        losses = (logits - 1 / (2*beta)) ** 2
    else:
        if robust_eps > 0.0:
            losses = (-F.logsigmoid(beta * logits) * (1. - robust_eps) - F.logsigmoid(-beta * logits) * robust_eps) / beta
        else:
            losses = -F.logsigmoid(beta * logits)
        if importance_correction:
            losses = losses * (policy_rejected_logps - reference_rejected_logps).detach().exp()
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.

           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name in ['sft', 'soft_sft'],
            prefs_path=config.prefs_path,
            num_turns=config.num_turns,
            data_fraction=config.data_fraction,
        )
        self.policy = policy
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.policy.config.pad_token_id = self.tokenizer.pad_token_id
            self.policy.resize_token_embeddings(len(self.tokenizer))
        if not self.config.snip:
            self.adjust_mask(config.mask_path, self.policy)
        else:
            file_path = f"{self.config.save_snip_mask_path}/{self.config.snip_sparsity}_mask.pt"
            if os.path.exists(file_path):
                self.adjust_mask(file_path, self.policy)
            else:
                print("No SNIP mask found. Run find_snip_mask.py")
                self.find_snip_mask(self.policy, file_path)
        self.reference_model = reference_model
        self._cache_dir = get_local_dir(config.local_dirs)

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=self._cache_dir)
        rank0_print(f'Loaded train data iterator')
        if self.config.fast_forward > 0:
            print(f"Fast forwarding {self.config.fast_forward} batches...")
            for _ in range(self.config.fast_forward):
                next(self.train_iterator)
        # self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=self._cache_dir)
        # self.eval_batches = list(self.eval_iterator)
        # rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in ['dpo', 'soft_sft']:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in ['dpo']:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded


    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'
        if loss_config.name == 'dpo':
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            losses, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
                reference_rejected_logps, beta=loss_config.beta, reference_free=loss_config.reference_free,
                importance_correction=loss_config.importance_correction, ipo=loss_config.ipo, robust_eps=loss_config.robust_eps)

            if loss_config.sft_reg > 0.0:
                losses -= loss_config.sft_reg * policy_chosen_logps
 
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/chosen_logprob_ratio'] = (chosen_rewards / loss_config.beta).cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected_logprob_ratio'] = (rejected_rewards / loss_config.beta).cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/beta_normalized_margin'] = ((chosen_rewards - rejected_rewards) / loss_config.beta).cpu().numpy().tolist()

            policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            reference_chosen_logps = all_gather_if_needed(reference_chosen_logps.detach(), self.rank, self.world_size)
            reference_rejected_logps = all_gather_if_needed(reference_rejected_logps.detach(), self.rank, self.world_size)

            metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
            metrics[f'logps_{train_test}/reference_chosen'] = reference_chosen_logps.cpu().numpy().tolist()
            metrics[f'logps_{train_test}/reference_rejected'] = reference_rejected_logps.cpu().numpy().tolist()
            metrics[f'importance_weights_{train_test}/rejected'] = torch.exp(policy_rejected_logps.detach() - reference_rejected_logps.detach()).cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            policy_chosen_ppl = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=True)
            policy_chosen_ppl = all_gather_if_needed(policy_chosen_ppl.detach(), self.rank, self.world_size)
            metrics[f'ppl_{train_test}/chosen'] = policy_chosen_ppl.cpu().numpy().tolist()

            losses = -policy_chosen_logps
            if train:
                losses.mean().backward()

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def adjust_mask(self, mask_path, policy):
        # Skip logic if mask_path ends with "0.0_mask.pt"
        if mask_path.endswith("0.0_mask.pt") or mask_path.endswith("0_mask.pt"):
            # print("Skipping adjust_mask due to mask_path ending with '0.0_mask.pt'")
            self.mask = defaultdict(lambda: None)
            return
        self.mask = self.load_mask(mask_path)
        for key in self.mask.keys():
            print(key)
        adjusted_mask = {}
        
        for name, param in policy.named_parameters():
            adjusted_name = name.replace("._checkpoint_wrapped_module", "")
            print(adjusted_name)
            if adjusted_name in self.mask:
                adjusted_mask[param] = self.mask[adjusted_name]
        self.mask = adjusted_mask

    def load_mask(self, file_path):
        """
        Load the mask from a file.

        Parameters:
        - file_path (str): The file path from where to load the mask.

        Returns:
        - dict: The loaded mask.
        """
        mask = torch.load(file_path, map_location='cpu')
        print(f"Mask loaded from {file_path}")
        return mask

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        from apply_mask_in_backward import _apply_masked_optimizer_in_backward
        optimizer_kwargs = {
            'lr': self.config.lr,
            'lr_scheduler_cls': torch.optim.lr_scheduler.LambdaLR,
            'lr_scheduler_kwargs': {'lr_lambda': lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1))},
            'max_grad_norm': self.config.max_grad_norm,
            'grad_norm_strategy': self.config.grad_norm_strategy,
        }
        # _apply_masked_optimizer_in_backward(getattr(torch.optim, self.config.optimizer), self.policy.parameters(), self.mask, optimizer_kwargs)
        optimizer_class = getattr(torch.optim, self.config.optimizer)
        optimizer = optimizer_class(self.policy.parameters(), lr = self.config.lr)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name == 'dpo':
            self.reference_model.eval()

        self.example_counter = self.config.fast_forward * self.config.batch_size
        self.batch_counter = self.config.fast_forward
        last_log = None
        if type(self.config.save_every) == str and self.config.save_every.startswith('epoch'):
            epoch_freq = float(self.config.save_every.split('_')[1])
            # compute the number of examples per epoch: TODO(works for single dataset only)
            total_num_data = 0
            for dataset in self.config.datasets:
                all_data = get_dataset(dataset, cache_dir=self._cache_dir, split='train', prefs_path=self.config.prefs_path, num_turns=self.config.num_turns, data_fraction=self.config.data_fraction)
                total_num_data += len(all_data)
            n_examples_per_epoch = (total_num_data // self.config.batch_size) * self.config.batch_size
            next_save = n_examples_per_epoch * epoch_freq
        else:
            next_save = self.config.save_every
        next_save += self.example_counter
        print(f'Saving every {next_save} examples')
        lambda_val = 0
        steps = 0
        for i, batch in enumerate(self.train_iterator):
            #### BEGIN EVALUATION ####
            mean_eval_metrics = {}
            if (self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval)): # or (self.example_counter + self.config.batch_size >= next_save):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name == 'dpo':
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                # for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                #     local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                #     with torch.no_grad():
                #         _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                #     for k, v in eval_metrics.items():
                #         all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name == 'dpo':
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len (v) for k, v in all_eval_metrics.items()}
                for k, v in mean_eval_metrics.items():
                    if 'ppl' in k:
                        mean_eval_metrics[k] = np.exp(-v)

                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name == 'dpo':
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name == 'dpo':
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                optimizer.zero_grad()
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                # (loss / self.config.gradient_accumulation_steps).backward()
                loss = 0
                for name, module in self.policy.named_modules():
                  # Check if the module is an instance of vera.Linear and has the calculate_reg_loss method
                  if hasattr(module, 'calculate_reg_loss'):
                    loss += module.calculate_reg_loss(task=1)
                loss = loss*lambda_val
                loss.backward()
                if(steps%1500==0):
                    lambda_val+=0.1
                steps+=1
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                optimizer.step()
                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = 0

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                for k, v in mean_train_metrics.items():
                    if 'ppl' in k:
                        mean_train_metrics[k] = np.exp(-v)
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####

            #### BEGIN SAVING ####
            if self.example_counter >= next_save:
                if self.config.debug:
                    rank0_print('skipping save in debug mode')
                else:
                    if type(self.config.save_every) == str and self.config.save_every.startswith('epoch'):
                        output_dir = os.path.join(self.run_dir, f'epoch-{self.example_counter // n_examples_per_epoch}')
                        next_save += n_examples_per_epoch * epoch_freq
                        os.makedirs(output_dir, exist_ok=True)
                        # the predictions are from a model that are 1 step old
                        # with open(os.path.join(output_dir, 'test_predictions.txt'), 'w') as f:
                            # for acc, loss in zip(all_eval_metrics['rewards_eval/accuracies'], all_eval_metrics['loss/eval']):
                                # f.write(f'{acc},{loss}\n')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        next_save += self.config.save_every
                    rank0_print(f'creating checkpoint to write to {output_dir}...')
                    self.save(output_dir, mean_eval_metrics, run_alpaca_eval=self.config.trigger_alpaca_eval)
            #### END SAVING ####

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()


    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def alpaca_eval(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, f'LATEST')

        if self.rank == 0:
            rank0_print('triggering alpaca evaluation...')
            proc = subprocess.Popen(['/bin/bash',
                                     'dpo-rlaif/eval_ckpt.sh', str(self.config.eval_gpu),
                                     f'{output_dir}', f'{self.config.exp_name}-step{self.example_counter}'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     close_fds=True)
            print(f'started alpaca evaluation for step-{self.example_counter} in process {proc.pid}')

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None, run_alpaca_eval: bool = False):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
