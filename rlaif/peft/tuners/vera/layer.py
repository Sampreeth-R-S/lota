# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
# from galore_torch.galore_projector import GaLoreProjector
# from galore_torch.galore_projector_tensor import GaLoreProjectorTensor
# galore_rank = 256
# galore_update_projection_gap = 500
# galore_scale = 4
# galore_projection_type = "std"
sparsity_ratio = 0.99
use_global_mask = True
global_maskA = None
global_maskB = None
from .._buffer_dict import BufferDict


class VeraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("adapter_weights",)
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.vera_dropout = nn.ModuleDict({})
        self.adapter_weights = nn.ParameterDict({})
        self.indices = {}
        global global_maskA, global_maskB
        if global_maskA is None:
            global_maskA = torch.load("/root/lota/rlaif/peft/tuners/vera/mask_tensor_A_10percent.pt").float()
            global_maskB = torch.load("/root/lota/rlaif/peft/tuners/vera/mask_tensor_B_10percent.pt").float()
        self.mask = None
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        vera_A: BufferDict,
        vera_B: BufferDict,
        r,
        vera_dropout,
        init_weights,
        d_initial: float = 0.1,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if vera_dropout > 0.0:
            vera_dropout_layer = nn.Dropout(p=vera_dropout)
        else:
            vera_dropout_layer = nn.Identity()

        self.vera_dropout.update(nn.ModuleDict({adapter_name: vera_dropout_layer}))
        self.indices[adapter_name] = []
        if self.out_features == 14336 and self.in_features == 4096:
            indices_tensor = torch.nonzero(global_maskA[:14336, :4096], as_tuple=False)
        elif self.out_features == 4096 and self.in_features == 14336:
            indices_tensor = torch.nonzero(global_maskA[:14336, :4096].T, as_tuple=False)
        else:
            indices_tensor = torch.nonzero(global_maskA[:self.out_features, :self.in_features], as_tuple=False)

        self.indices[adapter_name] = indices_tensor  # Store as tensor
        self.indices[adapter_name].requires_grad=False
        self.adapter_weights[adapter_name] = nn.Parameter(torch.zeros(indices_tensor.shape[0]), requires_grad=True)
        
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)
        

    def reset_vera_parameters(self, adapter_name, d_initial: float = 0.1):
        if adapter_name in self.vera_lambda_d_main.keys():
            with torch.no_grad():
                nn.init.constant_(self.vera_lambda_d_main[adapter_name], d_initial)
                nn.init.constant_(self.vera_lambda_b_main[adapter_name], 1.0)
                nn.init.constant_(self.vera_lambda_b_upper[adapter_name], 0.1)
                nn.init.constant_(self.vera_lambda_b_lower[adapter_name], 0.1)
                nn.init.constant_(self.vera_lambda_d_upper[adapter_name], 0.1)
                nn.init.constant_(self.vera_lambda_d_lower[adapter_name], 0.1)


class Linear(nn.Linear, VeraLayer):
    def __init__(
        self,
        base_layer,
        vera_A: BufferDict,
        vera_B: BufferDict,
        adapter_name: str,
        r: int = 0,
        vera_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        VeraLayer.__init__(self, base_layer, **kwargs)
        self.mask = (torch.rand((self.out_features,self.in_features)) < sparsity_ratio).float()
        num_ones = torch.sum(self.mask == 1).item()
        total_elements = self.mask.numel()
        percentage_ones = (num_ones / total_elements) * 100
        print(f"Percentage of 1's in the mask: {percentage_ones:.2f}%")
        self.steps = 0
        self.fan_in_fan_out = fan_in_fan_out
        self.task = 1
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, vera_A, vera_B, r, vera_dropout, init_weights, d_initial=d_initial)
        print(self.adapter_weights[adapter_name].requires_grad)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        
        

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.vera_lambda_d.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vera_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return self.adapter_weights[adapter]
    
    def calculate_reg_loss(self,task,*args,**kwargs):
      l2_norm = 0
      global global_maskA
      global global_maskB
      for active_adapter in self.active_adapters:
        if active_adapter not in self.adapter_weights.keys():
          continue
        if(use_global_mask):
            if(global_maskA is None):
                global_maskA = torch.load("/root/lota/rlaif/peft/tuners/vera/mask_tensor_A_10percent.pt").float()
                global_maskB = torch.load("/root/lota/rlaif/peft/tuners/vera/mask_tensor_B_10percent.pt").float()
            if(task == 1):
                masked_weights = self.adapter_weights[active_adapter] * global_maskA.to(self.adapter_weights[active_adapter].device)
            else:
                masked_weights = self.adapter_weights[active_adapter] * global_maskB.to(self.adapter_weights[active_adapter].device) * global_maskA.to(self.adapter_weights[active_adapter].device)
        else:
            masked_weights = self.adapter_weights[active_adapter] * self.mask.to(self.adapter_weights[active_adapter].device)
        l2_norm += torch.sum(masked_weights ** 2)
      return l2_norm
    
    def switch_task(self):
      self.task = (self.task + 1) % 2
      print("switched task")
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        device = x.device  # Get the device from input tensor
        
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            
            for active_adapter in self.active_adapters:
                if active_adapter not in self.adapter_weights.keys():
                    continue
                    
                temp_weight = torch.zeros((self.out_features, self.in_features), device=x.device)
                indices = self.indices[active_adapter].to(x.device)
                temp_weight[indices[:, 0], indices[:, 1]] = self.adapter_weights[active_adapter]
                
                if self.task == 1:
                    result += F.linear(x, temp_weight.to(x.dtype), bias=None)
                else:
                    # Ensure global mask is on the correct device
                    global_maskA_device = global_maskA.to(device)
                    result += F.linear(
                        x,
                        (adapter_weights * global_maskA_device).to(x.dtype),
                        bias=None
                    )
        
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "vera." + rep

