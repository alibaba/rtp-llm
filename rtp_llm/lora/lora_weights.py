import logging
import os
from typing import Any, Dict, List, NamedTuple, Optional

import torch
import torch.serialization

from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.database import BaseDatabase


class LoRAWeights:

    def __init__(self, num_layers: int) -> None:
        self.lora_a_weights: List[Dict[str, torch.Tensor]] = []
        self.lora_b_weights: List[Dict[str, torch.Tensor]] = []
        self.lora_rank = 0
        for _ in range(num_layers):
            self.lora_a_weights.append({})
            self.lora_b_weights.append({})

    def set_lora_rank(self, lora_rank: int):
        self.lora_rank = lora_rank

    def set_layer_weight(
        self, int8_flag: bool, layer_id: int, name: str, tensor: torch.Tensor
    ):
        assert not int8_flag, "LoRA does not support int8 mode"
        prefix_name = name[: -len(".lora_A")]
        if name.endswith(".lora_A"):
            self.lora_a_weights[layer_id][prefix_name] = tensor
        elif name.endswith(".lora_B"):
            self.lora_b_weights[layer_id][prefix_name] = tensor
        else:
            raise ValueError(f"Invalid lora weight name: {name}")

    def apply_scale(self, scale: float):
        logging.info(f"scale size {scale}")
        for i, layer_weights in enumerate(self.lora_b_weights):
            for name, weight in layer_weights.items():
                self.lora_b_weights[i][name] = weight * scale
