
import logging

import torch
from torch import nn

from typing import Optional

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights

from rtp_llm.ops import PyModelInputs, PyModelOutputs, PyAttentionInputs, PyModelInitResources
from rtp_llm.ops import get_device, DeviceType, DeviceExporter

class GptModelBase(nn.Module):
    def __init__(self, config: GptInitModelParameters, weight: ModelWeights) -> None:
        super().__init__()
        self.config = config
        self.weight = weight

        self.layer_num: int = config.layer_num
        self.vocab_size: int = config.vocab_size

        self.k_cache_base: Optional[torch.Tensor] = None
        self.v_cache_base: Optional[torch.Tensor] = None

        self.device_type: DeviceType = get_device().get_device_type()

        logging.info(f"GptModelBase initialized with layer_num={self.layer_num}, "
                     f"vocab_size={self.vocab_size}, device_type={self.device_type}")

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.k_cache_base = init_resource.k_cache_base
        self.v_cache_base = init_resource.v_cache_base
        logging.info(f"GptModelBase initialized with "
                     f"k_cache_base={self.k_cache_base.shape if self.k_cache_base is not None else None}, "
                     f"v_cache_base={self.v_cache_base.shape if self.v_cache_base is not None else None}")
        return True

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
