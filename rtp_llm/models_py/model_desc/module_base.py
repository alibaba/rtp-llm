import logging
from typing import Any, Optional

from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.symm_mem import get_symm_mem_communicator
from rtp_llm.ops.compute_ops import (
    DeviceType,
    KVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.ops.compute_ops import DeviceType, KVCache, get_device
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.utils.model_weight import W


class GptModelBase(nn.Module):
    def __init__(
        self, 
        config: ModelConfig, 
        parallelism_config,
        weight: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[DeviceResourceConfig] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1 if device_resource_config and device_resource_config.enable_layer_micro_batch == 0 else 2
        )
        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device().get_device_type()

        ## (batch_size -> fmha_params)
        self.params_dict: dict[int, Any] = {}


    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            logging.info(
                f"GptModelBase initialized with "
                f"kv_cache_base={self.kv_cache.kv_cache_base.shape if self.kv_cache.kv_cache_base is not None else None}, "
                f"kv_scale_base={self.kv_cache.kv_scale_base.shape if self.kv_cache.kv_scale_base is not None else None}, "
            )
        return True

    ## for cuda graph attn kernel params' fill
    def fill_params(
        self,
        sequence_lengths: Tensor,
        input_lengths: Tensor,
        kv_cache_block_id_host: Tensor,
        replay_batch_size: int,
        capture_batch_size: int,
        seq_size_per_block: int,
    ):
        assert capture_batch_size in self.params_dict
        params_ptr = self.params_dict[capture_batch_size]
        assert params_ptr is not None
        params_ptr.fillParams(
            sequence_lengths,
            input_lengths,
            kv_cache_block_id_host,
            replay_batch_size,
            seq_size_per_block,
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
