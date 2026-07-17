import logging
from typing import TYPE_CHECKING, Any, Optional

from torch import Tensor

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)

if TYPE_CHECKING:
    from rtp_llm.model_loader.model_weight_info import ModelWeights


def required_config_value(config: Any, *names: str) -> Any:
    for name in names:
        value = (
            config.get(name)
            if isinstance(config, dict)
            else getattr(config, name, None)
        )
        if value is not None:
            return value
    raise ValueError(f"Model config requires one of {names}")


class GptModelBase(RtpModule):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: Optional["ModelWeights"] = None,
        max_generate_batch_size: int = 0,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num = required_config_value(
            config, "num_layers", "num_hidden_layers"
        )
        self.vocab_size = required_config_value(config, "vocab_size")

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()

        ## (batch_size -> fmha_params)
        self.params_dict: dict[int, Any] = {}

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = len(self.kv_cache.kv_cache_base_by_layer)
            layer0_shape = (
                self.kv_cache.kv_cache_base_by_layer[0].shape
                if num_layers > 0
                and self.kv_cache.kv_cache_base_by_layer[0] is not None
                else None
            )
            num_scale_layers = len(self.kv_cache.kv_scale_base_by_layer)
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shape={layer0_shape}, "
                f"num_scale_layers={num_scale_layers}, "
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
    ) -> None:
        if capture_batch_size not in self.params_dict:
            raise ValueError(f"No captured FMHA params for batch {capture_batch_size}")
        params = self.params_dict[capture_batch_size]
        if params is None:
            raise RuntimeError("Captured FMHA params cannot be None")
        params.fillParams(
            sequence_lengths,
            input_lengths,
            kv_cache_block_id_host,
            replay_batch_size,
            seq_size_per_block,
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )
        return fmha_impl

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
