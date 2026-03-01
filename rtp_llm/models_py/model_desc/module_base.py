import logging
from typing import Any, Optional

from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    DeviceType,
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
    get_device,
)
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

    def warmup(self, max_generate_batch_size: int, max_batch_tokens_size: int) -> None:
        """Warm up optional CUDA kernels (e.g., DeepGEMM) after model initialization.

        This method is called by PyWrappedModel after initialize() succeeds.

        Args:
            max_generate_batch_size: Maximum batch size for generation (decode).
            max_batch_tokens_size: Maximum batch tokens size for prefill.
        """
        try:
            from rtp_llm.models_py.warmup.warmup import kernel_warmup

            kernel_warmup(
                model=self,
                max_generate_batch_size=max_generate_batch_size,
                max_batch_tokens_size=max_batch_tokens_size,
            )
        except Exception as e:
            logging.warning(f"Model warmup failed: {e}", exc_info=True)
            # Warmup failure should never block model initialization

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
