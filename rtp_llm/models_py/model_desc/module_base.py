import logging
from typing import Any, Optional

from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules import DECODE_MHA_IMPS, PREFILL_MHA_IMPS, FMHAImplBase
from rtp_llm.models_py.modules.fmha import DECODE_MLA_IMPS, PREFILL_MLA_IMPS
from rtp_llm.ops import (
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.ops.compute_ops import DeviceType, KVCache, get_device
from rtp_llm.utils.model_weight import W


class GptModelBase(nn.Module):
    def __init__(
        self, 
        config: ModelConfig, 
        parallelism_config,
        weight: ModelWeights,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config

        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device().get_device_type()

        ## (batch_size -> fmha_params)
        self.params_dict: dict[int, Any] = {}

        logging.info(
            f"GptModelBase initialized with layer_num={self.layer_num}, "
            f"vocab_size={self.vocab_size}, device_type={self.device_type}"
        )

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            logging.info(
                f"GptModelBase initialized with "
                f"k_cache_base={self.kv_cache.k_cache_base.shape if self.kv_cache.k_cache_base is not None else None}, "
                f"v_cache_base={self.kv_cache.v_cache_base.shape if self.kv_cache.v_cache_base is not None else None}"
                f"k_scale_base={self.kv_cache.k_scale_base.shape if self.kv_cache.k_scale_base is not None else None}"
                f"v_scale_base={self.kv_cache.v_scale_base.shape if self.kv_cache.v_scale_base is not None else None}"
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

    def get_mla_impl(self, attn_inputs: PyAttentionInputs) -> FMHAImplBase:
        mha_impls = PREFILL_MLA_IMPS if attn_inputs.is_prefill else DECODE_MLA_IMPS
        for fmha_impl in mha_impls:
            cos_sin_cache = self.weight.get_global_weight(W.rope_cos_sin_cache)
            impl = fmha_impl(
                self.config,
                self.parallelism_config,
                attn_inputs,
                self.weight.weights,
                cos_sin_cache,
            )
            if impl.support():
                return impl
        raise Exception(f"can not find fmha type")

    def get_fmha_impl(self, attn_inputs: PyAttentionInputs) -> FMHAImplBase:
        from rtp_llm.ops import FMHAType
        mha_impls = PREFILL_MHA_IMPS if attn_inputs.is_prefill else DECODE_MHA_IMPS
        for fmha_impl in mha_impls:
            # Check config at runtime to filter implementations
            if self.fmha_config is not None:
                fmha_type = fmha_impl.fmha_type()
                # Skip FlashInfer if disabled
                if fmha_type == FMHAType.FLASH_INFER and self.fmha_config.disable_flash_infer:
                    continue
                # Skip TRT FMHA if not enabled
                if fmha_type == FMHAType.TRT_V2 and not self.fmha_config.enable_paged_trt_fmha:
                    continue
                # Skip XQA if not enabled
                if fmha_type == FMHAType.XQA and not self.fmha_config.enable_xqa:
                    continue
            
            # Pass config objects to implementation if it accepts them
            # Some implementations may not accept these parameters yet
            try:
                # Try with config parameters first
                impl = fmha_impl(
                    self.config, 
                    self.parallelism_config, 
                    attn_inputs,
                    fmha_config=self.fmha_config,
                    py_hw_kernel_config=self.py_hw_kernel_config,
                )
            except TypeError:
                # Fallback to old signature if implementation doesn't accept config parameters yet
                impl = fmha_impl(self.config, self.parallelism_config, attn_inputs)
            if impl.support():
                return impl
        raise Exception(f"can not find fmha type")
