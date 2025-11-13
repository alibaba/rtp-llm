import logging
from typing import Any, Optional

from torch import Tensor, nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules import DECODE_MHA_IMPS, PREFILL_MHA_IMPS, FMHAImplBase
from rtp_llm.models_py.modules.fmha import DECODE_MLA_IMPS, PREFILL_MLA_IMPS
from rtp_llm.ops.compute_ops import (
    DeviceType,
    KVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
    get_device,
)
from rtp_llm.utils.model_weight import W


class GptModelBase(nn.Module):
    def __init__(self, config: GptInitModelParameters, weight: ModelWeights) -> None:
        super().__init__()
        self.config = config
        self.weight = weight

        self.layer_num: int = config.layer_num
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device().get_device_type()

        self.micro_batch_size: int = (
            1 if config.device_resource_config.enable_layer_micro_batch == 0 else 2
        )
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
                attn_inputs,
                self.weight.weights,
                cos_sin_cache,
            )
            if impl.support():
                return impl
        raise Exception(f"can not find fmha type")

    def get_fmha_impl(self, attn_inputs: PyAttentionInputs) -> FMHAImplBase:
        mha_impls = PREFILL_MHA_IMPS if attn_inputs.is_prefill else DECODE_MHA_IMPS
        for fmha_impl in mha_impls:
            impl = fmha_impl(self.config, attn_inputs)
            if impl.support():
                return impl
        raise Exception(f"can not find fmha type")
