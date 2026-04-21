import logging
from typing import Callable, Dict, List, Optional, Union

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W


def get_mla_impl(
    attn_configs: AttentionConfigs,
    weight: ModelWeights,
    attn_inputs: PyAttentionInputs,
    fmha_config: Optional[FMHAConfig] = None,
    quant_config: Optional[object] = None,
    is_cuda_graph: bool = False,
    max_seq_len: int = 0,
    parallelism_config: Optional[ParallelismConfig] = None,
) -> MlaImplBase:
    from rtp_llm.device import get_current_device

    device = get_current_device()
    priorities = (
        device.get_prefill_mla_priorities()
        if attn_inputs.is_prefill
        else device.get_decode_mla_priorities()
    )

    for impl in priorities:
        # Check support before creating instance
        if not impl.support(attn_configs, attn_inputs):
            continue

        cos_sin_cache = weight.get_global_weight(W.rope_cos_sin_cache)
        # TODO: support fast path for cp prefill
        use_fast_path = (
            attn_inputs.is_prefill
            and attn_inputs.cu_kv_seqlens.max().item() <= attn_configs.indexer_topk
            and not (
                parallelism_config and parallelism_config.prefill_cp_config.is_enabled()
            )
        )

        if not use_fast_path and not impl.support_parallelism_config(
            parallelism_config
        ):
            continue

        # Skip sparse MLA if fast path is enabled
        if use_fast_path and impl.is_sparse():
            logging.debug(
                f"skip sparse mla impl [{impl}] because fast path: {use_fast_path}"
            )
            continue

        if attn_configs.is_sparse and not use_fast_path and not impl.is_sparse():
            logging.debug(f"skip mla impl [{impl}] because sparse mla is not supported")
            continue

        instance = impl(
            attn_configs,
            attn_inputs,
            weight.weights,
            cos_sin_cache=cos_sin_cache,
            fmha_config=fmha_config,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )
        if not is_cuda_graph or instance.support_cuda_graph():
            return instance
    raise Exception(f"can not find mla type")


def get_fmha_impl(
    attn_configs: AttentionConfigs,
    weight: ModelWeights,
    attn_inputs: PyAttentionInputs,
    fmha_config: Optional[FMHAConfig] = None,
    quant_config: Optional[object] = None,
    is_cuda_graph: bool = False,
    max_seq_len: int = 0,
    parallelism_config: Optional[ParallelismConfig] = None,
) -> FMHAImplBase:
    from rtp_llm.device import get_current_device

    # Set is_cuda_graph as dynamic attribute on attn_inputs for base class to read
    attn_inputs.is_cuda_graph = is_cuda_graph

    device = get_current_device()
    priorities = (
        device.get_prefill_mha_priorities()
        if attn_inputs.is_prefill
        else device.get_decode_mha_priorities()
    )

    for impl_cls in priorities:
        # Skip if this implementation is disabled by user config
        if not impl_cls.is_available(fmha_config):
            continue

        # Check support before creating instance
        if not impl_cls.support(attn_configs, attn_inputs):
            continue

        # Check if implementation supports parallelism config
        if not impl_cls.support_parallelism_config(parallelism_config):
            continue
        try:
            instance = impl_cls(attn_configs, attn_inputs, parallelism_config)
            if not is_cuda_graph or instance.support_cuda_graph():
                return instance

        except Exception as e:
            # If instantiation fails, continue to next impl
            logging.warning(f"Failed to instantiate {impl_cls.__name__}: {e}")
            continue
    raise Exception(f"can not find mha type")


class AttnImplFactory(object):
    """Factory class for creating FMHA implementations based on attention_type."""

    # FMHA implementation registry - maps attention_type to impl method
    FMHA_IMPL_REGISTRY: Dict[
        str,
        Callable[
            [AttentionConfigs, ModelWeights, PyAttentionInputs, Optional[FMHAConfig]],
            Union[FMHAImplBase, MlaImplBase],
        ],
    ] = {
        "mha": get_fmha_impl,
        "mla": get_mla_impl,
    }

    @classmethod
    def get_fmha_impl(
        cls,
        model_config,  # ModelConfig - kept for backward compatibility, but will extract attn_configs
        parallelism_config,
        weight: ModelWeights,
        attn_inputs: PyAttentionInputs,
        fmha_config: Optional[FMHAConfig] = None,
        is_cuda_graph: bool = False,
    ) -> FMHAImplBase:
        # Extract AttentionConfigs from ModelConfig
        attn_configs = model_config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        attn_inputs.headwise_config = getattr(model_config, "headwise_config", None)
        key_str = "mla" if attn_configs.use_mla else "mha"
        fmha_impl_method = cls.FMHA_IMPL_REGISTRY[key_str]
        instance = fmha_impl_method(
            attn_configs,
            weight,
            attn_inputs,
            fmha_config,
            model_config.quant_config,
            is_cuda_graph,
            model_config.max_seq_len,
            parallelism_config,
        )
        logging.debug(f"get fmha impl: {type(instance).__name__}")
        return instance

    @classmethod
    def get_fmha_impl_method(cls, attention_type: str) -> str:
        """
        Get the appropriate FMHA implementation method based on attention_type.

        Args:
            attention_type: String identifying the attention type

        Returns:
            Method name to call for getting FMHA implementation

        Raises:
            ValueError: If attention_type is not supported
        """
        if attention_type not in cls.FMHA_IMPL_REGISTRY:
            available_types = list(cls.FMHA_IMPL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported attention type '{attention_type}'. Available types: {available_types}"
            )

        return cls.FMHA_IMPL_REGISTRY[attention_type]

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported attention types."""
        return list(cls.FMHA_IMPL_REGISTRY.keys())
