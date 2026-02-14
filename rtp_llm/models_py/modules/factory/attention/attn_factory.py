import logging
from typing import Callable, Dict, List, Optional, Union

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W

# Lists to store registered implementations
PREFILL_MHA_IMPS: List[type[FMHAImplBase]] = []
DECODE_MHA_IMPS: List[type[FMHAImplBase]] = []
PREFILL_MLA_IMPS: List[type[MlaImplBase]] = []
DECODE_MLA_IMPS: List[type[MlaImplBase]] = []


def get_mla_impl(
    attn_configs: AttentionConfigs,
    weight: ModelWeights,
    attn_inputs: PyAttentionInputs,
    fmha_config: Optional[FMHAConfig] = None,
    quant_config: Optional[object] = None,
    is_cuda_graph: bool = False,
    max_seq_len: int = 0,
) -> MlaImplBase:

    mla_impls = PREFILL_MLA_IMPS if attn_inputs.is_prefill else DECODE_MLA_IMPS
    for impl in mla_impls:
        # Check support before creating instance
        if not impl.support(attn_configs, attn_inputs):
            continue

        cos_sin_cache = weight.get_global_weight(W.rope_cos_sin_cache)
        use_fast_path = (
            attn_inputs.is_prefill
            and attn_inputs.cu_kv_seqlens.max().item() <= attn_configs.indexer_topk
            and False
        )
        # Skip sparse MLA if fast path is enabled
        if (use_fast_path and impl.is_sparse()) or (
            not use_fast_path and not impl.is_sparse()
        ):
            logging.debug(
                f"skip sparse mla impl [{impl}] because fast path: {use_fast_path}"
            )
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
        )
        if not is_cuda_graph or instance.support_cuda_graph():
            return instance
    raise Exception(f"can not find mla type")


def _is_fmha_impl_disabled(
    impl_class_name: str, fmha_config: Optional[FMHAConfig]
) -> bool:
    """Check if a FMHA implementation is disabled in fmha_config.

    Args:
        impl_class_name: The implementation class name
        fmha_config: The FMHA config, if None, assume not disabled

    Returns:
        True if the FMHA implementation is disabled, False otherwise
    """
    if fmha_config is None:
        return False

    # XQA implementations
    if "XQA" in impl_class_name:
        return not fmha_config.enable_xqa
    # TRT implementations
    elif impl_class_name == "TRTMHAImpl":
        return not fmha_config.enable_trt_fmha
    elif impl_class_name == "TRTPagedMHAImpl":
        return not fmha_config.enable_paged_trt_fmha
    # FlashInfer TRTLLM implementations
    elif "FlashInferTRTLLM" in impl_class_name:
        return fmha_config.disable_flash_infer
    # FlashInfer implementations
    elif "FlashInfer" in impl_class_name or "Flashinfer" in impl_class_name:
        return fmha_config.disable_flash_infer
    # Aiter ASM implementations
    elif (
        "AiterPrefillImplAsm" in impl_class_name
        or "AiterDecodeImplAsm" in impl_class_name
    ):
        return not fmha_config.use_asm_pa
    # Aiter Non-ASM implementations
    elif (
        "AiterPrefillImplNonAsm" in impl_class_name
        or "AiterDecodeImplNonAsm" in impl_class_name
    ):
        return not fmha_config.use_aiter_pa
    # Default: not disabled
    return False


def get_fmha_impl(
    attn_configs: AttentionConfigs,
    weight: ModelWeights,
    attn_inputs: PyAttentionInputs,
    fmha_config: Optional[FMHAConfig] = None,
    quant_config: Optional[object] = None,
    is_cuda_graph: bool = False,
    max_seq_len: int = 0,
) -> FMHAImplBase:
    # Set is_cuda_graph as dynamic attribute on attn_inputs for base class to read
    attn_inputs.is_cuda_graph = is_cuda_graph

    mha_impls = PREFILL_MHA_IMPS if attn_inputs.is_prefill else DECODE_MHA_IMPS
    for impl in mha_impls:
        # Check if this FMHA implementation is disabled before creating instance
        impl_class_name = impl.__name__

        # Skip if this FMHA implementation is disabled in config
        if _is_fmha_impl_disabled(impl_class_name, fmha_config):
            continue

        # Check support before creating instance
        if not impl.support(attn_configs, attn_inputs):
            continue

        try:
            instance = impl(attn_configs, attn_inputs)
            if not is_cuda_graph or instance.support_cuda_graph():
                return instance
        except Exception as e:
            # If instantiation fails, continue to next impl
            logging.warning(f"Failed to instantiate {impl_class_name}: {e}")
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
        attn_configs = model_config.getAttentionConfigs(parallelism_config.tp_size)
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
    def register_fmha_impl(cls, attention_type: str, impl_method: str):
        """
        Register a new FMHA implementation method for an attention type.

        Args:
            attention_type: String key for the attention type
            impl_method: Method name to call for getting FMHA implementation
        """
        cls.FMHA_IMPL_REGISTRY[attention_type] = impl_method

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported attention types."""
        return list(cls.FMHA_IMPL_REGISTRY.keys())
