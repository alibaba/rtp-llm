import logging
from typing import Callable, Dict, List, Optional, Union

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, KvCacheDataType
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
    # Set is_cuda_graph as dynamic attribute on attn_inputs for base class to read
    attn_inputs.is_cuda_graph = is_cuda_graph

    mla_impls = PREFILL_MLA_IMPS if attn_inputs.is_prefill else DECODE_MLA_IMPS
    for impl in mla_impls:
        cos_sin_cache = weight.get_global_weight(W.rope_cos_sin_cache)
        has_reuse_cache = False
        if (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
        ):
            has_reuse_cache = attn_inputs.prefix_lengths.max().item() > 0
        use_fp8_reuse_cache = (
            attn_configs.kv_cache_dtype == KvCacheDataType.FP8 and has_reuse_cache
        )
        use_fast_path = (
            attn_inputs.is_prefill
            and attn_inputs.cu_kv_seqlens.max().item() <= attn_configs.indexer_topk
            and not use_fp8_reuse_cache
        )
        # Skip sparse MLA if fast path is enabled
        if use_fast_path and impl.is_sparse():
            logging.debug(f"skip sparse mla impl [{impl}] because fast path is enabled")
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
        if instance.support() and (not is_cuda_graph or instance.support_cuda_graph()):
            return instance
    raise Exception(f"can not find mla type")


def _is_fmha_type_disabled(
    fmha_type: FMHAType, fmha_config: Optional[FMHAConfig]
) -> bool:
    """Check if a FMHA type is disabled in fmha_config.

    Args:
        fmha_type: The FMHA type to check
        fmha_config: The FMHA config, if None, assume not disabled
        impl_class_name: The implementation class name, used to distinguish between ASM and NonAsm variants

    Returns:
        True if the FMHA type is disabled, False otherwise
    """
    if fmha_config is None:
        return False

    if fmha_type == FMHAType.XQA:
        return not fmha_config.enable_xqa
    elif fmha_type == FMHAType.TRT_V2:
        return not fmha_config.enable_trt_fmha
    elif fmha_type == FMHAType.PAGED_TRT_V2:
        return not fmha_config.enable_paged_trt_fmha
    elif fmha_type == FMHAType.TRT_V1:
        return not fmha_config.enable_trtv1_fmha
    elif fmha_type == FMHAType.OPEN_SOURCE:
        return not fmha_config.enable_open_source_fmha
    elif fmha_type == FMHAType.PAGED_OPEN_SOURCE:
        return not fmha_config.enable_paged_open_source_fmha
    elif fmha_type == FMHAType.FLASH_INFER:
        return fmha_config.disable_flash_infer
    elif (
        fmha_type == FMHAType.AITER_ASM_DECODE
        or fmha_type == FMHAType.AITER_ASM_PREFILL
    ):
        return not fmha_config.use_asm_pa
    elif fmha_type == FMHAType.AITER_DECODE or fmha_type == FMHAType.AITER_PREFILL:
        return not fmha_config.use_aiter_pa
    # FMHAType.NONE
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
        # Check if this FMHA type is disabled before creating instance
        # We need to create a temporary instance to get its fmha_type
        # But we can optimize by checking the type first if possible
        try:
            # Try to get fmha_type without full instantiation if possible
            # For now, we'll create the instance and check both disabled status and support
            instance = impl(attn_configs, attn_inputs)
            fmha_type = instance.fmha_type()

            # Skip if this FMHA type is disabled in config
            if _is_fmha_type_disabled(fmha_type, fmha_config):
                continue

            if instance.support() and (
                not is_cuda_graph or instance.support_cuda_graph()
            ):
                return instance
        except Exception as e:
            # If instantiation fails, continue to next impl
            logging.warning(f"Failed to instantiate {impl}: {e}")
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
        logging.debug(f"get fmha impl: {instance.fmha_type()}")
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
