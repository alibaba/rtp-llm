import logging
from typing import Callable, Dict, List, Optional, Union

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
)
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
    parallelism_config: Optional[ParallelismConfig] = None,
) -> MlaImplBase:

    # MTP target-verify arrives with is_prefill=True (sequence_lengths is empty in
    # MtpBatchStreamProcessor::prepareOneStepSpecDecodeModelInput) but it is really
    # multi-token decode with prefix in cache — sglang/vllm both classify it as
    # decode. If we let it go through the prefill path the fast-path branch below
    # skips SparseMlaImpl, so the main (DSA) model uses dense MLA during verify and
    # baseline (DSA decode) uses sparse MLA — different attention algorithms over
    # the same KV cache → divergent main-model predictions and wrong response.
    is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))

    mla_impls = (
        PREFILL_MLA_IMPS
        if (attn_inputs.is_prefill and not is_target_verify)
        else DECODE_MLA_IMPS
    )
    allow_cuda_graph_prefill_absorb = False
    if is_cuda_graph and attn_inputs.is_prefill and not is_target_verify:
        try:
            input_lengths = attn_inputs.input_lengths
            prefix_lengths = attn_inputs.prefix_lengths
            total_q = int(input_lengths.sum().item())
            max_q_per_req = int(input_lengths.max().item())
            has_reuse_cache = (
                prefix_lengths is not None
                and prefix_lengths.numel() > 0
                and int(prefix_lengths.max().item()) > 0
            )
            allow_cuda_graph_prefill_absorb = (
                has_reuse_cache
                and total_q > 0
                and max_q_per_req > 0
                and total_q <= max_q_per_req
            )
        except Exception:
            allow_cuda_graph_prefill_absorb = False
    for impl in mla_impls:
        impl_name = impl.__name__
        if not impl.support(attn_configs, attn_inputs):
            continue

        cos_sin_cache = weight.get_global_weight(W.rope_cos_sin_cache)
        use_fast_path = (
            attn_inputs.is_prefill
            and not is_target_verify
            and attn_inputs.cu_kv_seqlens.max().item() <= attn_configs.indexer_topk
            and not (
                parallelism_config and parallelism_config.prefill_cp_config.is_enabled()
            )
        )

        if not use_fast_path and not impl.support_parallelism_config(
            parallelism_config
        ):
            continue

        if use_fast_path and impl.is_sparse():
            continue

        if (
            attn_configs.is_sparse
            and not use_fast_path
            and not impl.is_sparse()
            and not allow_cuda_graph_prefill_absorb
        ):
            continue

        try:
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
        except Exception as e:
            logging.warning(f"MLA skip {impl_name}: {e}")
            continue
        if not is_cuda_graph or instance.support_cuda_graph():
            return instance
    logging.error(
        f"can not find mla type: is_prefill={attn_inputs.is_prefill}, "
        f"is_target_verify={is_target_verify}, is_sparse={attn_configs.is_sparse}, "
        f"use_mla={attn_configs.use_mla}, kv_cache_dtype={attn_configs.kv_cache_dtype}, "
        f"is_cuda_graph={is_cuda_graph}, indexer_topk={attn_configs.indexer_topk}, "
        f"impls={[i.__name__ for i in mla_impls]}, "
        f"all_prefill={[i.__name__ for i in PREFILL_MLA_IMPS]}, "
        f"all_decode={[i.__name__ for i in DECODE_MLA_IMPS]}"
    )
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
    # Aiter ASM / Paged prefill
    elif (
        "AiterPrefillImplAsm" in impl_class_name
        or "AiterDecodeImplAsm" in impl_class_name
        or "AiterPrefillImplPaged" in impl_class_name
    ):
        return not fmha_config.use_asm_pa
    # Aiter Non-ASM implementations
    elif (
        "AiterPrefillImplNonAsm" in impl_class_name
        or "AiterDecodeImplNonAsm" in impl_class_name
    ):
        return not fmha_config.use_aiter_pa
    # Aiter Triton implementations
    elif "AiterDecodeImplTriton" in impl_class_name:
        return not fmha_config.use_triton_pa
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
    parallelism_config: Optional[ParallelismConfig] = None,
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

        # Check parallelism config first to avoid calling support() on impls
        # that don't support CP — some impls (e.g. TRT) abort in support().
        # CP only splits the prefill sequence; decode runs standard attention,
        # so the prefill-CP gate must not reject decode impls when CP is enabled.
        if attn_inputs.is_prefill and not impl.support_parallelism_config(
            parallelism_config
        ):
            continue

        # Check support before creating instance
        if not impl.support(attn_configs, attn_inputs):
            continue
        try:
            instance = impl(attn_configs, attn_inputs, parallelism_config)
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
