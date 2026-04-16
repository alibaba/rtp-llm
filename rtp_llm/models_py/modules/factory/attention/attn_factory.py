import logging
from typing import Callable, Dict, List, Optional, Union

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
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

    mla_impls = PREFILL_MLA_IMPS if attn_inputs.is_prefill else DECODE_MLA_IMPS
    for impl in mla_impls:
        # Check support before creating instance
        if not impl.support(attn_configs, attn_inputs):
            continue

        cos_sin_cache = weight.get_global_weight(W.rope_cos_sin_cache)
        use_fast_path = (
            attn_inputs.is_prefill
            and attn_inputs.cu_kv_seqlens.max().item() <= attn_configs.indexer_topk
        )
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
        )
        if not is_cuda_graph or instance.support_cuda_graph():
            return instance
    raise Exception(f"can not find mla type")


def _get_effective_backends(fmha_config: FMHAConfig, is_prefill: bool) -> List[str]:
    """Resolve the effective attn_backend list for the given stage.

    Priority: prefill/decode override > global attn_backend.
    Returns a list of backend NAMEs (possibly ["auto"] or ["none"]).

    Supports comma-separated ordered lists, e.g. "xqa,flashinfer" means
    try xqa first, then flashinfer. Each candidate's support() is still checked.
    """
    raw = ""
    if is_prefill and fmha_config.prefill_attn_backend:
        raw = fmha_config.prefill_attn_backend
    elif not is_prefill and fmha_config.decode_attn_backend:
        raw = fmha_config.decode_attn_backend
    else:
        raw = fmha_config.attn_backend
    return [s.strip() for s in raw.split(",") if s.strip()]


def _get_blocked_backends(fmha_config: FMHAConfig) -> set:
    if not fmha_config.disable_attn_backends:
        return set()
    return {s.strip() for s in fmha_config.disable_attn_backends.split(",") if s.strip()}


def _is_fmha_impl_disabled_legacy(
    impl_class: type, fmha_config: FMHAConfig
) -> bool:
    """Legacy boolean flag check. Only called when effective_backend == "auto"."""
    impl_class_name = impl_class.__name__
    if "XQA" in impl_class_name:
        return not fmha_config.enable_xqa
    elif impl_class_name == "TRTMHAImpl":
        return not fmha_config.enable_trt_fmha
    elif impl_class_name == "TRTPagedMHAImpl":
        return not fmha_config.enable_paged_trt_fmha
    elif "FlashInfer" in impl_class_name or "Flashinfer" in impl_class_name:
        return fmha_config.disable_flash_infer
    elif (
        "AiterPrefillImplAsm" in impl_class_name
        or "AiterDecodeImplAsm" in impl_class_name
        or "AiterPrefillImplPaged" in impl_class_name
    ):
        return not fmha_config.use_asm_pa
    elif (
        "AiterPrefillImplNonAsm" in impl_class_name
        or "AiterDecodeImplNonAsm" in impl_class_name
    ):
        return not fmha_config.use_aiter_pa
    elif "AiterDecodeImplTriton" in impl_class_name:
        return not fmha_config.use_triton_pa
    return False


def _try_instantiate(
    impl: type,
    attn_configs: AttentionConfigs,
    attn_inputs: PyAttentionInputs,
    parallelism_config: Optional[ParallelismConfig],
    is_cuda_graph: bool,
) -> Optional[FMHAImplBase]:
    """Try to create an impl instance, checking support/parallelism/cuda_graph."""
    if not impl.support(attn_configs, attn_inputs):
        return None
    if not impl.support_parallelism_config(parallelism_config):
        return None
    try:
        instance = impl(attn_configs, attn_inputs, parallelism_config)
        if is_cuda_graph and not instance.support_cuda_graph():
            return None
        return instance
    except Exception as e:
        logging.warning(f"Failed to instantiate {impl.__name__}: {e}")
        return None


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
    attn_inputs.is_cuda_graph = is_cuda_graph
    mha_impls = PREFILL_MHA_IMPS if attn_inputs.is_prefill else DECODE_MHA_IMPS

    if fmha_config is None:
        backends = ["auto"]
        blocked = set()
    else:
        backends = _get_effective_backends(fmha_config, attn_inputs.is_prefill)
        blocked = _get_blocked_backends(fmha_config)

    if backends == ["none"]:
        raise Exception("Attention is disabled (attn_backend=none)")

    if backends == ["auto"]:
        # Auto mode: iterate impls in registration order, check legacy flags + blocklist
        for impl in mha_impls:
            name = getattr(impl, "NAME", None)
            if name and name in blocked:
                continue
            if fmha_config and _is_fmha_impl_disabled_legacy(impl, fmha_config):
                continue
            instance = _try_instantiate(impl, attn_configs, attn_inputs, parallelism_config, is_cuda_graph)
            if instance is not None:
                return instance
    else:
        # Explicit backend list: iterate in user-specified order.
        # For each backend name, find all matching impls and try them.
        name_to_impls = {}
        for impl in mha_impls:
            name = getattr(impl, "NAME", None)
            if name:
                name_to_impls.setdefault(name, []).append(impl)

        for backend_name in backends:
            if backend_name in blocked:
                continue
            for impl in name_to_impls.get(backend_name, []):
                instance = _try_instantiate(impl, attn_configs, attn_inputs, parallelism_config, is_cuda_graph)
                if instance is not None:
                    return instance

    raise Exception(f"can not find mha type for backends={backends}")


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
