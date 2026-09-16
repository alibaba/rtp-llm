import logging
from typing import Callable, Dict, List, Optional, Union

from rtp_llm.config.cuda_graph import (
    CudaGraphSelectionMode,
    GenerationPrefillCudaGraphUnsupportedBackend,
)
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    KvCacheDataType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W

AttentionImpl = Union[FMHAImplBase, MlaImplBase]


def _normalize_cuda_graph_selection_mode(
    is_cuda_graph: bool,
    mode: Optional[Union[str, CudaGraphSelectionMode]],
) -> CudaGraphSelectionMode:
    if mode is None:
        return (
            CudaGraphSelectionMode.DECODE_GRAPH
            if is_cuda_graph
            else CudaGraphSelectionMode.EAGER
        )
    return CudaGraphSelectionMode(mode)


def _matches_cuda_graph_selection_mode(
    instance: AttentionImpl, mode: CudaGraphSelectionMode
) -> bool:
    if mode == CudaGraphSelectionMode.EAGER:
        return True
    if not instance.support_cuda_graph():
        return False
    if mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH:
        return instance.supports_generation_prefill_cuda_graph()
    return True


def _implementation_allows_cuda_graph_selection_mode(
    impl: type[AttentionImpl], mode: CudaGraphSelectionMode
) -> bool:
    """Filter role-specific implementations before support()/construction.

    Most attention implementations participate in the existing eager/decode
    routing and therefore leave ``cuda_graph_selection_modes`` unset. A
    backend that exists for one graph role only declares the exact modes it
    accepts, preventing a graph-shaped input for another role from selecting
    it merely because ``is_cuda_graph`` is true.
    """

    allowed_modes = getattr(impl, "cuda_graph_selection_modes", None)
    return allowed_modes is None or mode in allowed_modes


# Lists to store registered implementations
PREFILL_MHA_IMPS: List[type[FMHAImplBase]] = []
DECODE_MHA_IMPS: List[type[FMHAImplBase]] = []
PREFILL_MLA_IMPS: List[type[MlaImplBase]] = []
DECODE_MLA_IMPS: List[type[MlaImplBase]] = []

FLASHINFER_TRTLLM_GEN_IMPLS = {
    "FlashInferTRTLLMPrefillImpl",
    "FlashInferTRTLLMSpecDecodeImpl",
    "FlashInferTRTLLMDecodeImpl",
}

def get_mla_impl(
    attn_configs: AttentionConfigs,
    weight: ModelWeights,
    attn_inputs: PyAttentionInputs,
    fmha_config: Optional[FMHAConfig] = None,
    quant_config: Optional[object] = None,
    is_cuda_graph: bool = False,
    max_seq_len: int = 0,
    parallelism_config: Optional[ParallelismConfig] = None,
    cuda_graph_selection_mode: Optional[Union[str, CudaGraphSelectionMode]] = None,
) -> MlaImplBase:

    selection_mode = _normalize_cuda_graph_selection_mode(
        is_cuda_graph, cuda_graph_selection_mode
    )

    mla_impls = PREFILL_MLA_IMPS if attn_inputs.is_prefill else DECODE_MLA_IMPS
    candidates = _select_attn_impls(mla_impls, fmha_config, attn_inputs.is_prefill)
    for impl in candidates:
        if not _implementation_allows_cuda_graph_selection_mode(impl, selection_mode):
            continue
        # Check support before creating instance
        if not impl.support(attn_configs, attn_inputs):
            continue
        cos_sin_cache = weight.get_global_weight_or_none(W.rope_cos_sin_cache)
        # Short-circuit before touching cu_kv_seqlens when CP is enabled to avoid
        # an unnecessary GPU->CPU sync on the hot prefill routing path.
        cp_enabled = (
            parallelism_config is not None
            and parallelism_config.prefill_cp_config.is_enabled()
        )
        use_fast_path = (
            attn_inputs.is_prefill
            and not cp_enabled
            and attn_inputs.cu_kv_seqlens_device.max().item() <= attn_configs.indexer_topk
        )

        # Check parallelism config support (e.g. CP filtering). The fast path is
        # never taken when CP is enabled, so it bypasses this check (matches
        # upstream's "support fast path for cp prefill").
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
        if selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH:
            if not _matches_cuda_graph_selection_mode(instance, selection_mode):
                raise GenerationPrefillCudaGraphUnsupportedBackend(
                    "selected semantic attention backend "
                    f"{type(instance).__name__} is not generation-prefill CUDA Graph safe"
                )
            return instance
        if _matches_cuda_graph_selection_mode(instance, selection_mode):
            return instance
    if selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH:
        raise GenerationPrefillCudaGraphUnsupportedBackend(
            "MLA has no generation-prefill CUDA Graph implementation"
        )
    raise Exception("can not find mla type")


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


def _expand_flashinfer_alias(names: set) -> set:
    """Expand the flashinfer alias in both directions so that the public alias
    "flashinfer" and the canonical NAME "py_flashinfer" are treated
    interchangeably in blocklists / known-name sets. Returns a new set."""
    expanded = set(names)
    if "flashinfer" in expanded:
        expanded.add("py_flashinfer")
    if "py_flashinfer" in expanded:
        expanded.add("flashinfer")
    return expanded


def _get_blocked_backends(fmha_config: FMHAConfig) -> set:
    if not fmha_config.disable_attn_backends:
        return set()
    blocked = {s.strip() for s in fmha_config.disable_attn_backends.split(",") if s.strip()}
    # Expand alias so the blocklist works in both auto and explicit backend modes.
    return _expand_flashinfer_alias(blocked)


def _blocklist_known_names() -> set:
    """Valid names for the GLOBAL disable_attn_backends blocklist.

    disable_attn_backends applies to both prefill and decode AND to both the MHA
    and MLA registries, so its validity is the UNION of all four registries (plus
    the pseudo/alias names). Validating a global blocklist against only the
    current stage's / current attention-type's registry would wrongly reject a
    name that is valid for another stage or for MLA (e.g. rejecting "sparse_mla"
    during an MHA call, or a decode-only name during a prefill call).
    """
    names = {
        getattr(impl, "NAME", None)
        for impl in (
            *PREFILL_MHA_IMPS,
            *DECODE_MHA_IMPS,
            *PREFILL_MLA_IMPS,
            *DECODE_MLA_IMPS,
        )
    }
    names.discard(None)
    return _expand_flashinfer_alias(names | {"auto", "none", "flashinfer"})


def _select_attn_impls(
    impls: List[type],
    fmha_config: Optional[FMHAConfig],
    is_prefill: bool,
) -> List[type]:
    """Resolve which attention impl classes to try, in priority order.

    Shared by the MHA and MLA registries so both honor the same backend config:
      - ``attn_backend`` / ``prefill_attn_backend`` / ``decode_attn_backend``
        (comma-separated ordered lists),
      - ``attn_backend=none`` (disables attention -> raises),
      - the global ``disable_attn_backends`` blocklist,
      - validation of explicit / blocked backend names.

    Returns the ordered list of candidate impl classes (blocklist already
    applied). ``auto`` mode preserves registration order and lets callers apply
    any additional per-impl gating (e.g. MHA legacy flags). Callers keep their
    own instantiation/support logic.
    """
    if fmha_config is None:
        backends = ["auto"]
        blocked = set()
    else:
        backends = _get_effective_backends(fmha_config, is_prefill)
        blocked = _get_blocked_backends(fmha_config)

    if backends == ["none"]:
        raise Exception("Attention is disabled (attn_backend=none)")

    # Build registry metadata for the passed-in registry (MHA or MLA).
    registered_names = set()
    name_to_impls: Dict[str, List[type]] = {}
    for impl in impls:
        name = getattr(impl, "NAME", None)
        if name:
            registered_names.add(name)
            name_to_impls.setdefault(name, []).append(impl)

    # Public alias: "flashinfer" refers to the Python FlashInfer backend (MHA only).
    if "py_flashinfer" in registered_names:
        name_to_impls.setdefault("flashinfer", []).extend(
            name_to_impls.get("py_flashinfer", [])
        )
    blocked = _expand_flashinfer_alias(blocked)

    # Explicit attn_backend names are validated against THIS registry (selecting
    # an MHA-only backend for an MLA model, or vice versa, should fail loudly).
    known_names = registered_names | {"auto", "none", "flashinfer"}
    for backend_name in backends:
        if backend_name not in known_names:
            raise ValueError(
                f"Unknown attention backend {backend_name!r}. "
                f"Registered backends: {sorted(registered_names)}"
            )
    # disable_attn_backends is GLOBAL: validate against the union of all registries.
    blocklist_known_names = _blocklist_known_names()
    for blocked_name in blocked:
        if blocked_name not in blocklist_known_names:
            raise ValueError(
                f"Unknown attention backend in disable_attn_backends: {blocked_name!r}. "
                f"Valid backends: {sorted(blocklist_known_names)}"
            )

    if backends == ["auto"]:
        return [
            impl
            for impl in impls
            if not (getattr(impl, "NAME", None) and getattr(impl, "NAME") in blocked)
        ]
    # Explicit backend list: iterate in user-specified order, resolving each name
    # to its impl(s) and skipping blocked names.
    candidates: List[type] = []
    for backend_name in backends:
        if backend_name in blocked:
            continue
        resolved_name = "py_flashinfer" if backend_name == "flashinfer" else backend_name
        candidates.extend(name_to_impls.get(resolved_name, []))
    return candidates


def _is_fmha_impl_disabled_legacy(impl_class: type, fmha_config: FMHAConfig) -> bool:
    """Legacy boolean flag check. Only called when effective_backend == "auto"."""
    # Global FMHA switch: when false, disable all MHA implementations.
    if not fmha_config.enable_fmha:
        return True
    impl_class_name = impl_class.__name__
    if "XQA" in impl_class_name:
        return not fmha_config.enable_xqa
    elif impl_class_name in {"TRTMHAImpl", "FlashInferTRTLLMFMHAv2PrefillImpl"}:
        return not fmha_config.enable_trt_fmha or not fmha_config.enable_open_source_fmha
    elif impl_class_name in {
        "TRTPagedMHAImpl",
        "FlashInferTRTLLMFMHAv2PagedPrefillImpl",
    }:
        return not fmha_config.enable_paged_trt_fmha or not fmha_config.enable_open_source_fmha
    elif impl_class_name == "PyFlashinferHybridPrefillImpl":
        return (
            fmha_config.disable_flashinfer_hybrid_prefill
            or fmha_config.disable_flash_infer
        )
    elif "FlashInfer" in impl_class_name or "Flashinfer" in impl_class_name:
        return fmha_config.disable_flash_infer
    elif impl_class_name == "AiterPrefillImplTriton":
        return not (fmha_config.use_triton_pa and fmha_config.use_asm_pa)
    elif "AiterDecodeImplAsm" in impl_class_name:
        if fmha_config.use_triton_pa:
            return True
        return not fmha_config.use_asm_pa
    elif (
        "AiterPrefillImplAsm" in impl_class_name
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


def get_fmha_impl(
    attn_configs: AttentionConfigs,
    weight: ModelWeights,
    attn_inputs: PyAttentionInputs,
    fmha_config: Optional[FMHAConfig] = None,
    quant_config: Optional[object] = None,
    is_cuda_graph: bool = False,
    max_seq_len: int = 0,
    parallelism_config: Optional[ParallelismConfig] = None,
    cuda_graph_selection_mode: Optional[Union[str, CudaGraphSelectionMode]] = None,
) -> FMHAImplBase:
    selection_mode = _normalize_cuda_graph_selection_mode(
        is_cuda_graph, cuda_graph_selection_mode
    )
    # Set is_cuda_graph as dynamic attribute on attn_inputs for base class to read
    attn_inputs.is_cuda_graph = is_cuda_graph
    mha_impls = PREFILL_MHA_IMPS if attn_inputs.is_prefill else DECODE_MHA_IMPS
    strict_impl_selection = False
    if get_device_type() == DeviceType.ROCm:
        from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
            validate_v_layout,
        )

        strict_impl_selection = validate_v_layout(
            attn_configs, attn_inputs, fmha_config
        )

    # Shared backend resolution (attn_backend / overrides / none / blocklist +
    # name validation). Returns candidate impls in priority order.
    is_auto = fmha_config is None or _get_effective_backends(
        fmha_config, attn_inputs.is_prefill
    ) == ["auto"]
    candidates = _select_attn_impls(mha_impls, fmha_config, attn_inputs.is_prefill)

    for impl in candidates:
        impl_class_name = impl.__name__
        if not _implementation_allows_cuda_graph_selection_mode(impl, selection_mode):
            continue
        if is_auto and fmha_config and _is_fmha_impl_disabled_legacy(impl, fmha_config):
            continue

        # Check support before creating instance
        if not impl.support(attn_configs, attn_inputs):
            continue

        # Check if implementation supports parallelism config
        if not impl.support_parallelism_config(parallelism_config):
            continue
        kwargs = {"fmha_config": fmha_config} if impl.accepts_fmha_config else {}
        try:
            instance = impl(attn_configs, attn_inputs, parallelism_config, **kwargs)
        except Exception as e:
            if (
                selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH
                or strict_impl_selection
                or (
                    isinstance(e, RuntimeError)
                    and "illegal memory access" in str(e).lower()
                )
            ):
                raise
            logging.warning(f"Failed to instantiate {impl_class_name}: {e}")
            continue
        if selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH:
            # Backend priority is part of model semantics. Do not skip an eager
            # backend (for example HeadWise sink/sliding-window attention) and
            # silently capture a lower-priority dense implementation.
            if not _matches_cuda_graph_selection_mode(instance, selection_mode):
                raise GenerationPrefillCudaGraphUnsupportedBackend(
                    "selected semantic attention backend "
                    f"{impl_class_name} is not generation-prefill CUDA Graph safe"
                )
            return instance
        if _matches_cuda_graph_selection_mode(instance, selection_mode):
            return instance

    if (
        attn_configs.rope_config.style == RopeStyle.Mrope
        and not attn_configs.rope_config.mrope_interleaved
    ):
        raise ValueError(
            "No registered attention implementation supports non-interleaved MRoPE "
            "on this backend. Qwen2-VL/Qwen2.5-VL checkpoints use the "
            "non-interleaved layout by default; do not flip mrope_interleaved because "
            "that changes RoPE semantics. Use a CUDA backend for these checkpoints."
        )
    if selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH:
        raise GenerationPrefillCudaGraphUnsupportedBackend(
            "no generation-prefill CUDA Graph attention implementation "
            "matches the current model, cache layout, dtype, and GPU"
        )
    raise Exception("can not find mha type")


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
        cuda_graph_selection_mode: Optional[Union[str, CudaGraphSelectionMode]] = None,
    ) -> AttentionImpl:
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
            cuda_graph_selection_mode,
        )
        logging.info(
            "Selected attention implementation: type=%s stage=%s cuda_graph=%s impl=%s",
            key_str,
            "prefill" if attn_inputs.is_prefill else "decode",
            is_cuda_graph,
            type(instance).__name__,
        )
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
