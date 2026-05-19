from __future__ import annotations

import importlib
import atexit
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List

logger = logging.getLogger(__name__)
_FILE_LOGGER_INSTALLED = False
_FILE_LOG_PATH = ""


@dataclass(frozen=True)
class DSV4FusionPass:
    name: str
    pass_fn: Callable[[object], object]
    priority: int
    env_gate: str


_PASSES: List[DSV4FusionPass] = []
_FUSIONS_IMPORTED = False
_DYNAMO_LEAFS_REGISTERED = False
_GRAPH_COMPILE_COUNTS: Counter = Counter()
_GRAPH_COMPILE_LABEL_COUNTS: Counter = Counter()
_PASS_MISS_COUNTS: Counter = Counter()
_PASS_HIT_COUNTS: Counter = Counter()
_DYNAMO_CONFIGURED = False
_RTP_RMSNORM_WRAPPER_INSTALLED = False
_RTP_RMSNORM_ORIGINAL = None
_RTP_OPS_ORIGINAL = None
_RTP_OPS_PROXY = None
_RTP_OPS_ORIGINAL_ATTR_PATCHED = False
_RTP_RMSNORM_MUTATING_CUSTOM_OP = None
_TORCH_STREAM_COMPAT_INSTALLED = False
_TORCH_CURRENT_STREAM_ORIGINAL = None
_COMPILE_SUMMARY_REGISTERED = False
_COMPILE_DISABLE_WRAPPERS_INSTALLED = False


def _is_fake_or_meta_tensor(value: object) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.is_meta:
            return True
        from torch._subclasses.fake_tensor import FakeTensor

        return isinstance(value, FakeTensor)
    except Exception:
        return False


def _dsv4_compile_friendly_rtp_rmsnorm(
    output,
    hidden_states,
    weight,
    eps,
    stream_id,
):
    """FX-visible wrapper for the mutating RTP-LLM RMSNorm pybind op.

    Some DSV4 attention helper paths call ``rtp_llm_ops.rmsnorm`` directly
    with a raw stream id instead of going through the compile-friendly
    ``base.cuda.norm.rmsnorm`` wrapper.  Dynamo cannot safely execute that
    pybind op with FakeTensor inputs, so those producer sites graph-break
    before the RMSNorm+FP8-quant pass can see them.  This wrapper preserves
    eager behavior for real tensors and returns ``None`` for FakeTensor/meta
    tracing, exposing a normal mutating FX node that the pass can replace.
    """
    custom_op = _get_dsv4_rtp_rmsnorm_mutating_custom_op()
    if custom_op is not None:
        return custom_op(output, hidden_states, weight, float(eps), int(stream_id))
    if _RTP_RMSNORM_ORIGINAL is None:
        raise RuntimeError("DSV4 RTP RMSNorm wrapper was called before installation")
    if _is_fake_or_meta_tensor(output) or _is_fake_or_meta_tensor(hidden_states):
        return None
    return _RTP_RMSNORM_ORIGINAL(output, hidden_states, weight, eps, stream_id)


def _get_dsv4_rtp_rmsnorm_mutating_custom_op():
    """Return an opaque mutating custom op for direct pybind RMSNorm callsites.

    The DSV4 ``_rmsnorm_weighted`` helper writes into a preallocated output
    tensor and returns a view of that tensor.  Dynamo must see that mutation as
    a graph node, but it must not execute the pybind implementation during
    FakeTensor propagation because the pybind op reads raw data pointers.
    """
    global _RTP_RMSNORM_MUTATING_CUSTOM_OP
    if _RTP_RMSNORM_MUTATING_CUSTOM_OP is not None:
        return _RTP_RMSNORM_MUTATING_CUSTOM_OP
    if _RTP_RMSNORM_ORIGINAL is None:
        return None
    try:
        import torch

        def _rmsnorm_mutating(
            output,
            hidden_states,
            weight,
            eps,
            stream_id,
        ):
            _RTP_RMSNORM_ORIGINAL(output, hidden_states, weight, eps, stream_id)
            return None

        def _rmsnorm_mutating_fake(
            output,
            hidden_states,
            weight,
            eps,
            stream_id,
        ):
            return None

        tensor_type = torch.Tensor
        annotations = {
            "output": tensor_type,
            "hidden_states": tensor_type,
            "weight": tensor_type,
            "eps": float,
            "stream_id": int,
            "return": None,
        }
        _rmsnorm_mutating.__annotations__ = annotations
        _rmsnorm_mutating_fake.__annotations__ = annotations
        _rmsnorm_mutating = torch.library.custom_op(
            "rtp_llm_dsv4::rmsnorm_mutating",
            _rmsnorm_mutating,
            mutates_args=("output",),
        )
        _rmsnorm_mutating.register_fake(_rmsnorm_mutating_fake)
        _RTP_RMSNORM_MUTATING_CUSTOM_OP = getattr(
            _rmsnorm_mutating, "_opoverload", _rmsnorm_mutating
        )
        return _RTP_RMSNORM_MUTATING_CUSTOM_OP
    except Exception as exc:
        logger.warning("failed to register DSV4 mutating RMSNorm custom op: %s", exc)
        return None


def install_torch_stream_cuda_stream_compat() -> None:
    """Expose ``.cuda_stream`` on ``torch.Stream`` for legacy pybind callsites.

    A few older DSV4 helpers still compute raw stream ids inline through
    ``torch.cuda.current_stream().cuda_stream``.  During Dynamo tracing CUDA 13
    PyTorch may return the generic ``torch.Stream`` object, which only exposes
    ``__cuda_stream__``.  Adding this compatibility property keeps those
    callsites traceable so the following pybind RMSNorm call can enter FX and
    be rewritten by the registered pass.
    """
    global _TORCH_STREAM_COMPAT_INSTALLED, _TORCH_CURRENT_STREAM_ORIGINAL
    if _TORCH_STREAM_COMPAT_INSTALLED:
        return
    if not env_flag("DSV4_FUSED_RMSNORM_FP8_QUANT"):
        return
    try:
        import torch

        classes = []
        stream_cls = getattr(torch, "Stream", None)
        if stream_cls is not None:
            classes.append(stream_cls)
        try:
            classes.append(type(torch.cuda.current_stream()))
        except Exception:
            pass

        def _cuda_stream(self):
            stream_protocol = getattr(self, "__cuda_stream__", None)
            if stream_protocol is None:
                raise AttributeError(f"{type(self).__name__} does not expose __cuda_stream__")
            value = stream_protocol()
            if isinstance(value, tuple):
                return value[-1]
            return value

        installed_property = False
        property_errors = []
        for cls in dict.fromkeys(classes):
            if cls is None or hasattr(cls, "cuda_stream"):
                continue
            try:
                setattr(cls, "cuda_stream", property(_cuda_stream))
                installed_property = True
                logger.info("DSV4 GraphFX installed %s.cuda_stream compatibility property", cls.__name__)
            except Exception as exc:
                property_errors.append(f"{cls.__name__}: {exc}")
        if not installed_property:
            _TORCH_CURRENT_STREAM_ORIGINAL = torch.cuda.current_stream

            class _Dsv4CudaStreamCompatProxy:
                def __init__(self, stream):
                    self._stream = stream

                @property
                def cuda_stream(self):
                    return _cuda_stream(self._stream)

                def __cuda_stream__(self):
                    return self._stream.__cuda_stream__()

                def __getattr__(self, name):
                    return getattr(self._stream, name)

            def _current_stream_compat(*args, **kwargs):
                return _Dsv4CudaStreamCompatProxy(
                    _TORCH_CURRENT_STREAM_ORIGINAL(*args, **kwargs)
                )

            torch.cuda.current_stream = _current_stream_compat
            logger.info(
                "DSV4 GraphFX installed torch.cuda.current_stream compatibility wrapper"
                + (f" after property failures: {property_errors}" if property_errors else "")
            )
        _TORCH_STREAM_COMPAT_INSTALLED = True
    except Exception as exc:
        logger.warning("failed to install DSV4 torch.Stream cuda_stream compatibility: %s", exc)


def _refresh_compile_friendly_rtp_rmsnorm_references() -> list[str]:
    """Update already-imported modules that still hold the original op object."""
    if _RTP_OPS_ORIGINAL is None or _RTP_OPS_PROXY is None:
        return []
    replaced_modules = []
    for module_name, module in list(sys.modules.items()):
        if getattr(module, "rtp_llm_ops", None) is _RTP_OPS_ORIGINAL:
            try:
                setattr(module, "rtp_llm_ops", _RTP_OPS_PROXY)
                replaced_modules.append(module_name)
            except Exception:
                pass
    return replaced_modules


def install_compile_friendly_rtp_rmsnorm_wrapper() -> None:
    """Install an env-gated FX-visible alias for direct pybind RMSNorm calls."""
    global _RTP_RMSNORM_WRAPPER_INSTALLED, _RTP_RMSNORM_ORIGINAL, _RTP_OPS_ORIGINAL
    global _RTP_OPS_PROXY, _RTP_OPS_ORIGINAL_ATTR_PATCHED
    if not env_flag("DSV4_FUSED_RMSNORM_FP8_QUANT"):
        return
    if _RTP_RMSNORM_WRAPPER_INSTALLED:
        replaced_modules = _refresh_compile_friendly_rtp_rmsnorm_references()
        if replaced_modules:
            logger.info(
                "DSV4 GraphFX refreshed compile-friendly rtp_llm_ops proxy; updated_modules=%s",
                replaced_modules,
            )
        return
    try:
        import rtp_llm.ops as ops_pkg
        import rtp_llm.ops.compute_ops as compute_ops
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        original_ops = rtp_llm_ops
        current = getattr(original_ops, "rmsnorm", None)
        if current is None or current is _dsv4_compile_friendly_rtp_rmsnorm:
            _RTP_RMSNORM_WRAPPER_INSTALLED = True
            return
        _RTP_RMSNORM_ORIGINAL = current
        _RTP_OPS_ORIGINAL = original_ops

        class _Dsv4RtpLlmOpsProxy:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, name):
                if name == "rmsnorm":
                    return _dsv4_compile_friendly_rtp_rmsnorm
                return getattr(self._wrapped, name)

        proxy = _Dsv4RtpLlmOpsProxy(original_ops)
        _RTP_OPS_PROXY = proxy
        try:
            setattr(original_ops, "rmsnorm", _dsv4_compile_friendly_rtp_rmsnorm)
            _RTP_OPS_ORIGINAL_ATTR_PATCHED = True
        except Exception as exc:
            logger.info("DSV4 GraphFX could not patch original rtp_llm_ops.rmsnorm attr: %s", exc)
        compute_ops.rtp_llm_ops = proxy
        if getattr(ops_pkg, "rtp_llm_ops", None) is original_ops:
            ops_pkg.rtp_llm_ops = proxy
        replaced_modules = _refresh_compile_friendly_rtp_rmsnorm_references()
        custom_op = _get_dsv4_rtp_rmsnorm_mutating_custom_op()
        if custom_op is not None:
            _dynamo_allow_in_graph(custom_op)
        _dynamo_allow_in_graph(_dsv4_compile_friendly_rtp_rmsnorm)
        _RTP_RMSNORM_WRAPPER_INSTALLED = True
        logger.info(
            "DSV4 GraphFX installed compile-friendly rtp_llm_ops proxy for rmsnorm; "
            "original_attr_patched=%s custom_op=%s updated_modules=%s",
            _RTP_OPS_ORIGINAL_ATTR_PATCHED,
            custom_op is not None,
            replaced_modules,
        )
    except Exception as exc:
        logger.warning("failed to install DSV4 compile-friendly RMSNorm wrapper: %s", exc)


def _torch_compile_disable(fn: Callable, *, reason: str) -> Callable:
    """Return a callable excluded from Dynamo/torch.compile tracing."""
    try:
        import torch

        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        if disable is None:
            import torch._dynamo as dynamo

            disable = dynamo.disable
        try:
            return disable(fn, recursive=True, reason=reason)
        except TypeError:
            try:
                return disable(fn, recursive=True)
            except TypeError:
                return disable(fn)
    except Exception as exc:
        logger.warning("failed to compile-disable %s: %s", _callable_log_name(fn), exc)
        return fn


def _install_compile_disable_on_attr(owner: object, attr_name: str, qualname: str) -> bool:
    marker = f"_dsv4_graphfx_disable_{attr_name}"
    if getattr(owner, marker, False):
        return False
    fn = getattr(owner, attr_name, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, "_dsv4_graphfx_compile_disabled", False):
        setattr(owner, marker, True)
        return False
    disabled = _torch_compile_disable(
        fn,
        reason=f"DSV4 GraphFX keeps {qualname} eager to avoid metadata/debug graph fragmentation",
    )
    try:
        setattr(disabled, "_dsv4_graphfx_compile_disabled", True)
        setattr(disabled, "_dsv4_graphfx_compile_disable_original", fn)
    except Exception:
        pass
    setattr(owner, attr_name, disabled)
    setattr(owner, marker, True)
    logger.info("DSV4 GraphFX compile-disabled helper: %s", qualname)
    return True


def _install_compile_disable_targets(module_name: str, attr_names: tuple[str, ...]) -> int:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.info("DSV4 GraphFX skip compile-disable module %s: %s", module_name, exc)
        return 0
    installed = 0
    for attr_name in attr_names:
        installed += int(
            _install_compile_disable_on_attr(
                module,
                attr_name,
                f"{module_name}.{attr_name}",
            )
        )
    return installed


def _install_compile_disable_methods(
    module_name: str,
    class_name: str,
    method_names: tuple[str, ...],
) -> int:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.info("DSV4 GraphFX skip compile-disable module %s: %s", module_name, exc)
        return 0
    cls = getattr(module, class_name, None)
    if cls is None:
        return 0
    installed = 0
    for method_name in method_names:
        installed += int(
            _install_compile_disable_on_attr(
                cls,
                method_name,
                f"{module_name}.{class_name}.{method_name}",
            )
        )
    return installed


def install_dsv4_graphfx_compile_disable_wrappers() -> None:
    """Exclude non-compute DSV4 helpers from the unified forward compile.

    The GraphFX replacement boundary stays at ``DeepSeekV4Model.forward`` for
    both prefill and decode.  These helpers build metadata, update CUDA-graph
    metadata buffers, or record debug tensors; tracing them only creates extra
    Dynamo segments and request-shape guards.  Keeping them eager still lets
    Dynamo resume into the layer compute where the RMSNorm/RoPE/quant patterns
    are visible to FX passes.
    """
    global _COMPILE_DISABLE_WRAPPERS_INSTALLED
    if _COMPILE_DISABLE_WRAPPERS_INSTALLED:
        return
    if not env_flag("DSV4_GRAPHFX_DISABLE_NON_COMPUTE_HELPERS", True):
        return

    installed = 0
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4._record_tensor",
        (
            "_get_buf",
            "should_record_layer",
            "begin",
            "record",
            "record_if_level",
            "dump",
            "_snapshot_cpu",
        ),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.kv_cache_utils",
        ("build_block_tables", "build_block_tables_batched"),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.decode.forward",
        ("build_paged_pool_specs", "build_metadata_eager"),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.prefill.forward",
        ("set_cp_info",),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.hc.base",
        "HCUnitBase",
        ("pre", "post"),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.hc.base",
        "HCHeadBase",
        ("head",),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata",
        (
            "allocate_decode_metadata",
            "update_decode_metadata_in_place",
            "build_decode_metadata",
        ),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata",
        (
            "allocate_decode_metadata_fp8",
            "update_decode_metadata_in_place_fp8",
            "build_decode_metadata_fp8",
            "get_or_build_sched_meta",
        ),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.attention",
        "Attention",
        ("_set_compressor_pool_context",),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.attention_bf16_vllm",
        "AttentionBF16VLLM",
        ("_set_compressor_pool_context",),
    )
    expose_indexer_path1 = env_flag("DSV4_GRAPHFX_EXPOSE_INDEXER_PATH1", True)
    attention_fp8_disabled = (
        "_set_compressor_pool_context",
        "_get_fp8_decode_op",
        "_build_shared_prefill_meta",
        "_build_workspace_meta",
        "_build_swa_prefill_meta_varlen",
        "_decode_write_swa_fp8",
        "_forward_decode_swa_only",
        "_forward_decode_hca",
        "_forward_decode_compressed",
        "_attn_via_workspace",
    )
    if not expose_indexer_path1:
        # Restores the previous coarse boundary.  The default keeps CSA visible
        # so GraphFX can see indexer.wq_b input quant and fold it through the
        # q_lora_a_norm provenance path.
        attention_fp8_disabled = attention_fp8_disabled + ("_forward_decode_csa",)
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.fp8.attention",
        "AttentionFP8",
        attention_fp8_disabled,
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.write_swa",
        ("decode_write_swa_fp8",),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_kv_quant_decode_op",
        ("quantize_v4_kv_decode",),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8.compressor",
        ("_linear_bf16_bf16_fp32",),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.fp8.compressor",
        "CompressorFP8",
        ("_launch", "forward_decode_vectorized"),
    )
    if not expose_indexer_path1:
        installed += _install_compile_disable_methods(
            "rtp_llm.models_py.modules.dsv4.fp8.indexer",
            "IndexerFP8",
            ("forward_decode_vectorized",),
        )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8._indexer_score",
        ("fp8_paged_indexer_score",),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton",
        ("indexer_q_fp8_quant_fold",),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.attention_kernels",
        (
            "attn_fp8_swa_paged",
            "attn_fp8_dual_paged",
        ),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator",
        (
            "build_req_id_per_token",
            "translate_local_to_global_slots",
        ),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op",
        "SparseAttnV4DecodeFp8Op",
        ("forward",),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.decode.fp8_sparse_attn_decode_op",
        "SparseAttnV4DecodeFp8Op",
        ("forward",),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl",
        "DSv4DecodeFmhaImpl",
        ("prepare", "prepare_cuda_graph"),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl",
        "DSv4DecodeFmhaImplFP8",
        ("prepare", "prepare_cuda_graph"),
    )
    _COMPILE_DISABLE_WRAPPERS_INSTALLED = True
    logger.info("DSV4 GraphFX compile-disabled %d non-compute helpers", installed)


def _default_graphfx_log_file() -> str:
    explicit = os.environ.get("DSV4_GRAPHFX_LOG_FILE")
    if explicit:
        return explicit
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if outputs_dir:
        return os.path.join(outputs_dir, "dsv4_graphfx_compile.log")
    return f"/tmp/dsv4_graphfx_compile_{os.getpid()}.log"


def setup_dsv4_fusion_file_logger() -> str:
    """Mirror GraphFX compile/fusion logs into a standalone file.

    Smoke/server logs interleave model loading, RPC, CUDA, and torch.compile
    output. Keeping the fusion logs in their own file makes it much easier to
    check whether the backend ran and whether individual FX passes rewrote a
    graph. Bazel exposes TEST_UNDECLARED_OUTPUTS_DIR to test subprocesses, so
    the default file is collected into test.outputs/outputs.zip.
    """
    global _FILE_LOGGER_INSTALLED, _FILE_LOG_PATH
    if _FILE_LOGGER_INSTALLED:
        return _FILE_LOG_PATH

    log_file = _default_graphfx_log_file()
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(process)d %(levelname)s %(name)s:%(lineno)d %(message)s"
        )
    )
    handler._dsv4_graphfx_file_handler = True  # type: ignore[attr-defined]

    dsv4_logger = logging.getLogger("rtp_llm.models_py.modules.dsv4")
    if not any(
        getattr(existing, "_dsv4_graphfx_file_handler", False)
        for existing in dsv4_logger.handlers
    ):
        dsv4_logger.addHandler(handler)
    dsv4_logger.setLevel(logging.INFO)

    _FILE_LOGGER_INSTALLED = True
    _FILE_LOG_PATH = log_file
    logger.info("DSV4 GraphFX compile log file: %s", log_file)
    return log_file


def _log_dsv4_graphfx_compile_summary() -> None:
    if not _GRAPH_COMPILE_LABEL_COUNTS:
        return
    try:
        logger.info(
            "DSV4 GraphFX compile summary: labels=%s signatures=%s pass_hits=%s pass_misses=%s",
            dict(sorted(_GRAPH_COMPILE_LABEL_COUNTS.items())),
            dict(sorted(_GRAPH_COMPILE_COUNTS.items())),
            {str(key): value for key, value in sorted(_PASS_HIT_COUNTS.items())},
            {str(key): value for key, value in sorted(_PASS_MISS_COUNTS.items())},
        )
    except Exception:
        pass


def _register_compile_summary_logger() -> None:
    global _COMPILE_SUMMARY_REGISTERED
    if _COMPILE_SUMMARY_REGISTERED:
        return
    atexit.register(_log_dsv4_graphfx_compile_summary)
    _COMPILE_SUMMARY_REGISTERED = True


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def register_dsv4_fusion_pass(
    name: str,
    pass_fn: Callable[[object], object],
    *,
    priority: int = 100,
    env_gate: str,
) -> None:
    if any(item.name == name for item in _PASSES):
        return
    _PASSES.append(DSV4FusionPass(name=name, pass_fn=pass_fn, priority=priority, env_gate=env_gate))
    _PASSES.sort(key=lambda item: item.priority)


def ensure_dsv4_fusions_registered() -> None:
    global _FUSIONS_IMPORTED
    setup_dsv4_fusion_file_logger()
    install_torch_stream_cuda_stream_compat()
    install_compile_friendly_rtp_rmsnorm_wrapper()
    install_dsv4_graphfx_compile_disable_wrappers()
    if _FUSIONS_IMPORTED:
        return
    importlib.import_module("rtp_llm.models_py.modules.dsv4.fusions")
    _allow_dsv4_fusion_candidates_in_graph()
    _FUSIONS_IMPORTED = True


def _dynamo_allow_in_graph(fn: Callable) -> None:
    try:
        import torch

        allow_in_graph = getattr(getattr(torch, "compiler", None), "allow_in_graph", None)
        if allow_in_graph is None:
            import torch._dynamo as dynamo

            allow_in_graph = dynamo.allow_in_graph
        allow_in_graph(fn)
    except Exception as exc:
        logger.warning("failed to mark %s allow_in_graph: %s", _callable_log_name(fn), exc)


def _callable_log_name(fn: object) -> str:
    return getattr(fn, "__name__", None) or str(fn)


def _allow_dsv4_fusion_candidates_in_graph() -> None:
    """Keep existing DSV4 Python fusion helpers visible to FX passes.

    The production DSV4 path already calls Triton/Python helpers such as
    ``fused_rmsnorm_rope`` and ``sgl_per_token_group_quant_fp8``. Dynamo may
    inline those helpers into lower-level Triton launcher nodes, which makes a
    high-level GraphFX rewrite brittle. Mark the helpers as allowed graph calls
    so the pass can match the stable operator contract while the replacement
    CUDA wrapper still performs strict runtime shape/stride validation.
    """
    global _DYNAMO_LEAFS_REGISTERED
    if _DYNAMO_LEAFS_REGISTERED:
        return
    try:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
        from rtp_llm.models_py.modules.base.cuda.norm import rmsnorm
        from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_quant_runtime import (
            dsv4_rmsnorm_bf16_fp8_quant_from_provenance,
            dsv4_rmsnorm_bf16_fp8_quant_mutating_producer_token,
            dsv4_rmsnorm_bf16_fp8_quant_producer_token,
            dsv4_fused_rmsnorm_fp8_quant_from_provenance,
            dsv4_rmsnorm_quant_mutating_producer_token,
            dsv4_rmsnorm_quant_producer_token,
            dsv4_rmsnorm_quant_provenance_token,
        )
        from rtp_llm.models_py.modules.dsv4.fusions.kv_rope_fp8_quant_runtime import (
            dsv4_kv_rope_fp8_quant_from_provenance,
            dsv4_kv_rope_quant_producer_token,
        )
        from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
            fused_inv_rope_fp8_quant,
        )
        from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        for fn in (
            rmsnorm,
            _RTP_RMSNORM_MUTATING_CUSTOM_OP,
            rtp_llm_ops.rmsnorm,
            fused_rmsnorm_rope,
            fused_inv_rope_fp8_quant,
            sgl_per_token_group_quant_fp8,
            dsv4_rmsnorm_bf16_fp8_quant_mutating_producer_token,
            dsv4_rmsnorm_bf16_fp8_quant_producer_token,
            dsv4_rmsnorm_bf16_fp8_quant_from_provenance,
            dsv4_rmsnorm_quant_mutating_producer_token,
            dsv4_rmsnorm_quant_producer_token,
            dsv4_rmsnorm_quant_provenance_token,
            dsv4_fused_rmsnorm_fp8_quant_from_provenance,
            dsv4_kv_rope_quant_producer_token,
            dsv4_kv_rope_fp8_quant_from_provenance,
        ):
            if fn is None:
                continue
            try:
                _dynamo_allow_in_graph(fn)
                logger.info("DSV4 GraphFX allow_in_graph registered: %s", _callable_log_name(fn))
            except Exception as exc:
                logger.warning(
                    "failed to register DSV4 GraphFX allow_in_graph function %s: %s",
                    _callable_log_name(fn),
                    exc,
                )
    except Exception as exc:
        logger.warning("failed to register DSV4 GraphFX allow_in_graph functions: %s", exc)
    _DYNAMO_LEAFS_REGISTERED = True


def registered_dsv4_fusion_passes() -> List[DSV4FusionPass]:
    ensure_dsv4_fusions_registered()
    return list(_PASSES)


def _target_name(target: object) -> str:
    return getattr(target, "__name__", str(target))


def _meta_signature(node: object) -> tuple:
    meta = getattr(node, "meta", {}) or {}
    tensor_meta = meta.get("tensor_meta")
    if tensor_meta is None:
        return ()
    shape = getattr(tensor_meta, "shape", None)
    dtype = getattr(tensor_meta, "dtype", None)
    stride = getattr(tensor_meta, "stride", None)
    last_dim = None
    rank = None
    if shape is not None:
        rank = len(shape)
        if rank:
            candidate = shape[-1]
            last_dim = candidate if isinstance(candidate, int) else "dynamic"
    # M/bs/seq are intentionally excluded from the signature. Last dim,
    # dtype, and innermost stride are fixed operator-contract constraints.
    last_stride = None
    if stride:
        candidate = stride[-1]
        last_stride = candidate if isinstance(candidate, int) else "dynamic"
    return ("tensor", str(dtype), rank, last_dim, last_stride)


def _normalize_arg(value: object):
    graph_node_type = None
    try:
        import torch

        graph_node_type = torch.fx.Node
    except Exception:
        graph_node_type = None

    if graph_node_type is not None and isinstance(value, graph_node_type):
        return ("node", value.op, _target_name(value.target), _meta_signature(value))
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_arg(item) for item in value)
    if isinstance(value, slice):
        return ("slice", _normalize_arg(value.start), _normalize_arg(value.stop), _normalize_arg(value.step))
    if isinstance(value, dict):
        return tuple(sorted((str(key), _normalize_arg(item)) for key, item in value.items()))
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, bool) or value is None:
        return value
    if hasattr(value, "node") and value.__class__.__name__.lower().find("sym") >= 0:
        return "dynamic_sym"
    value_s = str(value)
    if value_s.startswith("s") and value_s[1:].isdigit():
        return "dynamic_sym"
    return value_s


def _normalized_graph_signature(gm: object) -> tuple:
    graph = getattr(gm, "graph", None)
    if graph is None:
        return ()
    return tuple(
        (
            node.op,
            _target_name(node.target),
            _meta_signature(node),
            _normalize_arg(node.args),
            _normalize_arg(node.kwargs),
        )
        for node in graph.nodes
    )


def _signature_id(signature: tuple) -> str:
    return str(abs(hash(repr(signature))) % 1_000_000_000)


def record_dsv4_fusion_miss(pass_name: str, reason: str) -> None:
    if env_flag("DSV4_GRAPHFX_MISS_LOG") or env_flag("DSV4_GRAPHFX_COMPILE_STATS"):
        _PASS_MISS_COUNTS[(pass_name, reason)] += 1


def record_dsv4_fusion_hit(pass_name: str, count: int = 1) -> None:
    if env_flag("DSV4_GRAPHFX_MISS_LOG") or env_flag("DSV4_GRAPHFX_COMPILE_STATS"):
        _PASS_HIT_COUNTS[pass_name] += count


def _counter_delta(after: Counter, before: Counter) -> dict:
    return {str(key): value for key, value in (after - before).items() if value}


def _should_dump_graph(targets: list[str], code: str) -> bool:
    if not env_flag("DSV4_GRAPHFX_DUMP_GRAPHS"):
        return False
    graph_filter = os.environ.get("DSV4_GRAPHFX_DUMP_FILTER", "").strip()
    if not graph_filter:
        return True
    haystack = "\n".join(targets) + "\n" + code
    return any(item.strip() and item.strip() in haystack for item in graph_filter.split(","))


def _graph_signature(gm: object) -> tuple:
    graph = getattr(gm, "graph", None)
    if graph is None:
        return ()
    return tuple(
        (
            node.op,
            _target_name(node.target),
            tuple(str(arg) for arg in node.args),
            tuple(sorted((str(key), str(value)) for key, value in node.kwargs.items())),
        )
        for node in graph.nodes
    )


def apply_registered_dsv4_fusions(gm: object, *, return_changed: bool = False):
    ensure_dsv4_fusions_registered()
    debug = env_flag("DSV4_FUSION_REGISTRY_DEBUG")
    if debug:
        logger.info(
            "DSV4 fusion registry passes: %s",
            [
                {
                    "name": item.name,
                    "env_gate": item.env_gate,
                    "enabled": env_flag(item.env_gate),
                }
                for item in _PASSES
            ],
        )
    out = gm
    changed = False
    for item in _PASSES:
        if not env_flag(item.env_gate):
            if debug:
                logger.info("skip DSV4 fusion pass %s: %s is disabled", item.name, item.env_gate)
            continue
        if debug:
            logger.info("apply DSV4 fusion pass %s", item.name)
        before = _graph_signature(out)
        out = item.pass_fn(out)
        changed = changed or _graph_signature(out) != before
    if return_changed:
        return out, changed
    return out


_MUTATING_LEAF_TARGETS = {
    "fp8_gemm_nt",
    "fake_mutating_rmsnorm",
}


def eliminate_dead_code_preserving_dsv4_side_effects(gm: object) -> None:
    """Run FX DCE without dropping opaque mutating CUDA calls.

    Some compile-visible helpers, for example DeepGEMM's ``fp8_gemm_nt``,
    mutate an output tensor and return ``None``.  After a fusion pass replaces
    producers such as RMSNorm+quant, the mutating call can have no direct FX
    users even though the returned model output depends on its side effect.
    The default FX DCE cannot infer that for plain Python call targets, so DSV4
    passes use this helper whenever rewritten graphs may contain such leaves.
    """
    graph = getattr(gm, "graph", None)
    if graph is None:
        return

    def is_impure_node(node: object) -> bool:
        target_name = _target_name(getattr(node, "target", None))
        if "rmsnorm" in target_name.lower():
            # ``rtp_llm_ops.rmsnorm(out, x, weight, eps, stream)`` mutates its
            # first argument and must survive DCE until a pass explicitly
            # erases it.  The compile-friendly functional wrapper
            # ``rmsnorm(x, weight, eps)`` and RoPE helpers are side-effect free
            # and should be removable after their GraphFX fusion rewrites.
            lowered = target_name.lower()
            if "rope" in lowered:
                return False
            return len(getattr(node, "args", ()) or ()) >= 4
        if target_name in _MUTATING_LEAF_TARGETS:
            return True
        is_impure = getattr(node, "is_impure", None)
        if callable(is_impure):
            try:
                return bool(is_impure())
            except TypeError:
                try:
                    return bool(is_impure(True))
                except Exception:
                    return False
        return False

    graph.eliminate_dead_code(is_impure_node=is_impure_node)


def any_dsv4_fusion_enabled() -> bool:
    ensure_dsv4_fusions_registered()
    return any(env_flag(item.env_gate) for item in _PASSES)


def _configure_dynamo_for_dsv4_graphfx() -> None:
    """Set Dynamo knobs needed by the GraphFX-only DSV4 fusion path.

    The compile boundary is the full V4 Python model forward, but request
    dimensions (M / batch / sequence) are intentionally dynamic.  These flags
    reduce specialisation on parameter objects and scalar ``.item()`` reads
    while preserving CUDA-wrapper runtime validation for fixed hidden/layout
    contracts.
    """
    global _DYNAMO_CONFIGURED
    if _DYNAMO_CONFIGURED:
        return
    try:
        import torch._dynamo as dynamo

        dynamo.config.dynamic_shapes = True
        dynamo.config.assume_static_by_default = False
        dynamo.config.force_parameter_static_shapes = False
        dynamo.config.force_nn_module_property_static_shapes = False
        dynamo.config.allow_unspec_int_on_nn_module = True
        dynamo.config.capture_scalar_outputs = True
        dynamo.config.capture_dynamic_output_shape_ops = True
        dynamo.config.prefer_deferred_runtime_asserts_over_guards = True
        dynamo.config.guard_nn_modules = False
        dynamo.config.guard_nn_modules_using_dict_tags = False
        # Keep enough entries for prompt/batch variation while logs tell us
        # whether signatures are actually reused.
        dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), 128)
        dynamo.config.accumulated_cache_size_limit = max(
            int(dynamo.config.accumulated_cache_size_limit), 512
        )
        logger.info(
            "DSV4 GraphFX Dynamo config: dynamic=True "
            "dynamic_shapes=%s assume_static_by_default=%s "
            "force_parameter_static_shapes=%s force_nn_module_property_static_shapes=%s "
            "allow_unspec_int_on_nn_module=%s capture_scalar_outputs=%s "
            "capture_dynamic_output_shape_ops=%s prefer_deferred_runtime_asserts_over_guards=%s "
            "guard_nn_modules=%s guard_nn_modules_using_dict_tags=%s "
            "cache_size_limit=%s accumulated_cache_size_limit=%s",
            dynamo.config.dynamic_shapes,
            dynamo.config.assume_static_by_default,
            dynamo.config.force_parameter_static_shapes,
            dynamo.config.force_nn_module_property_static_shapes,
            dynamo.config.allow_unspec_int_on_nn_module,
            dynamo.config.capture_scalar_outputs,
            dynamo.config.capture_dynamic_output_shape_ops,
            dynamo.config.prefer_deferred_runtime_asserts_over_guards,
            dynamo.config.guard_nn_modules,
            dynamo.config.guard_nn_modules_using_dict_tags,
            dynamo.config.cache_size_limit,
            dynamo.config.accumulated_cache_size_limit,
        )
        try:
            import torch.fx.experimental._config as fx_config

            fx_config.use_duck_shape = False
            logger.info("DSV4 GraphFX FX config: use_duck_shape=%s", fx_config.use_duck_shape)
        except Exception as exc:
            logger.info("DSV4 GraphFX FX config use_duck_shape unchanged: %s", exc)
    except Exception as exc:
        logger.warning("failed to configure DSV4 GraphFX Dynamo options: %s", exc)
    _DYNAMO_CONFIGURED = True


def _cuda_graph_capture_state() -> str:
    try:
        import torch

        checker = getattr(torch.cuda, "is_current_stream_capturing", None)
        if checker is None:
            return "unknown"
        return "capturing" if bool(checker()) else "not_capturing"
    except Exception:
        return "unknown"


def dsv4_fusion_backend(gm, example_inputs, *, label: str = "unknown"):
    setup_dsv4_fusion_file_logger()
    _register_compile_summary_logger()
    targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
    code = getattr(gm, "code", "<no code>")
    normalized_sig = _normalized_graph_signature(gm)
    normalized_sig_id = _signature_id(normalized_sig)
    labeled_sig = f"{label}:{normalized_sig_id}"
    _GRAPH_COMPILE_COUNTS[labeled_sig] += 1
    _GRAPH_COMPILE_LABEL_COUNTS[label] += 1
    if env_flag("DSV4_GRAPHFX_COMPILE_STATS"):
        logger.info(
            "DSV4 GraphFX compile stats: label=%s signature=%s count=%d "
            "label_total=%d nodes=%d cuda_graph=%s dynamic_dims_ignored=[M,bs,seq]",
            label,
            normalized_sig_id,
            _GRAPH_COMPILE_COUNTS[labeled_sig],
            _GRAPH_COMPILE_LABEL_COUNTS[label],
            len(targets),
            _cuda_graph_capture_state(),
        )
    if env_flag("DSV4_FUSION_REGISTRY_DEBUG"):
        logger.info("DSV4 GraphFX backend captured graph targets: %s", targets)
        if _should_dump_graph(targets, code):
            logger.info("DSV4 GraphFX captured graph code:\n%s", code)
    miss_before = _PASS_MISS_COUNTS.copy()
    hit_before = _PASS_HIT_COUNTS.copy()
    fused, changed = apply_registered_dsv4_fusions(gm, return_changed=True)
    if env_flag("DSV4_GRAPHFX_MISS_LOG") or env_flag("DSV4_GRAPHFX_COMPILE_STATS"):
        miss_delta = _counter_delta(_PASS_MISS_COUNTS, miss_before)
        hit_delta = _counter_delta(_PASS_HIT_COUNTS, hit_before)
        logger.info(
            "DSV4 GraphFX pass stats: label=%s signature=%s changed=%s hits=%s misses=%s",
            label,
            normalized_sig_id,
            changed,
            hit_delta,
            miss_delta,
        )
    if not changed and env_flag("DSV4_GRAPHFX_FALLBACK_UNFUSED", True):
        logger.info("DSV4 GraphFX no fusion rewrite; keep original FX graph for this segment")
        return gm.forward
    fused.recompile()
    if env_flag("DSV4_FUSION_REGISTRY_DEBUG") and changed:
        logger.info("DSV4 GraphFX rewritten graph code:\n%s", getattr(fused, "code", "<no code>"))
    return fused.forward


def compile_with_dsv4_fusions(fn, *, label: str | None = None, **compile_kwargs):
    import torch

    ensure_dsv4_fusions_registered()
    _configure_dynamo_for_dsv4_graphfx()
    if env_flag("DSV4_GRAPHFX_FALLBACK_UNFUSED", True):
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = True
    compile_label = label or getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or "unknown"

    def _backend(gm, example_inputs):
        return dsv4_fusion_backend(gm, example_inputs, label=compile_label)

    kwargs = {"backend": _backend, "fullgraph": False}
    kwargs.update(compile_kwargs)
    compiled = torch.compile(fn, **kwargs)

    def _compiled_with_dsv4_reference_refresh(*args, **inner_kwargs):
        # DSV4 modules are imported lazily during model construction.  Refresh
        # the module-level rtp_llm_ops proxy immediately before Dynamo captures
        # a new graph so late imports cannot retain the pybind RMSNorm object.
        install_compile_friendly_rtp_rmsnorm_wrapper()
        return compiled(*args, **inner_kwargs)

    return _compiled_with_dsv4_reference_refresh
