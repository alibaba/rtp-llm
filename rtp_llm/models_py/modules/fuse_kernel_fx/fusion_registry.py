"""GraphFX fusion registry for Qwen3.5 / GLM5 / Qwen3-MoE / DSV3.2 fuse kernels.

This module mirrors the DSV4 GraphFX registry pattern but lives under
``rtp_llm/models_py/modules/fuse_kernel_fx/`` and uses the ``QWEN35_*`` env
namespace.  Its responsibilities:

* Register a list of FX passes that rewrite unfused subgraphs into calls of
  the existing fused Triton kernels (``fused_add_rmsnorm_fp8_quant`` etc.).
* Provide a custom torch.compile backend that runs the registered passes,
  with per-pass env gating, miss/hit counters and optional graph dumping.
* Configure Dynamo for dynamic-shape friendly capture.
* Register pybind/mutating kernels (``rtp_llm_ops.fused_add_rmsnorm``) as
  ``torch.library`` custom ops so FX can see them as opaque mutating nodes
  without executing the pybind on FakeTensor inputs.
* Compile-disable a curated list of Python-only helpers (debug dump, metadata
  builders) so Dynamo does not pollute the captured graph with them.

Phase 1 only registers the ``add_rmsnorm_fp8_quant_fx`` pass.  Subsequent
phases plug additional passes through ``register_qwen35_fusion_pass``.
"""

from __future__ import annotations

import atexit
import importlib
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
class Qwen35FusionPass:
    name: str
    pass_fn: Callable[[object], object]
    priority: int
    env_gate: str


_PASSES: List[Qwen35FusionPass] = []
_FUSIONS_IMPORTED = False
_DYNAMO_LEAFS_REGISTERED = False
_GRAPH_COMPILE_LABEL_COUNTS: Counter = Counter()
_GRAPH_COMPILE_COUNTS: Counter = Counter()
_PASS_HIT_COUNTS: Counter = Counter()
_PASS_MISS_COUNTS: Counter = Counter()
_DYNAMO_CONFIGURED = False
_COMPILE_DISABLE_INSTALLED = False
_COMPILE_SUMMARY_REGISTERED = False
_FUSED_ADD_RMSNORM_WRAPPER_INSTALLED = False
_FUSED_ADD_RMSNORM_ORIGINAL = None
_FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP = None
_TORCH_STREAM_COMPAT_INSTALLED = False


# ------------------------------------------------------------------ env / log


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _default_log_file() -> str:
    explicit = os.environ.get("QWEN35_GRAPHFX_LOG_FILE")
    if explicit:
        return explicit
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if outputs_dir:
        return os.path.join(outputs_dir, "qwen35_graphfx_compile.log")
    return f"/tmp/qwen35_graphfx_compile_{os.getpid()}.log"


def setup_qwen35_fusion_file_logger() -> str:
    global _FILE_LOGGER_INSTALLED, _FILE_LOG_PATH
    if _FILE_LOGGER_INSTALLED:
        return _FILE_LOG_PATH
    log_file = _default_log_file()
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
    handler._qwen35_graphfx_file_handler = True  # type: ignore[attr-defined]
    pkg_logger = logging.getLogger("rtp_llm.models_py.modules.fuse_kernel_fx")
    if not any(
        getattr(existing, "_qwen35_graphfx_file_handler", False)
        for existing in pkg_logger.handlers
    ):
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.INFO)
    _FILE_LOGGER_INSTALLED = True
    _FILE_LOG_PATH = log_file
    logger.info("QWEN35 GraphFX log file: %s", log_file)
    return log_file


# ------------------------------------------------------------------ registry


def register_qwen35_fusion_pass(
    name: str,
    pass_fn: Callable[[object], object],
    *,
    priority: int = 100,
    env_gate: str,
) -> None:
    if any(item.name == name for item in _PASSES):
        return
    _PASSES.append(
        Qwen35FusionPass(
            name=name, pass_fn=pass_fn, priority=priority, env_gate=env_gate
        )
    )
    _PASSES.sort(key=lambda item: item.priority)


def registered_qwen35_fusion_passes() -> List[Qwen35FusionPass]:
    ensure_qwen35_fusions_registered()
    return list(_PASSES)


def ensure_qwen35_fusions_registered() -> None:
    """Idempotent setup: install pybind wrappers, compile-disables, import passes."""
    global _FUSIONS_IMPORTED
    setup_qwen35_fusion_file_logger()
    install_torch_stream_cuda_stream_compat()
    install_compile_friendly_fused_add_rmsnorm_wrapper()
    install_qwen35_graphfx_compile_disable_wrappers()
    if _FUSIONS_IMPORTED:
        return
    importlib.import_module("rtp_llm.models_py.modules.fuse_kernel_fx")
    _allow_qwen35_fusion_candidates_in_graph()
    _FUSIONS_IMPORTED = True


# ------------------------------- pybind / mutating-op compile-friendly wrapper


def _is_fake_or_meta_tensor(value: object) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.is_meta:
            return True
        from torch._subclasses.fake_tensor import FakeTensor

        return isinstance(value, FakeTensor)
    except Exception:
        return False


def install_torch_stream_cuda_stream_compat() -> None:
    """Expose ``.cuda_stream`` on ``torch.Stream`` for legacy pybind callsites.

    ``RMSResNorm.forward`` reads ``torch.cuda.current_stream().cuda_stream`` to
    pass a raw ``cuda_stream`` int into the pybind ``rtp_llm_ops.fused_add_rmsnorm``
    op.  In CUDA 13 / newer PyTorch this attribute is missing on the generic
    ``torch.Stream``.  Installing a property keeps the code traceable so the
    following pybind call lands in the FX graph for the pass to rewrite.
    """
    global _TORCH_STREAM_COMPAT_INSTALLED
    if _TORCH_STREAM_COMPAT_INSTALLED:
        return
    if not env_flag("QWEN35_FUSED_ADD_RMSNORM_FP8_QUANT") and not env_flag(
        "QWEN35_GRAPHFX_FUSION"
    ):
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
                raise AttributeError(f"{type(self).__name__} has no __cuda_stream__")
            value = stream_protocol()
            if isinstance(value, tuple):
                return value[-1]
            return value

        for cls in dict.fromkeys(classes):
            if cls is None or hasattr(cls, "cuda_stream"):
                continue
            try:
                setattr(cls, "cuda_stream", property(_cuda_stream))
                logger.info(
                    "QWEN35 GraphFX installed %s.cuda_stream compat", cls.__name__
                )
            except Exception as exc:
                logger.info(
                    "QWEN35 GraphFX could not patch %s.cuda_stream: %s", cls, exc
                )
        _TORCH_STREAM_COMPAT_INSTALLED = True
    except Exception as exc:
        logger.warning("QWEN35 torch.Stream compat install failed: %s", exc)


def _build_fused_add_rmsnorm_mutating_custom_op():
    global _FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP
    if _FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP is not None:
        return _FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP
    if _FUSED_ADD_RMSNORM_ORIGINAL is None:
        return None
    try:
        import torch

        def _mutating_impl(
            hidden_states,
            residual,
            weight,
            eps,
            stream_id,
        ):
            _FUSED_ADD_RMSNORM_ORIGINAL(hidden_states, residual, weight, eps, stream_id)
            return None

        def _mutating_fake(
            hidden_states,
            residual,
            weight,
            eps,
            stream_id,
        ):
            return None

        annotations = {
            "hidden_states": torch.Tensor,
            "residual": torch.Tensor,
            "weight": torch.Tensor,
            "eps": float,
            "stream_id": int,
            "return": None,
        }
        _mutating_impl.__annotations__ = annotations
        _mutating_fake.__annotations__ = annotations
        op = torch.library.custom_op(
            "rtp_llm_qwen35::fused_add_rmsnorm_mutating",
            _mutating_impl,
            mutates_args=("hidden_states", "residual"),
        )
        op.register_fake(_mutating_fake)
        _FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP = getattr(op, "_opoverload", op)
        return _FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP
    except Exception as exc:
        logger.warning(
            "QWEN35 GraphFX mutating fused_add_rmsnorm op build failed: %s", exc
        )
        return None


def install_compile_friendly_fused_add_rmsnorm_wrapper() -> None:
    """Replace ``rtp_llm_ops.fused_add_rmsnorm`` with a torch.library custom op.

    The pybind op mutates ``hidden_states`` and ``residual`` in place and reads
    raw data pointers — Dynamo cannot trace it directly with FakeTensor inputs.
    Wrapping it as a custom op keeps the mutating contract visible in FX while
    short-circuiting fake-tensor execution to a no-op.
    """
    global _FUSED_ADD_RMSNORM_WRAPPER_INSTALLED, _FUSED_ADD_RMSNORM_ORIGINAL
    if _FUSED_ADD_RMSNORM_WRAPPER_INSTALLED:
        return
    if not env_flag("QWEN35_GRAPHFX_FUSION"):
        return
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        original = getattr(rtp_llm_ops, "fused_add_rmsnorm", None)
        if original is None:
            logger.info(
                "QWEN35 GraphFX: rtp_llm_ops.fused_add_rmsnorm not present, skip wrapper"
            )
            return
        if getattr(original, "_qwen35_graphfx_wrapped", False):
            _FUSED_ADD_RMSNORM_WRAPPER_INSTALLED = True
            return
        _FUSED_ADD_RMSNORM_ORIGINAL = original
        custom_op = _build_fused_add_rmsnorm_mutating_custom_op()
        if custom_op is None:
            return

        def _compile_friendly(hidden_states, residual, weight, eps, stream_id):
            if _is_fake_or_meta_tensor(hidden_states) or _is_fake_or_meta_tensor(
                residual
            ):
                return None
            return custom_op(
                hidden_states, residual, weight, float(eps), int(stream_id)
            )

        try:
            setattr(_compile_friendly, "_qwen35_graphfx_wrapped", True)
            setattr(_compile_friendly, "_qwen35_graphfx_original", original)
        except Exception:
            pass
        try:
            rtp_llm_ops.fused_add_rmsnorm = _compile_friendly
        except Exception as exc:
            logger.warning(
                "QWEN35 GraphFX: failed to attach compile-friendly fused_add_rmsnorm: %s",
                exc,
            )
            return

        # Refresh module-level imports of rtp_llm_ops that already cached the
        # original attribute as a free-standing function, so models loaded
        # before this hook ran still see the wrapper.
        for module_name, module in list(sys.modules.items()):
            if getattr(module, "fused_add_rmsnorm_pybind", None) is original:
                try:
                    setattr(module, "fused_add_rmsnorm_pybind", _compile_friendly)
                except Exception:
                    pass

        _dynamo_allow_in_graph(custom_op)
        _dynamo_allow_in_graph(_compile_friendly)
        _FUSED_ADD_RMSNORM_WRAPPER_INSTALLED = True
        logger.info(
            "QWEN35 GraphFX installed compile-friendly fused_add_rmsnorm wrapper"
        )
    except Exception as exc:
        logger.warning(
            "QWEN35 GraphFX: compile-friendly fused_add_rmsnorm install failed: %s", exc
        )


# -------------------------------------------- compile-disable for non-compute helpers


def _torch_compile_disable(fn: Callable, *, reason: str) -> Callable:
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
        logger.warning("QWEN35 compile-disable failed for %s: %s", fn, exc)
        return fn


def _install_compile_disable_on_attr(
    owner: object, attr_name: str, qualname: str
) -> bool:
    marker = f"_qwen35_graphfx_disable_{attr_name}"
    if getattr(owner, marker, False):
        return False
    fn = getattr(owner, attr_name, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, "_qwen35_graphfx_compile_disabled", False):
        setattr(owner, marker, True)
        return False
    disabled = _torch_compile_disable(
        fn, reason=f"QWEN35 GraphFX keeps {qualname} eager"
    )
    try:
        setattr(disabled, "_qwen35_graphfx_compile_disabled", True)
        setattr(disabled, "_qwen35_graphfx_compile_disable_original", fn)
    except Exception:
        pass
    setattr(owner, attr_name, disabled)
    setattr(owner, marker, True)
    logger.info("QWEN35 GraphFX compile-disabled helper: %s", qualname)
    return True


def _install_compile_disable_targets(
    module_name: str, attr_names: tuple[str, ...]
) -> int:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.info(
            "QWEN35 GraphFX skip compile-disable module %s: %s", module_name, exc
        )
        return 0
    installed = 0
    for attr_name in attr_names:
        installed += int(
            _install_compile_disable_on_attr(
                module, attr_name, f"{module_name}.{attr_name}"
            )
        )
    return installed


def _install_compile_disable_methods(
    module_name: str, class_name: str, method_names: tuple[str, ...]
) -> int:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.info(
            "QWEN35 GraphFX skip compile-disable module %s: %s", module_name, exc
        )
        return 0
    cls = getattr(module, class_name, None)
    if cls is None:
        return 0
    installed = 0
    for method_name in method_names:
        installed += int(
            _install_compile_disable_on_attr(
                cls, method_name, f"{module_name}.{class_name}.{method_name}"
            )
        )
    return installed


def install_qwen35_graphfx_compile_disable_wrappers() -> None:
    """Keep non-compute Python helpers eager so they don't pollute FX graphs.

    These helpers build attention metadata, write KV cache, dump debug
    tensors, or do other host-side work that does not need to be captured by
    the fusion passes.  Letting Dynamo trace them only creates extra graph
    breaks and dynamic-shape guards.
    """
    global _COMPILE_DISABLE_INSTALLED
    if _COMPILE_DISABLE_INSTALLED:
        return
    if not env_flag("QWEN35_GRAPHFX_DISABLE_NON_COMPUTE_HELPERS", True):
        return

    installed = 0
    # ---- Debug / instrumentation ------------------------------------------------
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py._fuse_tensor_dump", ("dump",)
    )
    # ---- Block-map / metadata builders -----------------------------------------
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.model_desc.block_map",
        ("select_block_map_for_layer",),
    )
    installed += _install_compile_disable_targets(
        "rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d",
        ("prepare_causal_conv1d_metadata",),
    )
    # ---- Model-level forward helpers (Qwen3-Next) ------------------------------
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.model_desc.qwen3_next",
        "Qwen3NextModel",
        (
            "_build_cp_linear_attn_metadata",
            "prepare_fmha_impl",
            "_may_init_multimodal",
        ),
    )
    # ---- Model-level forward helpers (Generic MoE) -----------------------------
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.model_desc.generic_moe",
        "GenericMoeModel",
        ("prepare_fmha_impl",),
    )
    # ---- Layer-internal helpers that build/write KV cache ----------------------
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.base.common.kvcache_store",
        "WriteCacheStoreOp",
        ("forward",),
    )
    # ---- FMHA implementations (host-side prepare) ------------------------------
    for module_name, class_name in (
        ("rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa", "XQAImpl"),
        ("rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa", "XQADecodeImpl"),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen",
            "FlashInferTRTLLMPrefillImpl",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen",
            "FlashInferTRTLLMSpecDecodeImpl",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen",
            "FlashInferTRTLLMDecodeImpl",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt",
            "TRTMHAImpl",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt",
            "TRTPagedMHAImpl",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha",
            "PyFlashinferPrefillImplBase",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise",
            "HeadWisePrefillImpl",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise_fp8",
            "HeadWiseFP8PrefillImpl",
        ),
        (
            "rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer",
            "CPFlashInferImpl",
        ),
    ):
        installed += _install_compile_disable_methods(
            module_name,
            class_name,
            ("prepare", "prepare_cuda_graph"),
        )
    # ---- MlaAttention helpers (DSV3.2 / GLM5 path) -----------------------------
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.hybrid.mla_attention",
        "MlaAttention",
        ("_run_sparse_indexer",),
    )
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl",
        "FlashMLASparseImpl",
        ("prepare", "prepare_cuda_graph", "_apply_input_bmm", "_apply_output_bmm"),
    )
    # ---- Indexer host-side prepare ---------------------------------------------
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.hybrid.indexer",
        "Indexer",
        ("_prefill_cp_enabled", "_is_sparse_prefill_cp"),
    )
    _COMPILE_DISABLE_INSTALLED = True
    logger.info("QWEN35 GraphFX compile-disabled %d non-compute helpers", installed)


# ------------------------------------------------------------ allow_in_graph


def _dynamo_allow_in_graph(fn: Callable) -> None:
    try:
        import torch

        allow_in_graph = getattr(
            getattr(torch, "compiler", None), "allow_in_graph", None
        )
        if allow_in_graph is None:
            import torch._dynamo as dynamo

            allow_in_graph = dynamo.allow_in_graph
        allow_in_graph(fn)
    except Exception as exc:
        logger.warning("QWEN35 allow_in_graph failed for %s: %s", fn, exc)


def _allow_qwen35_fusion_candidates_in_graph() -> None:
    """Keep the high-level fused kernels visible to FX as single nodes."""
    global _DYNAMO_LEAFS_REGISTERED
    if _DYNAMO_LEAFS_REGISTERED:
        return
    try:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.modules.fuse_kernel_fx.add_rmsnorm_runtime import (
            qwen35_fused_add_rmsnorm_fp8_quant_from_provenance,
            qwen35_fused_add_rmsnorm_producer_token,
        )
        from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
            fused_add_rmsnorm_fp8_quant,
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
        )

        for fn in (
            sgl_per_token_group_quant_fp8,
            fused_add_rmsnorm_fp8_quant,
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
            qwen35_fused_add_rmsnorm_producer_token,
            qwen35_fused_add_rmsnorm_fp8_quant_from_provenance,
        ):
            try:
                _dynamo_allow_in_graph(fn)
            except Exception:
                continue
    except Exception as exc:
        logger.warning("QWEN35 allow_in_graph candidates registration failed: %s", exc)
    _DYNAMO_LEAFS_REGISTERED = True


# ------------------------------------------------------------ Dynamo config


def _configure_dynamo_for_qwen35_graphfx() -> None:
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
        dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), 128)
        dynamo.config.accumulated_cache_size_limit = max(
            int(dynamo.config.accumulated_cache_size_limit), 512
        )
        try:
            import torch.fx.experimental._config as fx_config

            fx_config.use_duck_shape = False
        except Exception:
            pass
        logger.info("QWEN35 GraphFX Dynamo configured (dynamic=True, cache=128/512)")
    except Exception as exc:
        logger.warning("QWEN35 GraphFX Dynamo config failed: %s", exc)
    _DYNAMO_CONFIGURED = True


# ------------------------------------------------------------ DCE helpers


_MUTATING_LEAF_TARGETS = {
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_mutating",
    "rtp_llm_qwen35::fused_add_rmsnorm_mutating",
}


def _target_name(target: object) -> str:
    return getattr(target, "__name__", str(target))


def eliminate_dead_code_preserving_qwen35_side_effects(gm: object) -> None:
    """Run FX DCE without dropping opaque mutating CUDA calls."""
    graph = getattr(gm, "graph", None)
    if graph is None:
        return

    def is_impure_node(node: object) -> bool:
        target_name = _target_name(getattr(node, "target", None))
        lowered = target_name.lower()
        if "fused_add_rmsnorm" in lowered and "fp8" not in lowered:
            return True
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


# ------------------------------------------------------------ pass driver


def record_qwen35_fusion_miss(pass_name: str, reason: str) -> None:
    if env_flag("QWEN35_GRAPHFX_MISS_LOG") or env_flag("QWEN35_GRAPHFX_COMPILE_STATS"):
        _PASS_MISS_COUNTS[(pass_name, reason)] += 1


def record_qwen35_fusion_hit(pass_name: str, count: int = 1) -> None:
    if env_flag("QWEN35_GRAPHFX_MISS_LOG") or env_flag("QWEN35_GRAPHFX_COMPILE_STATS"):
        _PASS_HIT_COUNTS[pass_name] += count


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


def apply_registered_qwen35_fusions(gm, *, return_changed: bool = False):
    ensure_qwen35_fusions_registered()
    debug = env_flag("QWEN35_FUSION_REGISTRY_DEBUG")
    if debug:
        logger.info(
            "QWEN35 fusion passes: %s",
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
            continue
        before = _graph_signature(out)
        out = item.pass_fn(out)
        changed = changed or _graph_signature(out) != before
    if return_changed:
        return out, changed
    return out


def _log_compile_summary() -> None:
    if not _GRAPH_COMPILE_LABEL_COUNTS and not _PASS_HIT_COUNTS:
        return
    try:
        logger.info(
            "QWEN35 GraphFX compile summary: labels=%s pass_hits=%s pass_misses=%s",
            dict(sorted(_GRAPH_COMPILE_LABEL_COUNTS.items())),
            {str(k): v for k, v in sorted(_PASS_HIT_COUNTS.items())},
            {str(k): v for k, v in sorted(_PASS_MISS_COUNTS.items())},
        )
    except Exception:
        pass


def _register_compile_summary_logger() -> None:
    global _COMPILE_SUMMARY_REGISTERED
    if _COMPILE_SUMMARY_REGISTERED:
        return
    atexit.register(_log_compile_summary)
    _COMPILE_SUMMARY_REGISTERED = True


def qwen35_fusion_backend(gm, example_inputs, *, label: str = "unknown"):
    setup_qwen35_fusion_file_logger()
    _register_compile_summary_logger()
    _GRAPH_COMPILE_LABEL_COUNTS[label] += 1
    if env_flag("QWEN35_GRAPHFX_COMPILE_STATS"):
        targets = [_target_name(node.target) for node in gm.graph.nodes]
        logger.info("QWEN35 GraphFX compile: label=%s nodes=%d", label, len(targets))
    fused, changed = apply_registered_qwen35_fusions(gm, return_changed=True)
    if not changed and env_flag("QWEN35_GRAPHFX_FALLBACK_UNFUSED", True):
        logger.info("QWEN35 GraphFX no-op for label=%s, keep original graph", label)
        return gm.forward
    fused.recompile()
    return fused.forward


def compile_with_qwen35_fusions(fn, *, label: str | None = None, **compile_kwargs):
    import torch

    ensure_qwen35_fusions_registered()
    _configure_dynamo_for_qwen35_graphfx()
    if env_flag("QWEN35_GRAPHFX_FALLBACK_UNFUSED", True):
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except Exception:
            pass
    compile_label = (
        label
        or getattr(fn, "__qualname__", None)
        or getattr(fn, "__name__", None)
        or "unknown"
    )

    def _backend(gm, example_inputs):
        return qwen35_fusion_backend(gm, example_inputs, label=compile_label)

    kwargs = {"backend": _backend, "fullgraph": False}
    kwargs.update(compile_kwargs)
    compiled = torch.compile(fn, **kwargs)

    def _compiled_with_reference_refresh(*args, **inner_kwargs):
        # Late-imported model modules may still hold the original
        # ``rtp_llm_ops.fused_add_rmsnorm`` attribute.  Reinstall the wrapper
        # before each compiled call so they pick up the FX-friendly version.
        install_compile_friendly_fused_add_rmsnorm_wrapper()
        return compiled(*args, **inner_kwargs)

    return _compiled_with_reference_refresh


def any_qwen35_fusion_enabled() -> bool:
    ensure_qwen35_fusions_registered()
    return any(env_flag(item.env_gate) for item in _PASSES)
