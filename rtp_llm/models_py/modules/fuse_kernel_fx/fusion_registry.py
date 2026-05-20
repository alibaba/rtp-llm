"""GraphFX fusion registry for Qwen3.5 / GLM5 / Qwen3-MoE / DSV3.2 fuse kernels.

This module mirrors the DSV4 GraphFX registry pattern but lives under
``rtp_llm/models_py/modules/fuse_kernel_fx/`` and uses the ``GRAPHFX_*`` env
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
phases plug additional passes through ``register_graphfx_fusion_pass``.
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
class GraphFxFusionPass:
    name: str
    pass_fn: Callable[[object], object]
    priority: int
    env_gate: str


_PASSES: List[GraphFxFusionPass] = []
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
_RMSNORM_WRAPPER_INSTALLED = False
_RMSNORM_ORIGINAL = None
_RMSNORM_MUTATING_CUSTOM_OP = None
_TORCH_STREAM_COMPAT_INSTALLED = False
_SILU_AND_MUL_WRAPPER_INSTALLED = False
_SILU_AND_MUL_ORIGINAL = None
_SILU_AND_MUL_MUTATING_CUSTOM_OP = None
_QUANT_FP8_WRAPPER_INSTALLED = False
_QUANT_FP8_ORIGINAL = None
_QUANT_FP8_CUSTOM_OP = None


# ------------------------------------------------------------------ env / log


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _default_log_file() -> str:
    explicit = os.environ.get("GRAPHFX_LOG_FILE")
    if explicit:
        return explicit
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if outputs_dir:
        return os.path.join(outputs_dir, "graphfx_compile.log")
    return f"/tmp/graphfx_compile_{os.getpid()}.log"


def setup_graphfx_fusion_file_logger() -> str:
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
    handler._graphfx_file_handler = True  # type: ignore[attr-defined]
    pkg_logger = logging.getLogger("rtp_llm.models_py.modules.fuse_kernel_fx")
    if not any(
        getattr(existing, "_graphfx_file_handler", False)
        for existing in pkg_logger.handlers
    ):
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.INFO)
    _FILE_LOGGER_INSTALLED = True
    _FILE_LOG_PATH = log_file
    logger.info("GraphFX log file: %s", log_file)
    return log_file


# ------------------------------------------------------------------ registry


def register_graphfx_fusion_pass(
    name: str,
    pass_fn: Callable[[object], object],
    *,
    priority: int = 100,
    env_gate: str,
) -> None:
    if any(item.name == name for item in _PASSES):
        return
    _PASSES.append(
        GraphFxFusionPass(
            name=name, pass_fn=pass_fn, priority=priority, env_gate=env_gate
        )
    )
    _PASSES.sort(key=lambda item: item.priority)


def registered_graphfx_fusion_passes() -> List[GraphFxFusionPass]:
    ensure_graphfx_fusions_registered()
    return list(_PASSES)


def ensure_graphfx_fusions_registered() -> None:
    """Idempotent setup: install pybind wrappers, compile-disables, import passes."""
    global _FUSIONS_IMPORTED
    setup_graphfx_fusion_file_logger()
    install_torch_stream_cuda_stream_compat()
    install_compile_friendly_fused_add_rmsnorm_wrapper()
    install_compile_friendly_rmsnorm_wrapper()
    install_compile_friendly_silu_and_mul_wrapper()
    install_compile_friendly_quant_fp8_wrapper()
    install_graphfx_compile_disable_wrappers()
    if _FUSIONS_IMPORTED:
        return
    importlib.import_module("rtp_llm.models_py.modules.fuse_kernel_fx")
    _allow_graphfx_fusion_candidates_in_graph()
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
    if not env_flag("GRAPHFX_FUSED_ADD_RMSNORM_FP8_QUANT") and not env_flag(
        "ENABLE_GRAPHFX_FUSION"
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
                logger.info("GraphFX installed %s.cuda_stream compat", cls.__name__)
            except Exception as exc:
                logger.info("GraphFX could not patch %s.cuda_stream: %s", cls, exc)
        _TORCH_STREAM_COMPAT_INSTALLED = True
    except Exception as exc:
        logger.warning("GraphFX torch.Stream compat install failed: %s", exc)


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
            "rtp_llm_graphfx::fused_add_rmsnorm_mutating",
            _mutating_impl,
            mutates_args=("hidden_states", "residual"),
        )
        op.register_fake(_mutating_fake)
        _FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP = getattr(op, "_opoverload", op)
        return _FUSED_ADD_RMSNORM_MUTATING_CUSTOM_OP
    except Exception as exc:
        logger.warning("GraphFX mutating fused_add_rmsnorm op build failed: %s", exc)
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
    if not env_flag("ENABLE_GRAPHFX_FUSION"):
        return
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        original = getattr(rtp_llm_ops, "fused_add_rmsnorm", None)
        if original is None:
            logger.info(
                "GraphFX: rtp_llm_ops.fused_add_rmsnorm not present, skip wrapper"
            )
            return
        if getattr(original, "_graphfx_wrapped", False):
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

        _compile_friendly.__name__ = "fused_add_rmsnorm_mutating"
        _compile_friendly.__qualname__ = "fused_add_rmsnorm_mutating"
        try:
            setattr(_compile_friendly, "_graphfx_wrapped", True)
            setattr(_compile_friendly, "_graphfx_original", original)
        except Exception:
            pass
        try:
            rtp_llm_ops.fused_add_rmsnorm = _compile_friendly
        except Exception as exc:
            logger.warning(
                "GraphFX: failed to attach compile-friendly fused_add_rmsnorm: %s",
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
        logger.info("GraphFX installed compile-friendly fused_add_rmsnorm wrapper")
    except Exception as exc:
        logger.warning(
            "GraphFX: compile-friendly fused_add_rmsnorm install failed: %s", exc
        )


# ------------------------------- rmsnorm (non-mutating output) compile-friendly wrapper


def _build_rmsnorm_mutating_custom_op():
    global _RMSNORM_MUTATING_CUSTOM_OP
    if _RMSNORM_MUTATING_CUSTOM_OP is not None:
        return _RMSNORM_MUTATING_CUSTOM_OP
    if _RMSNORM_ORIGINAL is None:
        return None
    try:
        import torch

        def _mutating_impl(output, hidden_states, weight, eps, stream_id):
            _RMSNORM_ORIGINAL(output, hidden_states, weight, eps, stream_id)
            return None

        def _mutating_fake(output, hidden_states, weight, eps, stream_id):
            return None

        annotations = {
            "output": torch.Tensor,
            "hidden_states": torch.Tensor,
            "weight": torch.Tensor,
            "eps": float,
            "stream_id": int,
            "return": None,
        }
        _mutating_impl.__annotations__ = annotations
        _mutating_fake.__annotations__ = annotations
        op = torch.library.custom_op(
            "rtp_llm_graphfx::rmsnorm_mutating",
            _mutating_impl,
            mutates_args=("output",),
        )
        op.register_fake(_mutating_fake)
        _RMSNORM_MUTATING_CUSTOM_OP = getattr(op, "_opoverload", op)
        return _RMSNORM_MUTATING_CUSTOM_OP
    except Exception as exc:
        logger.warning("GraphFX mutating rmsnorm op build failed: %s", exc)
        return None


def install_compile_friendly_rmsnorm_wrapper() -> None:
    """Replace ``rtp_llm_ops.rmsnorm`` with a torch.library custom op.

    The pybind op writes to ``output`` in place — Dynamo cannot trace it with
    FakeTensor inputs. Wrapping it as a custom op with ``mutates_args=("output",)``
    keeps the contract visible in FX while short-circuiting fake-tensor execution.
    """
    global _RMSNORM_WRAPPER_INSTALLED, _RMSNORM_ORIGINAL
    if _RMSNORM_WRAPPER_INSTALLED:
        return
    if not env_flag("ENABLE_GRAPHFX_FUSION"):
        return
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        original = getattr(rtp_llm_ops, "rmsnorm", None)
        if original is None:
            logger.info("GraphFX: rtp_llm_ops.rmsnorm not present, skip wrapper")
            return
        if getattr(original, "_graphfx_wrapped", False):
            _RMSNORM_WRAPPER_INSTALLED = True
            return
        _RMSNORM_ORIGINAL = original
        custom_op = _build_rmsnorm_mutating_custom_op()
        if custom_op is None:
            return

        def _compile_friendly(output, hidden_states, weight, eps, stream_id):
            if _is_fake_or_meta_tensor(hidden_states) or _is_fake_or_meta_tensor(
                output
            ):
                return None
            return custom_op(output, hidden_states, weight, float(eps), int(stream_id))

        _compile_friendly.__name__ = "rmsnorm_mutating"
        _compile_friendly.__qualname__ = "rmsnorm_mutating"
        try:
            setattr(_compile_friendly, "_graphfx_wrapped", True)
            setattr(_compile_friendly, "_graphfx_original", original)
        except Exception:
            pass
        try:
            rtp_llm_ops.rmsnorm = _compile_friendly
        except Exception as exc:
            logger.warning(
                "GraphFX: failed to attach compile-friendly rmsnorm: %s", exc
            )
            return

        for module_name, module in list(sys.modules.items()):
            if getattr(module, "rmsnorm_pybind", None) is original:
                try:
                    setattr(module, "rmsnorm_pybind", _compile_friendly)
                except Exception:
                    pass

        _dynamo_allow_in_graph(custom_op)
        _dynamo_allow_in_graph(_compile_friendly)
        _RMSNORM_WRAPPER_INSTALLED = True
        logger.info("GraphFX installed compile-friendly rmsnorm wrapper")
    except Exception as exc:
        logger.warning("GraphFX: compile-friendly rmsnorm install failed: %s", exc)


# ------------------------------- silu_and_mul compile-friendly wrapper


def _build_silu_and_mul_custom_op():
    global _SILU_AND_MUL_MUTATING_CUSTOM_OP
    if _SILU_AND_MUL_MUTATING_CUSTOM_OP is not None:
        return _SILU_AND_MUL_MUTATING_CUSTOM_OP
    if _SILU_AND_MUL_ORIGINAL is None:
        return None
    try:
        import torch

        def _real_impl(gate_up):
            d = gate_up.shape[-1] // 2
            output = torch.empty(
                gate_up.shape[:-1] + (d,),
                dtype=gate_up.dtype,
                device=gate_up.device,
            )
            stream_id = torch.cuda.current_stream().cuda_stream
            _SILU_AND_MUL_ORIGINAL(output, gate_up, stream_id)
            return output

        def _fake_impl(gate_up):
            d = gate_up.shape[-1] // 2
            return torch.empty(
                gate_up.shape[:-1] + (d,),
                dtype=gate_up.dtype,
                device=gate_up.device,
            )

        annotations = {
            "gate_up": torch.Tensor,
            "return": torch.Tensor,
        }
        _real_impl.__annotations__ = annotations
        _fake_impl.__annotations__ = annotations
        op = torch.library.custom_op(
            "rtp_llm_graphfx::silu_and_mul",
            _real_impl,
            mutates_args=(),
        )
        op.register_fake(_fake_impl)
        _SILU_AND_MUL_MUTATING_CUSTOM_OP = getattr(op, "_opoverload", op)
        return _SILU_AND_MUL_MUTATING_CUSTOM_OP
    except Exception as exc:
        logger.warning("GraphFX silu_and_mul custom op build failed: %s", exc)
        return None


def install_compile_friendly_silu_and_mul_wrapper() -> None:
    """Replace ``FusedSiluAndMul.forward`` with a non-mutating custom op.

    ``FusedSiluAndMul.forward`` allocates an output tensor then calls
    ``rtp_llm_ops.silu_and_mul(output, gate_up, stream_id)`` — a pybind that
    Dynamo cannot trace.  Wrapping the entire alloc+call as a single
    non-mutating custom op (``gate_up -> output``) lets Dynamo emit one
    ``call_function`` node that the ``silu_and_mul_fp8_quant_pass`` can match.
    """
    global _SILU_AND_MUL_WRAPPER_INSTALLED, _SILU_AND_MUL_ORIGINAL
    if _SILU_AND_MUL_WRAPPER_INSTALLED:
        return
    if not env_flag("ENABLE_GRAPHFX_FUSION"):
        return
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        original = getattr(rtp_llm_ops, "silu_and_mul", None)
        if original is None:
            logger.info(
                "GraphFX: rtp_llm_ops.silu_and_mul not present, skip wrapper"
            )
            return
        if getattr(original, "_graphfx_wrapped", False):
            _SILU_AND_MUL_WRAPPER_INSTALLED = True
            return
        _SILU_AND_MUL_ORIGINAL = original
        custom_op = _build_silu_and_mul_custom_op()
        if custom_op is None:
            return

        # Replace FusedSiluAndMul.forward to use the non-mutating custom op
        try:
            from rtp_llm.models_py.modules.base.cuda.activation import FusedSiluAndMul

            _orig_forward = FusedSiluAndMul.forward

            def _custom_op_forward(self, gate_up):
                return custom_op(gate_up)

            _custom_op_forward.__name__ = "forward"
            _custom_op_forward.__qualname__ = "FusedSiluAndMul.forward"
            FusedSiluAndMul.forward = _custom_op_forward
            logger.info(
                "GraphFX replaced FusedSiluAndMul.forward with custom op wrapper"
            )
        except Exception as exc:
            logger.info(
                "GraphFX: could not patch FusedSiluAndMul.forward: %s", exc
            )

        _dynamo_allow_in_graph(custom_op)
        _SILU_AND_MUL_WRAPPER_INSTALLED = True
        logger.info("GraphFX installed compile-friendly silu_and_mul wrapper")
    except Exception as exc:
        logger.warning(
            "GraphFX: compile-friendly silu_and_mul install failed: %s", exc
        )


# ------------------------------- sgl_per_token_group_quant_fp8 compile-friendly wrapper


def _build_quant_fp8_custom_op():
    """Build a torch.library.custom_op for sgl_per_token_group_quant_fp8.

    Unlike allow_in_graph (which Dynamo ignores when the function is already
    imported via closure), a custom_op is GUARANTEED opaque — Dynamo will never
    trace into it and will instead call the registered fake implementation to
    determine output shapes.
    """
    global _QUANT_FP8_CUSTOM_OP
    if _QUANT_FP8_CUSTOM_OP is not None:
        return _QUANT_FP8_CUSTOM_OP
    if _QUANT_FP8_ORIGINAL is None:
        return None
    try:
        from typing import Optional, Tuple

        import torch

        def _real_impl(
            x,
            group_size,
            eps,
            column_major_scales,
            scale_tma_aligned,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m=None,
        ):
            return _QUANT_FP8_ORIGINAL(
                x,
                group_size=group_size,
                eps=eps,
                column_major_scales=column_major_scales,
                scale_tma_aligned=scale_tma_aligned,
                scale_ue8m0=scale_ue8m0,
                fuse_silu_and_mul=fuse_silu_and_mul,
                masked_m=masked_m,
            )

        def _fake_impl(
            x,
            group_size,
            eps,
            column_major_scales,
            scale_tma_aligned,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m=None,
        ):
            out_n = x.shape[-1] // (2 if fuse_silu_and_mul else 1)
            out_shape = (*x.shape[:-1], out_n)
            x_q = torch.empty(out_shape, device=x.device, dtype=torch.float8_e4m3fn)
            if scale_ue8m0:
                *x_batch, x_q_mn, x_q_k = out_shape
                x_s_mn, x_s_k = x_q_mn, x_q_k // 128
                aligned_mn = ((x_s_mn + 3) // 4) * 4
                aligned_k = ((x_s_k + 3) // 4) * 4
                x_s = torch.empty(
                    (*x_batch, aligned_k // 4, aligned_mn),
                    device=x.device,
                    dtype=torch.int,
                ).transpose(-1, -2)[..., :x_s_mn, :]
            elif column_major_scales:
                if scale_tma_aligned:
                    aligned_size = (out_shape[-2] + 3) // 4 * 4
                    x_s = torch.empty(
                        out_shape[:-2] + (out_n // group_size, aligned_size),
                        device=x.device,
                        dtype=torch.float32,
                    ).permute(-1, -2)[: out_shape[-2], :]
                else:
                    x_s = torch.empty(
                        (out_n // group_size,) + out_shape[:-1],
                        device=x.device,
                        dtype=torch.float32,
                    ).permute(-1, -2)
            else:
                x_s = torch.empty(
                    out_shape[:-1] + (out_n // group_size,),
                    device=x.device,
                    dtype=torch.float32,
                )
            return x_q, x_s

        annotations = {
            "x": torch.Tensor,
            "group_size": int,
            "eps": float,
            "column_major_scales": bool,
            "scale_tma_aligned": bool,
            "scale_ue8m0": bool,
            "fuse_silu_and_mul": bool,
            "masked_m": Optional[torch.Tensor],
            "return": Tuple[torch.Tensor, torch.Tensor],
        }
        _real_impl.__annotations__ = annotations
        _fake_impl.__annotations__ = annotations

        op = torch.library.custom_op(
            "rtp_llm_graphfx::sgl_per_token_group_quant_fp8",
            _real_impl,
            mutates_args=(),
        )
        op.register_fake(_fake_impl)
        _QUANT_FP8_CUSTOM_OP = getattr(op, "_opoverload", op)
        return _QUANT_FP8_CUSTOM_OP
    except Exception as exc:
        logger.warning("GraphFX quant_fp8 custom op build failed: %s", exc)
        return None


def install_compile_friendly_quant_fp8_wrapper() -> None:
    """Replace ``sgl_per_token_group_quant_fp8`` with a torch.library custom op.

    The original function contains asserts and calls a pybind op
    (``per_token_group_quant_fp8``).  Dynamo traces INTO these, creating graph
    breaks that split the rmsnorm+quant pattern across subgraphs.  A custom_op
    is guaranteed opaque — Dynamo uses the registered fake implementation for
    shape inference and emits a single ``call_function`` node that FX passes
    can match.
    """
    global _QUANT_FP8_WRAPPER_INSTALLED, _QUANT_FP8_ORIGINAL
    if _QUANT_FP8_WRAPPER_INSTALLED:
        return
    if not env_flag("ENABLE_GRAPHFX_FUSION"):
        return
    try:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        if sgl_per_token_group_quant_fp8 is None:
            return
        if getattr(sgl_per_token_group_quant_fp8, "_graphfx_wrapped", False):
            _QUANT_FP8_WRAPPER_INSTALLED = True
            return
        _QUANT_FP8_ORIGINAL = sgl_per_token_group_quant_fp8
        custom_op = _build_quant_fp8_custom_op()
        if custom_op is None:
            return

        def _compile_friendly(
            x,
            group_size=128,
            eps=1e-10,
            column_major_scales=False,
            scale_tma_aligned=False,
            scale_ue8m0=False,
            fuse_silu_and_mul=False,
            masked_m=None,
        ):
            if _is_fake_or_meta_tensor(x):
                return custom_op(
                    x,
                    group_size,
                    eps,
                    column_major_scales,
                    scale_tma_aligned,
                    scale_ue8m0,
                    fuse_silu_and_mul,
                    masked_m,
                )
            return custom_op(
                x,
                group_size,
                eps,
                column_major_scales,
                scale_tma_aligned,
                scale_ue8m0,
                fuse_silu_and_mul,
                masked_m,
            )

        _compile_friendly.__name__ = "sgl_per_token_group_quant_fp8"
        _compile_friendly.__qualname__ = "sgl_per_token_group_quant_fp8"
        setattr(_compile_friendly, "_graphfx_wrapped", True)
        setattr(
            _compile_friendly,
            "_graphfx_original",
            sgl_per_token_group_quant_fp8,
        )

        # Patch the function in its home module and all importers
        import rtp_llm.models_py.kernels.cuda.fp8_kernel as fp8_pkg
        import rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel as fp8_mod

        fp8_mod.sgl_per_token_group_quant_fp8 = _compile_friendly
        fp8_pkg.sgl_per_token_group_quant_fp8 = _compile_friendly

        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
            if (
                getattr(module, "sgl_per_token_group_quant_fp8", None)
                is _QUANT_FP8_ORIGINAL
            ):
                try:
                    setattr(module, "sgl_per_token_group_quant_fp8", _compile_friendly)
                except Exception:
                    pass

        _dynamo_allow_in_graph(custom_op)
        _dynamo_allow_in_graph(_compile_friendly)
        _QUANT_FP8_WRAPPER_INSTALLED = True
        logger.info(
            "GraphFX installed compile-friendly sgl_per_token_group_quant_fp8 wrapper"
        )
    except Exception as exc:
        logger.warning(
            "GraphFX: compile-friendly sgl_per_token_group_quant_fp8 install failed: %s",
            exc,
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
        logger.warning("GraphFX compile-disable failed for %s: %s", fn, exc)
        return fn


def _install_compile_disable_on_attr(
    owner: object, attr_name: str, qualname: str
) -> bool:
    marker = f"_graphfx_disable_{attr_name}"
    if getattr(owner, marker, False):
        return False
    fn = getattr(owner, attr_name, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, "_graphfx_compile_disabled", False):
        setattr(owner, marker, True)
        return False
    disabled = _torch_compile_disable(fn, reason=f"GraphFX keeps {qualname} eager")
    try:
        setattr(disabled, "_graphfx_compile_disabled", True)
        setattr(disabled, "_graphfx_compile_disable_original", fn)
    except Exception:
        pass
    setattr(owner, attr_name, disabled)
    setattr(owner, marker, True)
    logger.info("GraphFX compile-disabled helper: %s", qualname)
    return True


def _install_compile_disable_targets(
    module_name: str, attr_names: tuple[str, ...]
) -> int:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.info("GraphFX skip compile-disable module %s: %s", module_name, exc)
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
        logger.info("GraphFX skip compile-disable module %s: %s", module_name, exc)
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


def install_graphfx_compile_disable_wrappers() -> None:
    """Keep non-compute Python helpers eager so they don't pollute FX graphs.

    These helpers build attention metadata, write KV cache, dump debug
    tensors, or do other host-side work that does not need to be captured by
    the fusion passes.  Letting Dynamo trace them only creates extra graph
    breaks and dynamic-shape guards.
    """
    global _COMPILE_DISABLE_INSTALLED
    if _COMPILE_DISABLE_INSTALLED:
        return
    if not env_flag("GRAPHFX_DISABLE_NON_COMPUTE_HELPERS", True):
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
    # ---- FP8 DeepGEMM linear (GEMM-only, quant is hoisted to wrapper) ---------
    installed += _install_compile_disable_methods(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear",
        "CudaFp8DeepGEMMLinear",
        ("forward",),
    )
    _COMPILE_DISABLE_INSTALLED = True
    logger.info("GraphFX compile-disabled %d non-compute helpers", installed)


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
        logger.warning("GraphFX allow_in_graph failed for %s: %s", fn, exc)


def _allow_graphfx_fusion_candidates_in_graph() -> None:
    """Keep the high-level fused kernels visible to FX as single nodes."""
    global _DYNAMO_LEAFS_REGISTERED
    if _DYNAMO_LEAFS_REGISTERED:
        return
    try:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.modules.fuse_kernel_fx.add_rmsnorm_runtime import (
            graphfx_fused_add_rmsnorm_fp8_quant_from_provenance,
            graphfx_fused_add_rmsnorm_producer_token,
        )
        from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
            fused_add_rmsnorm_fp8_quant,
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
        )

        for fn in (
            sgl_per_token_group_quant_fp8,
            fused_add_rmsnorm_fp8_quant,
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
            graphfx_fused_add_rmsnorm_producer_token,
            graphfx_fused_add_rmsnorm_fp8_quant_from_provenance,
        ):
            try:
                _dynamo_allow_in_graph(fn)
            except Exception:
                continue
    except Exception as exc:
        logger.warning("GraphFX allow_in_graph candidates registration failed: %s", exc)
    try:
        from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
            fused_strided_rmsnorm,
            fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
        )

        for fn in (
            fused_strided_rmsnorm,
            fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
        ):
            try:
                _dynamo_allow_in_graph(fn)
            except Exception:
                continue
    except Exception as exc:
        logger.warning(
            "GraphFX allow_in_graph strided rmsnorm registration failed: %s", exc
        )
    try:
        from rtp_llm.models_py.modules.fuse_kernel_fx.quant_provenance import (
            remember_quant,
        )
        from rtp_llm.models_py.modules.fuse_kernel_fx.rmsnorm_gated_fp8_quant_pass import (
            graphfx_rmsnorm_gated_producer_token,
        )
        from rtp_llm.models_py.modules.fuse_kernel_fx.sigmoid_mul_fp8_quant_pass import (
            graphfx_sigmoid_mul_producer_token,
        )
        from rtp_llm.models_py.modules.fuse_kernel_fx.silu_and_mul_fp8_quant_pass import (
            graphfx_silu_and_mul_producer_token,
        )
        from rtp_llm.models_py.triton_kernels.common.activation import (
            silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
        )
        from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
            sigmoid_mul_fp8_quant_fwd,
        )
        from rtp_llm.models_py.triton_kernels.common.fused_rmsnorm_gated_fp8_quant import (
            fused_rmsnorm_gated_fp8_quant,
        )

        for fn in (
            graphfx_sigmoid_mul_producer_token,
            graphfx_silu_and_mul_producer_token,
            graphfx_rmsnorm_gated_producer_token,
            sigmoid_mul_fp8_quant_fwd,
            silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
            fused_rmsnorm_gated_fp8_quant,
            remember_quant,
        ):
            try:
                _dynamo_allow_in_graph(fn)
            except Exception:
                continue
    except Exception as exc:
        logger.warning(
            "GraphFX allow_in_graph cross-graph producer registration failed: %s", exc
        )
    _DYNAMO_LEAFS_REGISTERED = True


# ------------------------------------------------------------ Dynamo config


def _configure_dynamo_for_graphfx() -> None:
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
        logger.info("GraphFX Dynamo configured (dynamic=True, cache=128/512)")
    except Exception as exc:
        logger.warning("GraphFX Dynamo config failed: %s", exc)
    _DYNAMO_CONFIGURED = True


# ------------------------------------------------------------ DCE helpers


_MUTATING_LEAF_TARGETS = {
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_mutating",
    "rtp_llm_graphfx::fused_add_rmsnorm_mutating",
}


def _target_name(target: object) -> str:
    return getattr(target, "__name__", str(target))


def eliminate_dead_code_preserving_graphfx_side_effects(gm: object) -> None:
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


def record_graphfx_fusion_miss(pass_name: str, reason: str) -> None:
    if env_flag("GRAPHFX_MISS_LOG") or env_flag("GRAPHFX_COMPILE_STATS"):
        _PASS_MISS_COUNTS[(pass_name, reason)] += 1


def record_graphfx_fusion_hit(pass_name: str, count: int = 1) -> None:
    if env_flag("GRAPHFX_MISS_LOG") or env_flag("GRAPHFX_COMPILE_STATS"):
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


def apply_registered_graphfx_fusions(gm, *, return_changed: bool = False):
    ensure_graphfx_fusions_registered()
    master_enabled = env_flag("ENABLE_GRAPHFX_FUSION")
    debug = env_flag("GRAPHFX_FUSION_REGISTRY_DEBUG")
    if debug:
        logger.info(
            "GraphFX fusion passes: %s",
            [
                {
                    "name": item.name,
                    "env_gate": item.env_gate,
                    "enabled": master_enabled or env_flag(item.env_gate),
                }
                for item in _PASSES
            ],
        )
    out = gm
    changed = False
    for item in _PASSES:
        if master_enabled:
            if env_flag(f"{item.env_gate}_DISABLE"):
                continue
        else:
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
            "GraphFX compile summary: labels=%s pass_hits=%s pass_misses=%s",
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


def graphfx_fusion_backend(gm, example_inputs, *, label: str = "unknown"):
    setup_graphfx_fusion_file_logger()
    _register_compile_summary_logger()
    _GRAPH_COMPILE_LABEL_COUNTS[label] += 1
    if env_flag("GRAPHFX_COMPILE_STATS"):
        targets = [_target_name(node.target) for node in gm.graph.nodes]
        logger.info("GraphFX compile: label=%s nodes=%d", label, len(targets))
    if env_flag("GRAPHFX_DUMP_GRAPHS"):
        _dump_graph_nodes(gm, label)
    fused, changed = apply_registered_graphfx_fusions(gm, return_changed=True)
    if not changed and env_flag("GRAPHFX_FALLBACK_UNFUSED", True):
        logger.info("GraphFX no-op for label=%s, keep original graph", label)
        return gm.forward
    fused.recompile()
    return fused.forward


def _dump_graph_nodes(gm, label: str) -> None:
    count = _GRAPH_COMPILE_LABEL_COUNTS[label]
    lines = [f"=== GraphFX DUMP label={label} #{count} ==="]
    for node in gm.graph.nodes:
        tname = _target_name(node.target)
        args_str = ", ".join(str(a) for a in node.args[:4])
        kwargs_str = ", ".join(f"{k}={v}" for k, v in list(node.kwargs.items())[:3])
        lines.append(
            f"  {node.op:15s} | {tname:50s} | args=({args_str}) | kwargs=({kwargs_str})"
        )
    logger.info("\n".join(lines))


def compile_with_graphfx_fusions(fn, *, label: str | None = None, **compile_kwargs):
    import torch

    ensure_graphfx_fusions_registered()
    _configure_dynamo_for_graphfx()
    if env_flag("GRAPHFX_FALLBACK_UNFUSED", True):
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
        return graphfx_fusion_backend(gm, example_inputs, label=compile_label)

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


def any_graphfx_fusion_enabled() -> bool:
    ensure_graphfx_fusions_registered()
    if env_flag("ENABLE_GRAPHFX_FUSION"):
        return True
    return any(env_flag(item.env_gate) for item in _PASSES)
