from __future__ import annotations

import logging
import os
from typing import Any

from rtp_llm.models_py.modules.dsv4.fusions.fusion_registry import (
    compile_with_dsv4_fusions,
    ensure_dsv4_fusions_registered,
    env_flag,
    registered_dsv4_fusion_passes,
    setup_dsv4_fusion_file_logger,
)

logger = logging.getLogger(__name__)

_CALLABLE_MARKER = "_dsv4_graphfx_compiled"

_DYNAMIC_DIM0_FIELDS = (
    "input_ids",
    "input_hiddens",
    "combo_tokens",
    "position_ids",
    "input_lengths",
    "input_lengths_d",
    "prefix_lengths",
    "prefix_lengths_d",
    "sequence_lengths",
    "sequence_lengths_plus_1_d",
    "padding_offset",
    "cu_kv_seqlens",
    "decode_cu_seqlens_d",
    "kv_cache_block_id_host",
    "kv_cache_block_id_device",
    "kv_cache_offset",
)

_MAYBE_DYNAMIC_DIM0_FIELDS = ("cu_seqlens",)

_DYNAMIC_DIM01_FIELDS = (
    "kv_cache_block_id_host",
    "kv_cache_block_id_device",
    "kv_cache_kernel_block_id_host",
    "kv_cache_kernel_block_id_device",
    "pool_block_tables",
    "swa_pool_bt",
    "swa_global_slots",
    "hca_cmp_global_slots",
    "pool_write_slot_mappings",
    "slot_mapping_swa",
    "slot_mapping_compressed",
)


def graphfx_fusion_enabled() -> bool:
    return env_flag("DSV4_GRAPHFX_FUSION")


def _debug_enabled() -> bool:
    return env_flag("DSV4_FUSION_REGISTRY_DEBUG")


def _compile_scope() -> str:
    """GraphFX compile boundary selector.

    The production boundary is the framework-visible model ``forward``.
    Metadata/debug helpers that should stay eager are excluded by the common
    compile-disable installer in ``fusion_registry`` instead of adding
    decode- or prefill-specific compiled entry points here.
    """
    return os.environ.get("DSV4_GRAPHFX_COMPILE_SCOPE", "forward").strip().lower() or "forward"


def _mark_dynamic_dim0(value: Any) -> None:
    """Tell Dynamo that request-sized axes are dynamic before compilation.

    The fused-op FX passes must ignore M / batch / sequence length while still
    checking fixed hidden dimensions and CUDA wrapper layout constraints.  This
    pre-call hook marks only leading request/token axes on framework input
    tensors.  It does not mark weight or hidden dimensions dynamic.
    """
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            torch._dynamo.mark_dynamic(value, 0)
    except Exception:
        # mark_dynamic is a best-effort cache hint.  Runtime validation remains
        # in the CUDA wrappers, so a failed mark should not change semantics.
        return


def _mark_dynamic_dim(value: Any, dim: int) -> None:
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.dim() > dim:
            torch._dynamo.mark_dynamic(value, dim)
    except Exception:
        return


def _mark_request_tensor_dims(value: Any) -> None:
    """Mark request axes on tensors passed to the compiled forward."""
    _mark_dynamic_dim(value, 0)
    _mark_dynamic_dim(value, 1)


def _mark_dynamic_dim01(value: Any) -> None:
    """Mark batch/token and block-table width axes as dynamic.

    Block-table shaped metadata is typically ``[B, num_blocks]``.  ``B`` and
    ``num_blocks`` both vary with request batch/sequence length, while hidden
    dimensions live elsewhere and remain static for the fusion contracts.
    """
    _mark_dynamic_dim(value, 0)
    _mark_dynamic_dim(value, 1)


def _maybe_mark_dynamic_dim0(value: Any) -> None:
    """Best-effort dynamic hint for metadata axes that may be specialized.

    ``cu_seqlens`` is ``[B + 1]``.  Some DSV4 prefill/decode control paths
    intentionally branch on ``cu_seqlens.numel()`` and specialize B==1 to a
    constant length of 2.  Force-marking that axis dynamic can violate Dynamo
    constraints, while ``maybe_mark_dynamic`` allows dynamic reuse when the
    captured graph does not require a constant.
    """
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            maybe_mark_dynamic = getattr(torch._dynamo, "maybe_mark_dynamic", None)
            if maybe_mark_dynamic is not None:
                maybe_mark_dynamic(value, 0)
    except Exception:
        return


def _mark_dynamic_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    candidates = list(args)
    if "inputs" in kwargs:
        candidates.append(kwargs["inputs"])
    for candidate in candidates:
        _mark_request_tensor_dims(candidate)
        for field in _DYNAMIC_DIM0_FIELDS:
            _mark_dynamic_dim0(getattr(candidate, field, None))
        for field in _DYNAMIC_DIM01_FIELDS:
            value = getattr(candidate, field, None)
            if isinstance(value, dict):
                for item in value.values():
                    _mark_dynamic_dim01(item)
            else:
                _mark_dynamic_dim01(value)
        for field in _MAYBE_DYNAMIC_DIM0_FIELDS:
            _maybe_mark_dynamic_dim0(getattr(candidate, field, None))
        attn = getattr(candidate, "attention_inputs", None)
        if attn is not None:
            for field in _DYNAMIC_DIM0_FIELDS:
                _mark_dynamic_dim0(getattr(attn, field, None))
            for field in _DYNAMIC_DIM01_FIELDS:
                value = getattr(attn, field, None)
                if isinstance(value, dict):
                    for item in value.values():
                        _mark_dynamic_dim01(item)
                else:
                    _mark_dynamic_dim01(value)
            for field in _MAYBE_DYNAMIC_DIM0_FIELDS:
                _maybe_mark_dynamic_dim0(getattr(attn, field, None))
            by_group = getattr(attn, "kv_cache_kernel_block_id_device_by_group", None)
            if by_group is not None:
                try:
                    for item in by_group:
                        _mark_dynamic_dim01(item)
                except TypeError:
                    pass
            by_group_host = getattr(attn, "kv_cache_kernel_block_id_host_by_group", None)
            if by_group_host is not None:
                try:
                    for item in by_group_host:
                        _mark_dynamic_dim01(item)
                except TypeError:
                    pass

        # Only request-sized buffers are marked dynamic; fixed hidden
        # dimensions and per-layer weights are intentionally left static.
        for field in (
            "start_pos",
            "position_ids",
            "slot_mapping_swa",
            "slot_mapping_compressed",
            "cache_seqlens_i32",
            "pool_block_tables",
            "swa_global_slots",
            "hca_cmp_global_slots",
            "swa_pool_bt",
            "pool_write_slot_mappings",
        ):
            value = getattr(candidate, field, None)
            if isinstance(value, dict):
                for item in value.values():
                    _mark_dynamic_dim01(item)
            else:
                _mark_dynamic_dim0(value)


def _compile_callable(owner: Any, attr_name: str, label: str) -> bool:
    owner_marker = f"_dsv4_graphfx_{attr_name}_compiled"
    if getattr(owner, owner_marker, False):
        return False
    fn = getattr(owner, attr_name, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, _CALLABLE_MARKER, False):
        return False

    compiled = compile_with_dsv4_fusions(
        fn,
        label=label,
        dynamic=True,
        fullgraph=False,
    )

    def wrapped(*args: Any, **kwargs: Any):
        _mark_dynamic_inputs(args, kwargs)
        return compiled(*args, **kwargs)

    try:
        setattr(wrapped, _CALLABLE_MARKER, True)
        setattr(wrapped, "_dsv4_graphfx_label", label)
        setattr(wrapped, "_dsv4_graphfx_compiled_callable", compiled)
    except Exception:
        pass
    setattr(owner, f"_dsv4_graphfx_original_{attr_name}", fn)
    setattr(owner, attr_name, wrapped)
    setattr(owner, owner_marker, True)
    logger.info("DSV4 GraphFX compile installed: %s dynamic=True", label)
    return True


def maybe_install_dsv4_graphfx_fusions(py_model: Any) -> bool:
    """Install the DSV4 FX backend on the framework-visible model forward.

    The RTP-LLM framework dispatches DSV4 decode/prefill through
    ``DeepSeekV4Model.forward`` and free-function helpers, not through the
    standalone ``V4Transformer.forward_decode`` method.  Wrapping this single
    public entry point lets Dynamo capture the normal execution path without
    reaching into transformer internals or replacing decode helper modules.
    Dynamic batch/sequence/M dimensions are left to Dynamo guards and runtime
    CUDA wrapper validation.
    """
    if not graphfx_fusion_enabled():
        return False
    setup_dsv4_fusion_file_logger()
    ensure_dsv4_fusions_registered()

    if py_model is None:
        return False
    forward = getattr(py_model, "forward", None)
    if forward is None or not callable(forward):
        logger.warning(
            "DSV4 GraphFX fusion enabled but py_model has no callable forward: %s",
            type(py_model).__name__,
        )
        return False

    scope = _compile_scope()
    if scope != "forward":
        logger.warning(
            "DSV4 GraphFX unsupported compile scope %s; falling back to py_model.forward",
            scope,
        )
    installed = int(
        _compile_callable(py_model, "forward", f"{type(py_model).__name__}.forward")
    )

    if _debug_enabled():
        logger.info(
            "DSV4 GraphFX injector status: installed=%d compile_boundary=%s passes=%s",
            installed,
            "py_model.forward" if installed else "<none>",
            [
                {
                    "name": item.name,
                    "env_gate": item.env_gate,
                }
                for item in registered_dsv4_fusion_passes()
            ],
        )
    else:
        logger.info(
            "DSV4 GraphFX injector installed %d compiled callables",
            installed,
        )
    return installed > 0
