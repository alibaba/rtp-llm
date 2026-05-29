"""GraphFX install entry point for Qwen3.5 / GLM5 / Qwen3-MoE / DSV3.2 models.

``maybe_install_graphfx_fusions(py_model)`` is meant to be called once
from ``BaseModel.load()`` after ``_create_python_model()`` when the env flag
``ENABLE_GRAPHFX_FUSION`` is set.  It wraps ``py_model.forward`` with a
``torch.compile``-d callable that uses the GraphFX fusion registry as a
backend so the registered FX passes can rewrite RMSResNorm/quant/gate
patterns into fused-kernel calls without modifying eager model code.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rtp_llm.models_py.modules.fuse_kernel_fx.fusion_registry import (
    compile_with_graphfx_fusions,
    ensure_graphfx_fusions_registered,
    env_flag,
    registered_graphfx_fusion_passes,
    setup_graphfx_fusion_file_logger,
)

logger = logging.getLogger(__name__)

_CALLABLE_MARKER = "_graphfx_compiled"


# Fields on PyModelInputs / PyAttentionInputs whose dim-0 is the per-request
# axis.  Marked dynamic so Dynamo doesn't recompile per token count.
_DYNAMIC_DIM0_FIELDS = (
    "input_ids",
    "input_hiddens",
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

# Fields that may specialize on B==1; use maybe_mark_dynamic instead of
# force-marking.
_MAYBE_DYNAMIC_DIM0_FIELDS = ("cu_seqlens",)

# Block-table style fields whose 0/1 dims are batch / num_blocks.
_DYNAMIC_DIM01_FIELDS = (
    "kv_cache_block_id_host",
    "kv_cache_block_id_device",
    "kv_cache_kernel_block_id_host",
    "kv_cache_kernel_block_id_device",
    "pool_block_tables",
    "pool_write_slot_mappings",
)


def graphfx_fusion_enabled() -> bool:
    return env_flag("ENABLE_GRAPHFX_FUSION")


def _debug_enabled() -> bool:
    return env_flag("GRAPHFX_FUSION_REGISTRY_DEBUG")


def _mark_dynamic_dim0(value: Any) -> None:
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            torch._dynamo.mark_dynamic(value, 0)
    except Exception:
        return


def _mark_dynamic_dim(value: Any, dim: int) -> None:
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.dim() > dim:
            torch._dynamo.mark_dynamic(value, dim)
    except Exception:
        return


def _maybe_mark_dynamic_dim0(value: Any) -> None:
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            maybe_mark_dynamic = getattr(torch._dynamo, "maybe_mark_dynamic", None)
            if maybe_mark_dynamic is not None:
                maybe_mark_dynamic(value, 0)
    except Exception:
        return


def _mark_dynamic_dim01(value: Any) -> None:
    _mark_dynamic_dim(value, 0)
    _mark_dynamic_dim(value, 1)


def _mark_request_tensor_dims(value: Any) -> None:
    _mark_dynamic_dim(value, 0)
    _mark_dynamic_dim(value, 1)


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


def _compile_callable(owner: Any, attr_name: str, label: str) -> bool:
    owner_marker = f"_graphfx_{attr_name}_compiled"
    if getattr(owner, owner_marker, False):
        return False
    fn = getattr(owner, attr_name, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, _CALLABLE_MARKER, False):
        return False

    compiled = compile_with_graphfx_fusions(
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
        setattr(wrapped, "_graphfx_label", label)
        setattr(wrapped, "_graphfx_compiled_callable", compiled)
    except Exception:
        pass
    setattr(owner, f"_graphfx_original_{attr_name}", fn)
    setattr(owner, attr_name, wrapped)
    setattr(owner, owner_marker, True)
    logger.info("GraphFX compile installed: %s dynamic=True", label)
    return True


def _install_subboundary_for_indexer(py_model: Any) -> int:
    """Walk submodules, install compile boundaries on Indexer methods that
    contain fuse-kernel call sites Dynamo cannot otherwise see.

    Wrapped methods (per Indexer instance):
      * ``_get_logits_head_gate``  — pure fp32 chain, FX rewrites to
        ``fused_logits_head_gate``.
      * ``_fused_forward_decode``  — calls ``fused_qk_rope_quant`` and
        ``indexer_k_quant_and_cache`` (both wrapped as ``torch.library
        .custom_op`` / ``allow_in_graph``).  After this sub-boundary the
        fused kernel calls appear as opaque leaf nodes in their own FX
        subgraph instead of being hidden inside the compile-disabled
        ``_run_sparse_indexer`` subtree.
    """
    try:
        from rtp_llm.models_py.modules.hybrid.indexer import Indexer
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    seen_ids: set[int] = set()
    modules = getattr(py_model, "modules", None)
    if not callable(modules):
        return 0
    for module in modules():
        if not isinstance(module, Indexer):
            continue
        if id(module) in seen_ids:
            continue
        seen_ids.add(id(module))
        layer_idx = getattr(module, "layer_idx", "?")
        if _compile_callable(
            module,
            "_get_logits_head_gate",
            f"Indexer.layer{layer_idx}._get_logits_head_gate",
        ):
            count += 1
        if _compile_callable(
            module,
            "_fused_forward_decode",
            f"Indexer.layer{layer_idx}._fused_forward_decode",
        ):
            count += 1
    return count


def maybe_install_graphfx_fusions(py_model: Any) -> bool:
    """Install the GraphFX backend on the framework-visible model forward.

    Compatible with ``Qwen3NextModel`` / ``Qwen3NextMTPModel`` /
    ``GenericMoeModel`` (all expose ``forward(inputs, fmha_impl)``).  Returns
    True iff the wrap actually happened.
    """
    if not graphfx_fusion_enabled():
        return False
    setup_graphfx_fusion_file_logger()
    ensure_graphfx_fusions_registered()

    if py_model is None:
        return False
    forward = getattr(py_model, "forward", None)
    if forward is None or not callable(forward):
        logger.warning(
            "GraphFX enabled but py_model has no callable forward: %s",
            type(py_model).__name__,
        )
        return False

    label = f"{type(py_model).__name__}.forward"
    installed = int(_compile_callable(py_model, "forward", label))

    # Also wrap Indexer._get_logits_head_gate as its own compile boundary so
    # the indexer_logits_head_gate_fx pass can match its unfused fp32 chain.
    # The enclosing _run_sparse_indexer is in the compile-disable list (the
    # indexer body uses pybind ops Dynamo can't trace), but this method is
    # pure PyTorch and traces cleanly into a small subgraph.
    installed += _install_subboundary_for_indexer(py_model)

    if _debug_enabled():
        logger.info(
            "GraphFX injector: installed=%d compile_boundary=%s passes=%s",
            installed,
            label if installed else "<none>",
            [
                {"name": item.name, "env_gate": item.env_gate}
                for item in registered_graphfx_fusion_passes()
            ],
        )
    else:
        logger.info("GraphFX injector installed=%d", installed)

    return installed > 0


def _prewarm_triton_kernels(py_model: Any) -> None:
    """JIT-compile Triton kernels that will be called during CUDA graph capture."""
    import torch

    try:
        from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
        )
    except Exception:
        return

    if fused_add_rmsnorm_fp8_quant_with_bf16_output is None:
        return

    layers = getattr(py_model, "layers", None)
    if not layers:
        return
    first_layer = layers[0] if hasattr(layers, "__getitem__") else next(iter(layers))
    norm = getattr(first_layer, "input_layernorm", None)
    if norm is None:
        return
    weight = getattr(norm, "weight", None)
    if weight is None:
        weight = getattr(norm, "gamma", None)
    if weight is None:
        return
    H = weight.shape[0]
    device = weight.device

    for T in [1, 2, 4, 8, 16, 32]:
        try:
            dummy_h = torch.randn(T, H, dtype=torch.bfloat16, device=device)
            dummy_r = torch.randn(T, H, dtype=torch.bfloat16, device=device)
            dummy_w = torch.ones(H, dtype=torch.bfloat16, device=device)
            fused_add_rmsnorm_fp8_quant_with_bf16_output(
                dummy_h,
                dummy_r,
                dummy_w,
                eps=1e-6,
                group_size=128,
                scale_ue8m0=False,
            )
        except Exception:
            break
    logger.info("GraphFX prewarm: fused_add_rmsnorm_fp8_quant_with_bf16_output H=%d", H)
