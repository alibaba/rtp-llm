"""Runtime helpers for cross-graph add+RMSNorm+FP8 quant fusion.

Dynamo frequently splits the producer (``fused_add_rmsnorm`` mutating call)
and the consumer (``sgl_per_token_group_quant_fp8`` inside an FP8 linear) into
two separate FX GraphModules. The same-graph rewrite in
``add_rmsnorm_fp8_quant_pass`` cannot help in that case.

The producer custom_op runs ``fused_add_rmsnorm_fp8_quant_inplace`` which:
  1. Mutates ``residual`` (residual += hidden_states)
  2. Mutates ``hidden_states`` (overwritten with normed bf16 — zero-copy via
     passing hidden_states as bf16_out_ptr to the dual-output Triton kernel)
  3. Produces ``(fp8, scale)`` and stashes them in the quant provenance
     registry (``remember_quant``)

The consumer looks up the registry; when it hits, the standalone quant
kernel is eliminated.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

try:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
except Exception:  # noqa: BLE001 - keep import-safe in CPU/no-triton dev shells
    sgl_per_token_group_quant_fp8 = None  # type: ignore[assignment]

from rtp_llm.models_py.modules.fuse_kernel_fx.quant_provenance import (
    lookup_quant,
    remember_quant,
)

try:
    import triton

    from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
        MAX_INREG_H,
        _baseline_add_rmsnorm_fp8_quant_with_bf16_output,
        _fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel,
        _select_num_warps,
        create_per_token_group_quant_fp8_output_scale,
    )

    _TRITON_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRITON_AVAILABLE = False


def _fused_add_rmsnorm_fp8_quant_inplace(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 128,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place add+RMSNorm+FP8 quant: writes normed bf16 directly into hidden_states.

    Same semantics as fused_add_rmsnorm_fp8_quant_with_bf16_output but avoids
    the extra memcpy by passing hidden_states as the bf16 output buffer.
    """
    T, H = hidden_states.shape
    block_n = triton.next_power_of_2(H)
    if block_n > MAX_INREG_H:
        bf16_out, fp8, scale = _baseline_add_rmsnorm_fp8_quant_with_bf16_output(
            hidden_states, residual, weight, eps, group_size, scale_ue8m0
        )
        hidden_states.copy_(bf16_out)
        return fp8, scale

    fp8_out = torch.empty(
        (T, H), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    scale_out = create_per_token_group_quant_fp8_output_scale(
        x_shape=(T, H),
        device=hidden_states.device,
        group_size=group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    if T == 0:
        return fp8_out, scale_out

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    grid = (T,)

    _fused_add_rmsnorm_fp8_quant_dual_output_singlepass_kernel[grid](
        hidden_states,
        residual,
        weight,
        hidden_states,  # bf16_out = hidden_states (zero-copy in-place)
        fp8_out,
        scale_out,
        H,
        eps,
        fp8_max,
        fp8_min,
        hidden_states.stride(0),
        residual.stride(0),
        hidden_states.stride(0),  # bf16_out stride = hidden_states stride
        fp8_out.stride(0),
        scale_out.stride(0),
        scale_out.stride(1),
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        SCALE_UE8M0=scale_ue8m0,
        num_warps=_select_num_warps(H),
    )
    return fp8_out, scale_out


logger = logging.getLogger(__name__)

_PRODUCER_CUSTOM_OP = None


def _is_fake_or_meta(t: torch.Tensor) -> bool:
    if t.is_meta:
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        return isinstance(t, FakeTensor)
    except Exception:
        return False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _debug_enabled() -> bool:
    return _env_flag("GRAPHFX_FUSED_ADD_RMSNORM_DEBUG")


def build_producer_custom_op():
    """Build the producer custom_op with proper mutates_args.

    Returns a custom_op that has ``mutates_args=("hidden_states", "residual")``
    so Dynamo preserves mutation tracking across subgraph boundaries.
    """
    global _PRODUCER_CUSTOM_OP
    if _PRODUCER_CUSTOM_OP is not None:
        return _PRODUCER_CUSTOM_OP

    def _impl(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int,
        scale_ue8m0: bool,
    ) -> None:
        if not _TRITON_AVAILABLE:
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            stream_id = torch.cuda.current_stream().cuda_stream
            rtp_llm_ops.fused_add_rmsnorm(
                hidden_states, residual, weight, float(eps), int(stream_id)
            )
            return None
        fp8, scale = _fused_add_rmsnorm_fp8_quant_inplace(
            hidden_states,
            residual,
            weight,
            eps=eps,
            group_size=group_size,
            scale_ue8m0=scale_ue8m0,
        )
        remember_quant(hidden_states, fp8, scale)
        if _debug_enabled():
            logger.info(
                "GraphFX producer: fused add+rmsnorm+fp8_quant shape=%s",
                tuple(hidden_states.shape),
            )
        return None

    def _fake(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        group_size: int,
        scale_ue8m0: bool,
    ) -> None:
        return None

    annotations = {
        "hidden_states": torch.Tensor,
        "residual": torch.Tensor,
        "weight": torch.Tensor,
        "eps": float,
        "group_size": int,
        "scale_ue8m0": bool,
        "return": None,
    }
    _impl.__annotations__ = annotations
    _fake.__annotations__ = annotations
    op = torch.library.custom_op(
        "rtp_llm_graphfx::fused_add_rmsnorm_fp8_quant_producer",
        _impl,
        mutates_args=("hidden_states", "residual"),
    )
    op.register_fake(_fake)
    _PRODUCER_CUSTOM_OP = getattr(op, "_opoverload", op)
    return _PRODUCER_CUSTOM_OP


def graphfx_fused_add_rmsnorm_fp8_quant_from_provenance(
    y: torch.Tensor,
    *,
    fallback_y: Optional[torch.Tensor] = None,
    group_size: int = 128,
    eps: float = 1e-4,
    column_major_scales: bool = True,
    scale_tma_aligned: bool = True,
    scale_ue8m0: bool = True,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-graph consumer-side replacement for ``sgl_per_token_group_quant_fp8``.

    Looks up the producer-side provenance for ``y``:
      * Precompute hit -> return the precomputed (fp8, scale).
      * Miss -> fall back to ``sgl_per_token_group_quant_fp8``.
    """
    if not fuse_silu_and_mul:
        expected_scale_dtype = torch.int32 if scale_ue8m0 else torch.float32
        cached = lookup_quant(y)
        if cached is not None and cached[1].dtype == expected_scale_dtype:
            if _debug_enabled():
                logger.info("GraphFX quant provenance hit shape=%s", tuple(y.shape))
            return cached

    if sgl_per_token_group_quant_fp8 is None:
        raise RuntimeError(
            "sgl_per_token_group_quant_fp8 unavailable: triton/CUDA build required"
        )
    quant_input = fallback_y if fallback_y is not None else y
    if _debug_enabled():
        logger.info(
            "GraphFX consumer FALLBACK to standalone quant shape=%s fallback_y=%s",
            tuple(y.shape),
            fallback_y is not None,
        )
    return sgl_per_token_group_quant_fp8(
        quant_input,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )
