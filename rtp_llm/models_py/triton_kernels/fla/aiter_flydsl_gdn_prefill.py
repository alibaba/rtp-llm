"""AITER FlyDSL GDN prefill adapter for RTP-LLM.

The adapter fixes the optimized pipeline to FlyDSL K1-K4, FlyDSL K5, and
RTP's Triton K6. Model routing and serving configuration intentionally live in
the separate end-to-end integration layer.
"""

import functools
import logging
from collections.abc import Callable
from typing import NamedTuple

import torch

from rtp_llm.models_py.triton_kernels.fla.chunk_o import (
    chunk_fwd_o_head_major_vk,
)
from rtp_llm.models_py.triton_kernels.fla.l2norm import fused_l2norm_qk
from rtp_llm.models_py.triton_kernels.fla.utils import env_flag

CHUNK_SIZE = 64
_LOGGER = logging.getLogger(__name__)


class _AiterFlydslGdnPrefillOps(NamedTuple):
    """Complete pinned-AITER API required by the prefill adapter."""

    prepare_supported: Callable[..., bool]
    prepare: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    chunk_state: Callable[
        ..., tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ]
    build_metadata: Callable[..., object]


@functools.cache
def _get_aiter_flydsl_gdn_prefill_ops() -> _AiterFlydslGdnPrefillOps | None:
    """Resolve the complete prefill API or disable the optimized path."""
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
            gdn_prepare_flydsl_supported,
            gdn_prepare_fwd_flydsl,
        )
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )
    except (AttributeError, ImportError) as error:
        _LOGGER.warning(
            "AITER FlyDSL GDN prefill API is unavailable; falling back: %s",
            error,
        )
        return None
    return _AiterFlydslGdnPrefillOps(
        prepare_supported=gdn_prepare_flydsl_supported,
        prepare=gdn_prepare_fwd_flydsl,
        chunk_state=chunk_gated_delta_rule_fwd_h_flydsl_opt,
        build_metadata=build_gated_delta_rule_prefill_metadata,
    )


@functools.cache
def _is_aiter_flydsl_gdn_prefill_disabled() -> bool:
    """Return the process-start emergency rollback setting.

    Dispatch is automatic by default. Restart the serving process after
    changing this setting because its value is cached.
    """
    return env_flag("DISABLE_AITER_FLYDSL_GDN_PREFILL")


def is_aiter_flydsl_gdn_prefill_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> bool:
    """Return whether the fixed FlyDSL K1-K5 pipeline supports the inputs."""
    if _is_aiter_flydsl_gdn_prefill_disabled():
        return False
    if torch.version.hip is None or q.device.type != "cuda":
        return False
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        return False
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        return False
    if g.dtype not in (torch.bfloat16, torch.float32) or beta.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    batch, tokens, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if batch != 1 or v.shape[:2] != (batch, tokens):
        return False
    if key_dim != 128 or value_dim != 128:
        return False
    if key_heads < 1 or value_heads % key_heads != 0:
        return False
    expected_gate_shape = (batch, tokens, value_heads)
    if g.shape != expected_gate_shape or beta.shape != expected_gate_shape:
        return False
    ops = _get_aiter_flydsl_gdn_prefill_ops()
    if ops is None:
        return False
    return ops.prepare_supported(k, v, BT=CHUNK_SIZE)


@torch.compiler.disable
def build_aiter_flydsl_gdn_prefill_metadata(
    sequence_lengths: tuple[int, ...], cu_seqlens: torch.Tensor
) -> object:
    """Build the reusable AITER schedule shared by FlyDSL K1-K5."""
    ops = _get_aiter_flydsl_gdn_prefill_ops()
    if ops is None:
        raise RuntimeError("AITER FlyDSL GDN prefill API is unavailable")
    return ops.build_metadata(
        sequence_lengths,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )


@torch.compiler.disable
def chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    state_dtype: torch.dtype,
    snapshot_dtype: torch.dtype,
    prefill_metadata: object | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run FlyDSL K1-K5 and RTP K6 while exposing chunk/final states."""
    if not is_aiter_flydsl_gdn_prefill_supported(q, k, v, g, beta):
        raise ValueError(
            "AITER FlyDSL GDN prefill requires ROCm BF16 q/k/v, B=1, "
            "K=V=128, and compatible g/beta"
        )
    if state_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported state dtype: {state_dtype}")
    if snapshot_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported snapshot dtype: {snapshot_dtype}")
    if cu_seqlens is not None and prefill_metadata is None:
        raise ValueError("varlen FlyDSL GDN prefill requires prefill_metadata")

    ops = _get_aiter_flydsl_gdn_prefill_ops()
    if ops is None:
        raise RuntimeError("AITER FlyDSL GDN prefill API became unavailable")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if initial_state is not None:
        initial_state = initial_state.to(state_dtype).contiguous()
    if use_qk_l2norm_in_kernel:
        q, k = fused_l2norm_qk(q, k)
    if not ops.prepare_supported(k, v, BT=CHUNK_SIZE):
        raise ValueError("AITER FlyDSL K1-K4 does not support this device or shape")
    if scale is None:
        scale = k.shape[-1] ** -0.5

    w, u, g_cumsum = ops.prepare(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        BT=CHUNK_SIZE,
        use_exp2=True,
        prefill_metadata=prefill_metadata,
    )
    h, v_new, final_state = ops.chunk_state(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=CHUNK_SIZE,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        snapshot_dtype=snapshot_dtype,
        use_exp2=True,
        g_head_major=True,
        prefill_metadata=prefill_metadata,
    )
    if v_new is None:
        raise RuntimeError("AITER FlyDSL K5 did not return v_new")
    output = chunk_fwd_o_head_major_vk(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )
    return output.to(q.dtype), h, final_state
