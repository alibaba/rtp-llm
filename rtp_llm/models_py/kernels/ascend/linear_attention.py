"""Ascend Qwen3.5 chunked linear-attention operators.

The public functions in this module intentionally match the existing RTP-LLM
Triton interfaces and route inputs to ``flash-linear-attention-npu``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

_CHUNK_SIZE = 64


def _get_ascendc_ops():
    try:
        from fla_npu.ops import ascendc
    except ImportError as exc:  # pragma: no cover - depends on the NPU image
        raise RuntimeError(
            "Qwen3.5 on Ascend requires flash-linear-attention-npu. "
            "Build and install its SoC-specific wheel, then verify "
            "`from fla_npu.ops import ascendc`."
        ) from exc
    return ascendc


def _get_npu_l2norm():
    try:
        from fla.ops.triton.triton_core.l2norm import l2norm_fwd as npu_l2norm_fwd
    except ImportError:
        return None
    return npu_l2norm_fwd


def _to_int_list(value: Optional[torch.Tensor]) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().tolist()]
    return [int(item) for item in value]


def _canonical_cu_seqlens(
    cu_seqlens: Optional[torch.Tensor], batch: int, seqlen: int
) -> Optional[list[int]]:
    values = _to_int_list(cu_seqlens)
    if values is None:
        return None
    if batch != 1:
        raise ValueError("cu_seqlens requires flattened batch size 1")
    if len(values) < 2 or values[0] != 0:
        raise ValueError("cu_seqlens must start with zero and contain an end offset")
    if values[-1] != seqlen:
        raise ValueError("cu_seqlens end offset must equal the flattened token count")
    return values


def _prepare_chunk_indices(
    cu_seqlens: Optional[list[int]], chunk_size: int
) -> Optional[list[int]]:
    if cu_seqlens is None:
        return None
    indices: list[int] = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        if end < start:
            raise ValueError("cu_seqlens must be non-decreasing")
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            indices.extend((seq_idx, chunk_idx))
    return indices


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _cumsum_block_size(chunk_size: int) -> int:
    # Mirrors the FLA-NPU tiling-side BLOCK_T computation used by the
    # standalone GPU/NPU golden tests in example/ascendc_npu.
    return _next_power_of_two((1 << 17) // chunk_size)


def _dtype_name(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return "same"
    if dtype in (torch.float, torch.float32):
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported output dtype for FLA-NPU: {dtype}")


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """L2-normalize the last dimension while preserving RTP's return type."""
    # The migration guide selects FLA's Triton-for-Ascend implementation.  A
    # few deployment images intentionally ship only the AscendC wheel, so keep
    # a mathematically equivalent torch_npu path for that packaging variant.
    npu_l2norm_fwd = _get_npu_l2norm()
    if npu_l2norm_fwd is None:
        target_dtype = x.dtype if output_dtype is None else output_dtype
        inv_norm = torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + eps)
        return (x.float() * inv_norm).to(target_dtype)

    result = npu_l2norm_fwd(x.contiguous(), eps=eps)
    # FLA-NPU returns (y, rstd), whereas RTP-LLM's historical wrapper returns y.
    y = result[0] if isinstance(result, (tuple, list)) else result
    return y.to(output_dtype) if output_dtype is not None else y


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    **kwargs,
) -> torch.Tensor:
    if g.ndim != 3:
        raise ValueError("FLA-NPU chunk_local_cumsum currently supports scalar gates")
    g_head_first = g.contiguous() if head_first else g.transpose(1, 2).contiguous()
    batch, _, seqlen = g_head_first.shape
    cu = _canonical_cu_seqlens(cu_seqlens, batch, seqlen)
    block_indices = _prepare_chunk_indices(cu, _cumsum_block_size(chunk_size))
    out = _get_ascendc_ops().npu_chunk_local_cumsum(
        g=g_head_first.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu,
        chunk_indices_out=block_indices,
        reverse=reverse,
        scale=1.0 if scale is None else float(scale),
        head_first=True,
        output_dtype=_dtype_name(output_dtype),
    )
    if output_dtype is None:
        out = out.to(g.dtype)
    return out if head_first else out.transpose(1, 2).contiguous()


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if k.ndim != 4 or beta.ndim != 3:
        raise ValueError("k and beta must use BSND/BST layouts")
    batch, seqlen, key_heads, _ = k.shape
    value_heads = beta.shape[-1]
    if value_heads % key_heads:
        raise ValueError("value heads must be divisible by key heads")
    head_ratio = value_heads // key_heads
    k_h = k.transpose(1, 2).contiguous()
    beta_h = beta.transpose(1, 2).contiguous().float()
    if g_cumsum is None:
        g_h = torch.zeros_like(beta_h, dtype=torch.float32)
    else:
        g_h = g_cumsum.transpose(1, 2).contiguous().float()
    cu = _canonical_cu_seqlens(cu_seqlens, batch, seqlen)
    chunk_indices = _prepare_chunk_indices(cu, chunk_size)
    ascendc = _get_ascendc_ops()

    # The AscendC KKT operator emits one result per key head, while RTP's
    # GVA semantics require a distinct matrix for every value head because
    # each one has its own gate and beta.  Evaluate each offset within a GVA
    # group and interleave the results instead of repeating offset zero.
    grouped_outputs = []
    for head_offset in range(head_ratio):
        grouped_outputs.append(
            ascendc.npu_chunk_scaled_dot_kkt(
                k=k_h,
                g=g_h[:, head_offset::head_ratio].contiguous(),
                beta=beta_h[:, head_offset::head_ratio].contiguous(),
                cu_seqlens=cu,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
            )
        )
    if head_ratio == 1:
        out = grouped_outputs[0]
    else:
        out = torch.stack(grouped_outputs, dim=2).flatten(1, 2)
    return out.transpose(1, 2).contiguous().to(output_dtype)


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    batch, seqlen, _, chunk_size = A.shape
    cu = _canonical_cu_seqlens(cu_seqlens, batch, seqlen)
    # The standalone seq2047 comparison establishes FP16 as substantially
    # more accurate than BF16 for this inverse on the currently supported SoC.
    x = A.to(torch.float16)
    ascendc = _get_ascendc_ops()
    if cu is None:
        out = ascendc.npu_solve_tri(
            x=x,
            cu_seqlens=None,
            chunk_indices=None,
            layout="bsnd",
        )
    else:
        # varlen：tnd 布局（[T_total, H, CS]），kernel 依 cu_seqlens 按序列起点切块，
        # chunk_indices 供 tiling 统计 (seq_idx, chunk_idx) tile 分核。
        out = ascendc.npu_solve_tri(
            x=x.reshape(seqlen, A.shape[2], chunk_size),
            cu_seqlens=cu,
            chunk_indices=_prepare_chunk_indices(cu, chunk_size),
            layout="tnd",
        ).reshape(A.shape)
    return out.to(output_dtype)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen = k.shape[:2]
    chunk_size = A.shape[-1]
    cu = _canonical_cu_seqlens(cu_seqlens, batch, seqlen)
    chunk_indices = _prepare_chunk_indices(cu, chunk_size)
    compute_dtype = torch.bfloat16
    w, u = _get_ascendc_ops().npu_recompute_w_u_fwd(
        k=k.transpose(1, 2).contiguous().to(compute_dtype),
        v=v.transpose(1, 2).contiguous().to(compute_dtype),
        beta=beta.transpose(1, 2).contiguous().float(),
        A=A.transpose(1, 2).contiguous().to(compute_dtype),
        chunk_size=chunk_size,
        g=g_cumsum.transpose(1, 2).contiguous().float(),
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    return (
        w.transpose(1, 2).contiguous().to(k.dtype),
        u.transpose(1, 2).contiguous().to(v.dtype),
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    if gk is not None:
        raise NotImplementedError("FLA-NPU fwd_h does not support gk")
    batch, seqlen, _, key_dim = k.shape
    value_heads, value_dim = u.shape[-2:]
    cu = _canonical_cu_seqlens(cu_seqlens, batch, seqlen)
    chunk_indices = _prepare_chunk_indices(cu, chunk_size)
    # States are K-first ([N, HV, DK, DV]) throughout this interface:
    # load_initial_state_from_block_map transposes the V-first paged cache into
    # this layout, chunk_fwd_o consumes h in it, and store_ssm_state_to_block_map
    # transposes back on the way out.  Keep the kernel default (state_v_first
    # False) so all four agree -- DK == DV would otherwise hide a transpose.
    if initial_state is None:
        initial_state = torch.zeros(
            batch if cu is None else len(cu) - 1,
            value_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=k.device,
        )
    g_input = (
        torch.zeros(batch, seqlen, value_heads, dtype=torch.float32, device=k.device)
        if g is None
        else g
    )
    result = _get_ascendc_ops().npu_chunk_gated_delta_rule_fwd_h(
        k=k.transpose(1, 2).contiguous().to(torch.bfloat16),
        w=w.transpose(1, 2).contiguous().to(torch.bfloat16),
        u=u.transpose(1, 2).contiguous().to(torch.bfloat16),
        g=g_input.transpose(1, 2).contiguous().float(),
        initial_state=initial_state.float().contiguous(),
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_fwd_h returned an invalid result"
        )
    h, v_new, final_state = result
    h = h.transpose(1, 2).contiguous()
    v_new_out = (
        v_new.transpose(1, 2).contiguous().to(u.dtype) if save_new_value else None
    )
    return h, v_new_out, final_state


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    batch, seqlen = q.shape[:2]
    value_heads = v.shape[-2]
    effective_chunk_size = min(chunk_size, max(16, _next_power_of_two(seqlen)))
    cu = _canonical_cu_seqlens(cu_seqlens, batch, seqlen)
    chunk_indices = _prepare_chunk_indices(cu, effective_chunk_size)
    g_input = (
        torch.zeros(batch, seqlen, value_heads, dtype=torch.float32, device=q.device)
        if g is None
        else g
    )
    compute_dtype = torch.bfloat16
    out = _get_ascendc_ops().npu_chunk_fwd_o(
        q=q.transpose(1, 2).contiguous().to(compute_dtype),
        k=k.transpose(1, 2).contiguous().to(compute_dtype),
        v=v.transpose(1, 2).contiguous().to(compute_dtype),
        h=h.transpose(1, 2).contiguous().to(compute_dtype),
        scale=float(q.shape[-1] ** -0.5 if scale is None else scale),
        g=g_input.transpose(1, 2).contiguous().float(),
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        chunk_size=effective_chunk_size,
    )
    return out.transpose(1, 2).contiguous().to(v.dtype)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Run the complete Qwen3.5 prefill Gated-DeltaNet pipeline."""
    if head_first:
        raise NotImplementedError(
            "head_first=True is not supported by the RTP Qwen3.5 interface"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.dtype == torch.float32:
        raise TypeError("chunk_gated_delta_rule requires float16 or bfloat16 inputs")
    if beta.ndim != 3:
        raise ValueError("beta must have shape [B,T,HV]")
    cu_values = _to_int_list(cu_seqlens)
    if cu_values is not None:
        if q.shape[0] != 1:
            raise ValueError("cu_seqlens requires flattened batch size 1")
        if initial_state is not None and initial_state.shape[0] != len(cu_values) - 1:
            raise ValueError(
                "initial_state count must match the sequences in cu_seqlens"
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        # AscendC chunk kernels expect already normalized q/k.
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    g_cumsum = chunk_local_cumsum(
        g,
        chunk_size=_CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k,
        beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_size=_CHUNK_SIZE,
        output_dtype=torch.float32,
    )
    A = solve_tril(A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=_CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=_CHUNK_SIZE,
    )
    # The AscendC fwd_h kernel emits h/final_state in bfloat16, while the
    # block-map state store (and CUDA FLA semantics) requires float32 states.
    final_state_out = (
        final_state.float() if final_state is not None else None
    )
    return o.to(q.dtype), h.float(), final_state_out


__all__ = [
    "chunk_fwd_o",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt_fwd",
    "l2norm_fwd",
    "recompute_w_u_fwd",
    "solve_tril",
]
