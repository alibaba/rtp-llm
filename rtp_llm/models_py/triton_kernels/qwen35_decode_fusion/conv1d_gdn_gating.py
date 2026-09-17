"""Fuse decode ``causal_conv1d_update`` (silu) with ``fused_gdn_gating``.

Group D / P1. The two ops have **no data dependency**: gating reads ``a``/``b``
only. Fusion is one launch so silu(conv(x)) and (g, beta) do not need two
kernels. BA GEMM and the 31µs GDN recurrent SSM stay outside this file.

Triton bodies are copied from:

- ``triton_kernels/causal_conv1d/causal_conv1d.py`` ``_causal_conv1d_update_kernel``
- ``triton_kernels/fla/gdn_gating.py`` ``fused_gdn_gating_kernel``

Do not rewrite silu / softplus / sigmoid. Reference path is the two existing
Python wrappers. v1 gate: decode ``seq=1``, width=4, silu, no bias, no
``cache_seqlens``, paged ``block_map`` + ``sequence_lengths`` like ``_conv1d``.
Target-verify (seq>1) is unsupported.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
    causal_conv1d_update,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d.op import cal_block_idx
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.utils import is_amd
from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    is_decode_fusion_enabled,
)

# Match causal_conv1d_update decode tile.
_BLOCK_N = 256
_DECODE_WIDTH = 4
_DECODE_SEQLEN = 1


def _fusion_enabled() -> bool:
    return is_decode_fusion_enabled()


def _as_activation(activation: Union[bool, str, None]) -> Optional[str]:
    if isinstance(activation, bool):
        return "silu" if activation else None
    return activation


def _is_decode_x(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    if x.dim() == 2:
        return x.size(0) >= 0 and x.size(1) > 0
    if x.dim() == 3:
        return x.size(2) == _DECODE_SEQLEN and x.size(1) > 0
    return False


def _tensor_ok(t: torch.Tensor, dtypes: tuple[torch.dtype, ...]) -> bool:
    return t.is_cuda and t.dtype in dtypes


def _flags_supported(
    bias: Optional[torch.Tensor],
    activation: Union[bool, str, None],
    cache_seqlens: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    block_map: Optional[torch.Tensor],
    sequence_lengths: Optional[torch.Tensor],
) -> bool:
    if not _fusion_enabled():
        return False
    if bias is not None or cache_seqlens is not None or query_start_loc is not None:
        return False
    if _as_activation(activation) not in ("silu", "swish"):
        return False
    return block_map is not None and sequence_lengths is not None


def _conv_tensors_supported(
    x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor
) -> bool:
    if not _is_decode_x(x) or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if weight.dim() != 2 or weight.size(1) != _DECODE_WIDTH:
        return False
    dim = x.size(1)
    if weight.size(0) != dim or not weight.is_cuda:
        return False
    if conv_state.dim() != 3 or conv_state.size(1) != dim:
        return False
    if conv_state.size(2) < _DECODE_WIDTH - 1 or not conv_state.is_cuda:
        return False
    return True


def _gating_tensors_supported(
    x: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> bool:
    if a.dim() != 2 or b.dim() != 2:
        return False
    batch = x.size(0)
    if a.size(0) != batch or b.size(0) != batch:
        return False
    heads = a.size(1)
    if heads != b.size(1) or heads != A_log.numel() or heads != dt_bias.numel():
        return False
    if a.stride(-1) != 1 or b.stride(-1) != 1 or a.stride(0) != b.stride(0):
        return False
    dtypes = (torch.bfloat16, torch.float16)
    if not _tensor_ok(a, dtypes) or not _tensor_ok(b, dtypes):
        return False
    return A_log.is_cuda and dt_bias.is_cuda


def _paged_meta_supported(
    batch: int,
    block_map: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> bool:
    if batch == 0:
        return True
    if block_map.dim() != 2 or block_map.size(0) != batch or block_map.size(1) < 1:
        return False
    if sequence_lengths.dim() != 1 or sequence_lengths.size(0) != batch:
        return False
    return block_map.is_cuda and sequence_lengths.is_cuda


def is_supported(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = "silu",
    cache_seqlens: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
) -> bool:
    """Host-side gate. False → call site must use the old two-kernel path.

    Supported (production ``_conv1d`` decode, not target-verify):

    - ``RTP_QWEN35_DECODE_FUSION=1`` (or true/on/yes)
    - ``x``: ``[T, dim]`` or ``[T, dim, 1]`` CUDA BF16/FP16
    - ``weight``: ``[dim, 4]``
    - ``activation`` silu/swish, ``bias is None``, ``cache_seqlens is None``
    - paged ``block_map`` ``[T, max_blocks]`` and ``sequence_lengths`` ``[T]``
    - ``a``, ``b``: ``[T, heads]`` last-dim contiguous
    - T=0 is supported and returns empty outputs (no launch)

    Unsupported: seq!=1, width!=4, missing block_map, target-verify, CPU.
    """
    if not _flags_supported(
        bias, activation, cache_seqlens, query_start_loc, block_map, sequence_lengths
    ):
        return False
    if not _conv_tensors_supported(x, conv_state, weight):
        return False
    if not _gating_tensors_supported(x, A_log, a, b, dt_bias):
        return False
    assert block_map is not None and sequence_lengths is not None
    return _paged_meta_supported(x.size(0), block_map, sequence_lengths)


@triton.jit
def _fused_conv1d_update_gdn_gating_kernel(
    x_ptr,
    w_ptr,
    conv_state_ptr,
    block_map_ptr,
    stride_block_map: tl.int64,
    max_block_size: tl.int32,
    sequence_lengths_ptr,
    o_ptr,
    g_ptr,
    beta_output_ptr,
    A_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_heads: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_ab: tl.constexpr,
    gdn_beta: tl.constexpr,
    gdn_threshold: tl.constexpr,
    NP2_STATELEN_TOTAL: tl.constexpr,
    NP2_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
):
    """Decode specialization of causal_conv1d_update + fused_gdn_gating.

    Conv / silu / sliding-window / paged block_map: copied from
    ``_causal_conv1d_update_kernel`` (width=4, silu, no bias).
    Gating: copied from ``fused_gdn_gating_kernel``; run on feature tile 0.
    """
    # --- causal_conv1d_update (decode) ---
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    x_offset = idx_seq * stride_x_seq
    o_offset = idx_seq * stride_o_seq

    sequence_length = tl.load(sequence_lengths_ptr + idx_seq).to(tl.int32)
    read_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
    read_block_offset = tl.minimum(read_block_offset, max_block_size - 1)
    read_block_id = tl.load(
        block_map_ptr + idx_seq * stride_block_map + read_block_offset
    ).to(tl.int64)

    conv_states_base = (
        conv_state_ptr
        + (read_block_id * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim
    prior_tokens = conv_states_base
    col0 = tl.load(prior_tokens, mask_w, 0.0)
    col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask_w, 0.0)
    col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask_w, 0.0)

    idx_tokens = tl.arange(0, NP2_STATELEN_TOTAL)
    conv_state_ptrs_source = (
        conv_state_ptr
        + (read_block_id * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + 1) * stride_conv_state_tok)[:, None]
    )
    mask = ((idx_tokens + 1) < state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - 1
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)
    for idx in tl.range(seqlen):
        write_block_offset = cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + idx
        write_ok = write_block_offset < max_block_size
        safe_write_offset = tl.minimum(write_block_offset, max_block_size - 1)
        write_block_id = tl.load(
            block_map_ptr + idx_seq * stride_block_map + safe_write_offset,
        ).to(tl.int64)
        if write_ok and (write_block_id != -1):
            conv_state_base = (
                conv_state_ptr
                + (write_block_id * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )
            idx_tokens_offset = idx_tokens - idx
            conv_state_ptrs_target = (
                conv_state_base + (idx_tokens_offset * stride_conv_state_tok)[:, None]
            )
            store_mask = (
                (idx_tokens_offset >= 0)[:, None]
                & (idx_tokens_offset < state_len)[:, None]
                & (idx_feats < dim)[None, :]
            )
            tl.store(conv_state_ptrs_target, new_conv_state, store_mask)

    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = w_ptr + (idx_feats * stride_w_dim)
    w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
    w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    for idx_token in tl.range(seqlen):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if j == 1:
                matrix_w = w_col1
                matrix_x = col1
            elif j == 2:
                matrix_w = w_col2
                matrix_x = col2
            elif j == 3:
                matrix_w = w_col3
                x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            acc += matrix_x * matrix_w

        # silu: identical to _causal_conv1d_update_kernel
        acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d)

    # --- fused_gdn_gating (independent of conv out); tile 0 only ---
    if tl.program_id(1) == 0:
        head_off = tl.arange(0, NP2_HEADS)
        mask_h = head_off < num_heads
        off = idx_seq * num_heads + head_off
        ba_off = idx_seq * stride_ab + head_off
        blk_A_log = tl.load(A_log_ptr + head_off, mask=mask_h)
        blk_a = tl.load(a_ptr + ba_off, mask=mask_h)
        blk_b = tl.load(b_ptr + ba_off, mask=mask_h)
        blk_bias = tl.load(dt_bias_ptr + head_off, mask=mask_h)
        gate_x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
        softplus_x = tl.where(
            gdn_beta * gate_x <= gdn_threshold,
            (1 / gdn_beta) * tl.log(1 + tl.exp(gdn_beta * gate_x)),
            gate_x,
        )
        blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
        tl.store(g_ptr + off, blk_g.to(g_ptr.dtype.element_ty), mask=mask_h)
        blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
        tl.store(
            beta_output_ptr + off,
            blk_beta_output.to(beta_output_ptr.dtype.element_ty),
            mask=mask_h,
        )


def _empty_outputs(
    batch: int,
    dim: int,
    seqlen: int,
    num_heads: int,
    x: torch.Tensor,
    b: torch.Tensor,
    unsqueeze: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty((batch, seqlen, dim), device=x.device, dtype=x.dtype).transpose(
        1, 2
    )
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=x.device)
    beta_dtype = torch.float32 if is_amd else b.dtype
    beta = torch.empty(1, batch, num_heads, dtype=beta_dtype, device=b.device)
    if unsqueeze:
        out = out.squeeze(-1)
    return out, g, beta


def fused_conv1d_update_gdn_gating(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    block_map: torch.Tensor,
    seq_size_per_block: int = 1,
    sequence_lengths: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = "silu",
    cache_seqlens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    gdn_beta: float = 1.0,
    gdn_threshold: float = 20.0,
    out: Optional[torch.Tensor] = None,
    g_out: Optional[torch.Tensor] = None,
    beta_out: Optional[torch.Tensor] = None,
    block_n: int = _BLOCK_N,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One launch: inplace conv_state update + silu(conv(x)) + GDN gating.

    Args:
        x: ``[T, dim]`` or ``[T, dim, 1]`` (``_conv1d`` after reshape/transpose).
        conv_state: ``[blocks, dim, state_len]`` as passed to
            ``causal_conv1d_update`` (cache view after ``transpose(1, 2)``).
        weight: ``[dim, 4]``.
        A_log, a, b, dt_bias: same as ``fused_gdn_gating``; ``a``/``b`` are
            ``[T, heads]`` last-dim contiguous.

    Returns:
        mixed_qkv_out: same layout as ``causal_conv1d_update`` (``[T, dim, 1]``
            or ``[T, dim]`` if ``x`` was 2D).
        g: fp32 ``[1, T, heads]``.
        beta: bf16 ``[1, T, heads]`` on CUDA (fp32 on ROCm).
    """
    if not is_supported(
        x,
        conv_state,
        weight,
        A_log,
        a,
        b,
        dt_bias,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        block_map=block_map,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
    ):
        raise ValueError("unsupported inputs for fused_conv1d_update_gdn_gating")

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    num_heads = a.size(1)
    state_len = _DECODE_WIDTH - 1
    np2_statelen_total = triton.next_power_of_2(state_len - 1 + seqlen)
    np2_heads = triton.next_power_of_2(max(num_heads, 1))

    if batch == 0:
        out, g, beta_output = _empty_outputs(
            batch, dim, seqlen, num_heads, x, b, unsqueeze
        )
        return out.to(original_x_dtype), g, beta_output

    if out is None:
        out = torch.empty(
            (batch, seqlen, dim), device=x.device, dtype=x.dtype
        ).transpose(1, 2)
    if g_out is None:
        g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    else:
        g = g_out
    beta_dtype = torch.float32 if is_amd else b.dtype
    if beta_out is None:
        beta_output = torch.empty(1, batch, num_heads, dtype=beta_dtype, device=b.device)
    else:
        beta_output = beta_out
    if num_warps is None:
        num_warps = 8
    if num_stages is None:
        num_stages = 2

    stride_w_dim, stride_w_width = weight.stride()
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_block_map = block_map.stride(0)
    max_block_size = block_map.shape[1]
    stride_ab = a.stride(0)

    def grid(meta):
        return (batch, triton.cdiv(dim, meta["BLOCK_N"]))

    _fused_conv1d_update_gdn_gating_kernel[grid](
        x,
        weight,
        conv_state,
        block_map,
        stride_block_map,
        max_block_size,
        sequence_lengths,
        out,
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        batch,
        dim,
        seqlen,
        state_len,
        num_heads,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        stride_ab,
        gdn_beta,
        gdn_threshold,
        np2_statelen_total,
        np2_heads,
        block_n,
        seq_size_per_block,
        _DECODE_WIDTH,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype), g, beta_output


def maybe_fused_conv1d_update_gdn_gating(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = "silu",
    cache_seqlens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    gdn_beta: float = 1.0,
    gdn_threshold: float = 20.0,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """CUDA-graph-safe wrapper. Returns None when ``is_supported`` is False."""
    if not is_supported(
        x,
        conv_state,
        weight,
        A_log,
        a,
        b,
        dt_bias,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        block_map=block_map,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
    ):
        return None
    return fused_conv1d_update_gdn_gating(
        x,
        conv_state,
        weight,
        A_log,
        a,
        b,
        dt_bias,
        block_map=block_map,
        seq_size_per_block=seq_size_per_block,
        sequence_lengths=sequence_lengths,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        query_start_loc=query_start_loc,
        gdn_beta=gdn_beta,
        gdn_threshold=gdn_threshold,
    )


def conv1d_update_and_gdn_gating_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    block_map: torch.Tensor,
    seq_size_per_block: int = 1,
    sequence_lengths: torch.Tensor,
    activation: Union[bool, str, None] = "silu",
    gdn_beta: float = 1.0,
    gdn_threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Old path: existing ``causal_conv1d_update`` then ``fused_gdn_gating``.

    Call-site fallback when ``maybe_*`` returns None. Mutates ``conv_state``
    inplace with the same semantics as production ``_conv1d``.
    """
    mixed_qkv_out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=None,
        activation=activation,
        cache_seqlens=None,
        block_map=block_map,
        seq_size_per_block=seq_size_per_block,
        sequence_lengths=sequence_lengths,
    )
    g, beta = fused_gdn_gating(
        A_log, a, b, dt_bias, beta=gdn_beta, threshold=gdn_threshold
    )
    return mixed_qkv_out, g, beta


__all__ = [
    "is_supported",
    "fused_conv1d_update_gdn_gating",
    "maybe_fused_conv1d_update_gdn_gating",
    "conv1d_update_and_gdn_gating_ref",
]
