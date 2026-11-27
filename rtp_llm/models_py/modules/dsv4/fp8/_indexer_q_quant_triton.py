"""DSv4 Indexer Q per-(token, head) FP8 quant + weight fold.

Companion to :func:`deep_gemm.fp8_paged_mqa_logits` — that kernel takes
``q`` already in FP8 e4m3fn (no per-token Q scale tensor; the per-token
scale must be folded into ``weights`` upstream).

Math:

    For each (b, s, h):
        absmax_h = max_d(|q_bf16[b,s,h,d]|)
        scale_h  = absmax_h / 448.0           (clamped to avoid /0)
        q_fp8_h  = round_e4m3(q_bf16 / scale_h)
        w_fold_h = w_bf16[b,s,h] * scale_h

Then DeepGEMM does ``score = q_fp8 @ k_dequant`` and the caller multiplies
by ``w_fold`` head-wise to get the correct ``ReLU(real_q · k) * w``
weighted sum (positive scale commutes with ReLU).

Public API:

  * :func:`indexer_q_fp8_quant_fold` — bf16 q [B,S,H,D] + bf16 weights [B,S,H]
    → (fp8 q [B,S,H,D], folded weights [B,S,H] fp32). One kernel pass.
  * :func:`indexer_q_rope_fp8_quant_fold` — same output contract, but applies
    RoPE to the q tail in registers before quantization. This replaces
    ``rope_only_inplace(q[..., -rope_dim:])`` followed by
    :func:`indexer_q_fp8_quant_fold` on the hot indexer path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_CACHED_CUDA_IDX_Q_OP = False
_CUDA_IDX_Q_OP = None


def _get_cuda_indexer_q_rope_op():
    global _CACHED_CUDA_IDX_Q_OP, _CUDA_IDX_Q_OP
    if not _CACHED_CUDA_IDX_Q_OP:
        try:
            from rtp_llm.ops import rtp_llm_ops

            _CUDA_IDX_Q_OP = getattr(
                rtp_llm_ops, "dsv4_indexer_q_rope_fp8_fold", None
            )
        except Exception:
            _CUDA_IDX_Q_OP = None
        _CACHED_CUDA_IDX_Q_OP = True
    return _CUDA_IDX_Q_OP

INDEXER_HEAD_DIM = 128
FP8_E4M3_MAX = 448.0


@triton.jit(do_not_specialize=["BSH"])
def _indexer_q_fp8_fold_kernel(
    q_ptr,  # [B*S, H, D] bf16
    w_ptr,  # [B*S, H]    bf16/fp32
    q_fp8_ptr,  # [B*S, H, D] fp8e4nv (uint8 view)
    w_fold_ptr,  # [B*S, H]    fp32
    # geometry
    BSH,  # B*S*H — rows
    D: tl.constexpr,  # head_dim = 128
    fp8_max: tl.constexpr,
    BLOCK_R: tl.constexpr = 8,
):
    """BLOCK_R rows per program for coalesced 128-bit loads and lower CTA overhead."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < BSH

    d_off = tl.arange(0, D)
    q = tl.load(
        q_ptr + rows[:, None] * D + d_off[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(q), axis=1)
    scale = tl.maximum(absmax / fp8_max, 1e-12)

    q_fp8 = (q / scale[:, None]).to(tl.float8e4nv)
    tl.store(
        (q_fp8_ptr + rows[:, None] * D + d_off[None, :]).to(tl.pointer_type(tl.uint8)),
        q_fp8.to(tl.uint8, bitcast=True),
        mask=row_mask[:, None],
    )

    w = tl.load(w_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    tl.store(w_fold_ptr + rows, w * scale, mask=row_mask)


def indexer_q_fp8_quant_fold(
    q_bf16: torch.Tensor,  # [B, S, H, D] bf16, D=128
    weights: torch.Tensor,  # [B, S, H]    bf16 or fp32
):
    """Per-(token, head) FP8 quant of indexer Q + scale-fold into weights.

    Returns ``(q_fp8, w_folded)`` where ``q_fp8`` is contiguous
    ``[B, S, H, D] float8_e4m3fn`` and ``w_folded`` is contiguous
    ``[B, S, H] float32``.
    """
    assert q_bf16.dim() == 4 and q_bf16.shape[-1] == INDEXER_HEAD_DIM
    assert q_bf16.dtype == torch.bfloat16
    assert q_bf16.is_contiguous()
    assert weights.shape == q_bf16.shape[:3]
    if weights.dtype != torch.bfloat16 and weights.dtype != torch.float32:
        weights = weights.float()
    weights = weights.contiguous()

    B, S, H, D = q_bf16.shape
    BSH = B * S * H
    q_fp8 = torch.empty(B, S, H, D, dtype=torch.float8_e4m3fn, device=q_bf16.device)
    w_fold = torch.empty(B, S, H, dtype=torch.float32, device=q_bf16.device)
    if BSH == 0:
        return q_fp8, w_fold

    block_r = 8
    _indexer_q_fp8_fold_kernel[(triton.cdiv(BSH, block_r),)](
        q_bf16,
        weights,
        q_fp8,
        w_fold,
        BSH=BSH,
        D=D,
        fp8_max=FP8_E4M3_MAX,
        BLOCK_R=block_r,
        num_warps=4,
    )
    return q_fp8, w_fold


@triton.jit(do_not_specialize=["BSH"])
def _indexer_q_rope_fp8_fold_kernel(
    q_ptr,  # [B*S, H, D] bf16, pre-RoPE
    w_ptr,  # [B*S, H]    bf16/fp32
    freqs_ri_ptr,  # [N_freq, RD/2, 2] float32 view_as_real(complex freqs)
    q_fp8_ptr,  # [B*S, H, D] fp8e4nv (uint8 view)
    w_fold_ptr,  # [B*S, H]    fp32
    BSH,  # B*S*H — rows
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    ROPE_START: tl.constexpr,
    fp8_max: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= BSH:
        return

    d_off = tl.arange(0, D)
    q = tl.load(q_ptr + pid * D + d_off).to(tl.float32)

    NUM_PAIRS: tl.constexpr = D // 2
    NOPE_PAIRS: tl.constexpr = ROPE_START // 2
    pair_off = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_off - NOPE_PAIRS
    is_rope_pair = (rope_pair_local >= 0) & (rope_pair_local < RD_HALF)
    cs_idx = tl.where(is_rope_pair, rope_pair_local, 0)

    freq_idx = pid // freq_stride_n
    freq_base = freqs_ri_ptr + freq_idx * freqs_stride_b + cs_idx * freqs_stride_k
    cos = tl.load(freq_base, mask=is_rope_pair, other=1.0)
    sin = tl.load(freq_base + 1, mask=is_rope_pair, other=0.0)

    q_even, q_odd = tl.split(tl.reshape(q, (NUM_PAIRS, 2)))
    rot_even = q_even * cos - q_odd * sin
    rot_odd = q_odd * cos + q_even * sin
    new_even = tl.where(is_rope_pair, rot_even, q_even)
    new_odd = tl.where(is_rope_pair, rot_odd, q_odd)
    q = tl.interleave(new_even, new_odd)
    q = q.to(tl.bfloat16).to(tl.float32)

    absmax = tl.max(tl.abs(q), axis=0)
    scale = tl.maximum(absmax / fp8_max, 1e-12)

    q_fp8 = (q / scale).to(tl.float8e4nv)
    tl.store(
        (q_fp8_ptr + pid * D + d_off).to(tl.pointer_type(tl.uint8)),
        q_fp8.to(tl.uint8, bitcast=True),
    )

    w = tl.load(w_ptr + pid).to(tl.float32)
    tl.store(w_fold_ptr + pid, w * scale)


@triton.jit(do_not_specialize=["freq_stride_n"])
def _indexer_q_rope_fp8_fold_group_heads_kernel(
    q_ptr,  # [B*S, H, D] bf16, pre-RoPE
    w_ptr,  # [B*S, H]    bf16/fp32
    freqs_ri_ptr,  # [N_freq, RD/2, 2] float32 view_as_real(complex freqs)
    q_fp8_ptr,  # [B*S, H, D] fp8e4nv (uint8 view)
    w_fold_ptr,  # [B*S, H]    fp32
    freq_stride_n,
    freqs_stride_b,
    freqs_stride_k,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    ROPE_START: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_HEADS: tl.constexpr,
):
    """Grouped-head RoPE + FP8 quant + scale fold.

    Single coalesced ``[GROUP_HEADS, D]`` load/store preserves full 128-col
    memory bursts and exact 0-ULP BF16/FP8 parity across group sizes.
    """
    freq_idx = tl.program_id(0).to(tl.int64)
    head_tile = tl.program_id(1).to(tl.int64)
    row_start = freq_idx * freq_stride_n + head_tile * GROUP_HEADS

    h_off = tl.arange(0, GROUP_HEADS)
    d_off = tl.arange(0, D)
    rows = row_start + h_off
    q = tl.load(q_ptr + rows[:, None] * D + d_off[None, :]).to(tl.float32)

    NUM_PAIRS: tl.constexpr = D // 2
    NOPE_PAIRS: tl.constexpr = ROPE_START // 2
    pair_off = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_off - NOPE_PAIRS
    is_rope_pair = (rope_pair_local >= 0) & (rope_pair_local < RD_HALF)
    cs_idx = tl.where(is_rope_pair, rope_pair_local, 0)

    freq_base = freqs_ri_ptr + freq_idx * freqs_stride_b + cs_idx * freqs_stride_k
    cos = tl.load(freq_base, mask=is_rope_pair, other=1.0)
    sin = tl.load(freq_base + 1, mask=is_rope_pair, other=0.0)

    q_even, q_odd = tl.split(tl.reshape(q, (GROUP_HEADS, NUM_PAIRS, 2)))
    rot_even = q_even * cos[None, :] - q_odd * sin[None, :]
    rot_odd = q_odd * cos[None, :] + q_even * sin[None, :]
    new_even = tl.where(is_rope_pair[None, :], rot_even, q_even)
    new_odd = tl.where(is_rope_pair[None, :], rot_odd, q_odd)
    q = tl.interleave(new_even, new_odd)
    q = q.to(tl.bfloat16).to(tl.float32)

    absmax = tl.max(tl.abs(q), axis=1)
    scale = tl.maximum(absmax / fp8_max, 1e-12)

    q_fp8 = (q / scale[:, None]).to(tl.float8e4nv)
    tl.store(
        (q_fp8_ptr + rows[:, None] * D + d_off[None, :]).to(tl.pointer_type(tl.uint8)),
        q_fp8.to(tl.uint8, bitcast=True),
    )

    w = tl.load(w_ptr + rows).to(tl.float32)
    tl.store(w_fold_ptr + rows, w * scale)


def _indexer_q_rope_fp8_quant_fold_impl(
    q_bf16: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    group_heads: int | None = None,
    allow_cuda: bool = True,
):
    assert q_bf16.dim() == 4 and q_bf16.shape[-1] == INDEXER_HEAD_DIM
    assert q_bf16.dtype == torch.bfloat16
    assert q_bf16.is_cuda
    assert weights.shape == q_bf16.shape[:3]
    assert rope_head_dim > 0 and rope_head_dim <= q_bf16.shape[-1]
    assert rope_head_dim % 2 == 0
    if not q_bf16.is_contiguous():
        q_bf16 = q_bf16.contiguous()
    if weights.dtype != torch.bfloat16 and weights.dtype != torch.float32:
        weights = weights.float()
    weights = weights.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()

    B, S, H, D = q_bf16.shape
    BSH = B * S * H
    q_fp8 = torch.empty(B, S, H, D, dtype=torch.float8_e4m3fn, device=q_bf16.device)
    w_fold = torch.empty(B, S, H, dtype=torch.float32, device=q_bf16.device)
    if BSH == 0:
        return q_fp8, w_fold

    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    n_freq = freqs_flat.shape[0]
    assert n_freq > 0
    assert BSH % n_freq == 0, f"N_rows={BSH} not divisible by N_freq={n_freq}"
    assert freqs_flat.shape[-1] == rope_head_dim // 2
    freqs_ri = torch.view_as_real(freqs_flat)
    freq_stride_n = BSH // n_freq

    if allow_cuda and group_heads is None:
        cuda_op = _get_cuda_indexer_q_rope_op()
        if (
            cuda_op is not None
            and D == INDEXER_HEAD_DIM
            and (q_bf16.data_ptr() & 15) == 0
            and (q_fp8.data_ptr() & 7) == 0
            and (rope_head_dim != 64 or (freqs_ri.data_ptr() & 15) == 0)
        ):
            cuda_op(
                q_bf16.view(BSH, D),
                weights.view(BSH),
                freqs_ri,
                q_fp8.view(BSH, D),
                w_fold.view(BSH),
                freq_stride_n,
                rope_head_dim,
                FP8_E4M3_MAX,
            )
            return q_fp8, w_fold

    if group_heads is not None:
        if group_heads not in (1, 2, 4, 8, 16):
            raise ValueError(
                f"invalid group_heads={group_heads}; expected 1, 2, 4, 8, or 16"
            )
        selected_group_heads = group_heads
    else:
        selected_group_heads = 1
        if D == INDEXER_HEAD_DIM:
            for candidate in (16, 8, 4, 2):
                if candidate == 16 and BSH < 65536:
                    continue
                if (
                    freq_stride_n % candidate == 0
                    and (freq_stride_n // candidate) <= 65535
                ):
                    selected_group_heads = candidate
                    break

    rope_start = D - rope_head_dim
    can_group_heads = (
        selected_group_heads > 1
        and D == INDEXER_HEAD_DIM
        and freq_stride_n % selected_group_heads == 0
        and (freq_stride_n // selected_group_heads) <= 65535
    )
    if can_group_heads:
        grid = (n_freq, freq_stride_n // selected_group_heads)
        _indexer_q_rope_fp8_fold_group_heads_kernel[grid](
            q_bf16,
            weights,
            freqs_ri,
            q_fp8,
            w_fold,
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            D=D,
            RD=rope_head_dim,
            RD_HALF=rope_head_dim // 2,
            ROPE_START=rope_start,
            fp8_max=FP8_E4M3_MAX,
            GROUP_HEADS=selected_group_heads,
            num_warps=4,
        )
    else:
        _indexer_q_rope_fp8_fold_kernel[(BSH,)](
            q_bf16,
            weights,
            freqs_ri,
            q_fp8,
            w_fold,
            BSH=BSH,
            freq_stride_n=freq_stride_n,
            freqs_stride_b=freqs_ri.stride(0),
            freqs_stride_k=freqs_ri.stride(1),
            D=D,
            RD=rope_head_dim,
            RD_HALF=rope_head_dim // 2,
            ROPE_START=rope_start,
            fp8_max=FP8_E4M3_MAX,
            num_warps=4,
        )
    return q_fp8, w_fold


def _indexer_q_rope_fp8_quant_fold_triton(
    q_bf16: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    group_heads: int | None = None,
):
    return _indexer_q_rope_fp8_quant_fold_impl(
        q_bf16,
        weights,
        freqs_cis,
        rope_head_dim,
        group_heads=group_heads,
        allow_cuda=False,
    )


def indexer_q_rope_fp8_quant_fold(
    q_bf16: torch.Tensor,  # [B, S, H, D] bf16, D=128, pre-RoPE
    weights: torch.Tensor,  # [B, S, H] bf16/fp32
    freqs_cis: torch.Tensor,  # complex [N_freq, RD/2]
    rope_head_dim: int,
    *,
    group_heads: int | None = None,
):
    """Fused RoPE + per-(token, head) FP8 quant + scale fold.

    Output contract is identical to :func:`indexer_q_fp8_quant_fold`.
    ``freqs_cis`` follows the same flattened-frequency mapping as
    ``rope_only_inplace``: each frequency row covers
    ``(B*S*H) // N_freq`` consecutive q rows.
    """
    return _indexer_q_rope_fp8_quant_fold_impl(
        q_bf16,
        weights,
        freqs_cis,
        rope_head_dim,
        group_heads=group_heads,
        allow_cuda=True,
    )
