"""DeepSeek-V4 Q/KV fused kernels, structured to match vLLM's flow.

vLLM (``vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py`` +
``vllm/model_executor/layers/deepseek_v4_attention.py``) runs four ops
around the fused ``[wq_a | wkv]`` GEMM:

  1. ``fused_wqa_wkv`` — single FP8 GEMM on ``x``.
  2. ``fused_q_kv_rmsnorm(qr_slice, kv_slice, q_norm, kv_norm, eps)``
     — one Triton launch that RMSNorms BOTH halves of the GEMM output
     using their respective weights, BEFORE ``wq_b``.
  3. ``wq_b(qr_normed)``.
  4. ``_fused_qnorm_rope_kv_insert(q, kv_normed, …)`` — per-head Q
     RMSNorm (no weight) + Q-RoPE + KV-RoPE + FP8 quant + cache
     insert in one C++ op.

This file mirrors steps 2 and 4 as Triton kernels:

  * :func:`fused_q_kv_rmsnorm` — vLLM's grid-``(N_tok, 2)`` kernel.
    Reads ``qr`` from ``qkv_a[..., 0:q_lora_rank]`` and ``kv`` from
    ``qkv_a[..., q_lora_rank:q_lora_rank + head_dim]`` directly via
    a constexpr offset, so the caller never needs ``.split()`` or
    ``.contiguous()`` on the Python side. Returns ``(qr, kv)`` as
    fresh contiguous buffers ready for ``wq_b`` and the post-wq_b
    kernel respectively.
  * :func:`fused_q_perhead_norm_qkv_rope` — per-head Q RMSNorm
    (no weight) + Q-RoPE + KV-RoPE. Q programs tile ``GROUP_HEADS``
    heads at a time so each program amortizes its freqs load over G
    heads; the last block-row of the grid is the KV slot (RoPE only —
    KV was already normed in step 2). Launch config (GROUP_HEADS,
    num_warps, num_stages) is picked from a shape-keyed dispatch
    table tuned for V4-Flash (n_heads=64) and V4-Pro (n_heads=128).

V4-Flash and V4-Pro both have ``head_dim == kv_dim == 512`` and share
``rope_head_dim == 64`` between Q and KV, so one kernel template
handles both row shapes — the branch on the KV slot is per-CTA
(no thread divergence).

Correctness gate:
``rtp_llm/models_py/modules/dsv4/test/test_fused_qkv_rmsnorm_rope.py``
(parity vs torch reference, parametrized over both V4 variants).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pre-wq_b: fused RMSNorm of [qr_slice, kv_slice] out of the packed
# ``qkv_a`` GEMM output.  Mirrors vLLM's ``fused_q_kv_rmsnorm`` —
# grid (N_tok, 2), pid_task selects qr vs kv.  Reads slices directly
# from ``qkv_a`` via constexpr offsets so the caller doesn't have to
# ``.split()``.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    qkv_a_ptr,
    qr_out_ptr,
    kv_out_ptr,
    q_weight_ptr,
    kv_weight_ptr,
    qkv_a_stride_n,
    qr_out_stride_n,
    kv_out_stride_n,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    KV_OFFSET: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # token on grid-x (max 2^31 - 1); task on grid-y. CUDA caps grid-y/z
    # at 65535 — putting num_tokens there crashes the launch at
    # max-num-batched-tokens >= 65536 (matches the same warning vLLM
    # leaves in its kernel).
    token = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = qkv_a_ptr + token * qkv_a_stride_n  # offset 0
        row_out = qr_out_ptr + token * qr_out_stride_n
        weight_ptr = q_weight_ptr
    else:
        SIZE = KV_SIZE
        row_in = qkv_a_ptr + token * qkv_a_stride_n + KV_OFFSET
        row_out = kv_out_ptr + token * kv_out_stride_n
        weight_ptr = kv_weight_ptr

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + EPS)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


def fused_q_kv_rmsnorm(
    qkv_a: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    *,
    q_size: int,
    kv_offset: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm both halves of the fused ``[wq_a | wkv]`` GEMM output.

    ``qkv_a``  — ``[..., total_dim]`` bf16, row-major last dim
                  (CudaFp8DeepGEMMLinear output is always contiguous).
    ``q_norm`` — ``[q_size]`` bf16 RMSNorm weight for the qr slice.
    ``kv_norm`` — ``[kv_size]`` bf16 RMSNorm weight; ``kv_size`` is
                  inferred from ``kv_norm.shape[0]``.
    ``q_size`` — length of the qr slice along the last dim, starting
                  at offset 0 (== ``q_lora_rank``).
    ``kv_offset`` — start of the kv slice along the last dim
                  (== ``q_lora_rank``).

    Returns ``(qr, kv)``, both fresh contiguous tensors:
      * ``qr``  shape ``qkv_a.shape[:-1] + (q_size,)``
      * ``kv``  shape ``qkv_a.shape[:-1] + (kv_norm.shape[0],)``
    """
    assert qkv_a.is_cuda and qkv_a.dtype == torch.bfloat16
    assert qkv_a.stride(-1) == 1, "qkv_a must be row-major"
    assert q_norm.is_contiguous() and q_norm.shape == (q_size,)
    assert kv_norm.is_contiguous() and kv_norm.dim() == 1
    kv_size = int(kv_norm.shape[0])
    assert kv_offset + kv_size <= qkv_a.shape[-1], (
        f"kv slice [{kv_offset}:{kv_offset + kv_size}] out of range for "
        f"qkv_a last dim {qkv_a.shape[-1]}"
    )

    leading_shape = qkv_a.shape[:-1]
    qkv_a_flat = qkv_a.reshape(-1, qkv_a.shape[-1])
    n_tokens = qkv_a_flat.shape[0]

    qr_out = torch.empty(n_tokens, q_size, dtype=qkv_a.dtype, device=qkv_a.device)
    kv_out = torch.empty(n_tokens, kv_size, dtype=qkv_a.dtype, device=qkv_a.device)
    if n_tokens == 0:
        return qr_out.view(*leading_shape, q_size), kv_out.view(*leading_shape, kv_size)

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    assert block_size <= 4096

    _fused_q_kv_rmsnorm_kernel[(n_tokens, 2)](
        qkv_a_flat,
        qr_out,
        kv_out,
        q_norm,
        kv_norm,
        qkv_a_flat.stride(0),
        qr_out.stride(0),
        kv_out.stride(0),
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        KV_OFFSET=kv_offset,
        EPS=eps,
        BLOCK_SIZE=block_size,
    )
    return qr_out.view(*leading_shape, q_size), kv_out.view(*leading_shape, kv_size)


# ---------------------------------------------------------------------------
# Post-wq_b: per-head Q RMSNorm (no weight) + Q partial RoPE + KV
# partial RoPE.  V4-Flash/V4-Pro both have head_dim == kv_dim == 512 and
# share rope_head_dim == 64, so one kernel template handles both row
# shapes — the program at the last grid-y row takes the KV branch
# (RoPE only — kv was already normed in step 2).
#
# Q programs tile ``GROUP_HEADS`` heads at a time. Each program loads a
# ``[G, D]`` slab and a single ``cos/sin`` row from ``freqs_cis``,
# broadcasting against G heads — cuts freqs traffic by G× and shrinks
# the grid from (N_tok, n_heads+1) to (N_tok, ceil(n_heads/G)+1).
# ---------------------------------------------------------------------------
@triton.jit
def _fused_q_perhead_norm_qkv_rope_kernel(
    q_ptr,
    kv_ptr,
    kv_out_ptr,
    freqs_ri_ptr,
    q_stride_n,
    q_stride_h,
    kv_stride_n,
    kv_out_stride_n,
    freqs_stride_n,
    freqs_stride_k,
    N_HEADS: tl.constexpr,
    D: tl.constexpr,
    RD_HALF: tl.constexpr,
    NOPE_OFFSET: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_HEADS: tl.constexpr,
    N_BLOCKS_Q: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block_h = tl.program_id(1)
    # Last block-row is the KV slot. N_BLOCKS_Q = ceil(N_HEADS/G), grid-y = N_BLOCKS_Q + 1.
    is_kv = block_h == N_BLOCKS_Q

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    nope_mask = d_mask & (d_off < NOPE_OFFSET)
    pair_off = tl.arange(0, RD_HALF)
    real_off = NOPE_OFFSET + 2 * pair_off
    imag_off = real_off + 1

    # Freqs shared across all heads of this program — loaded once.
    freq_base = freqs_ri_ptr + token * freqs_stride_n + pair_off * freqs_stride_k
    cos = tl.load(freq_base)  # [RD_HALF]
    sin = tl.load(freq_base + 1)  # [RD_HALF]

    # Two self-contained branches; each uses distinct variable names so
    # Triton's SSA type-unification at the join point doesn't complain
    # about ``imag`` being [RD_HALF] in one branch and [G, RD_HALF] in
    # the other.
    if is_kv:
        # KV: RoPE-only — already RMSNormed by fused_q_kv_rmsnorm.
        # Copy NOPE region straight through, rotate RoPE region.
        kv_in = kv_ptr + token * kv_stride_n
        kv_out = kv_out_ptr + token * kv_out_stride_n
        x_nope = tl.load(kv_in + d_off, mask=nope_mask, other=0.0)
        tl.store(kv_out + d_off, x_nope, mask=nope_mask)
        kv_real = tl.load(kv_in + real_off).to(tl.float32)
        kv_imag = tl.load(kv_in + imag_off).to(tl.float32)
        kv_new_real = kv_real * cos - kv_imag * sin
        kv_new_imag = kv_real * sin + kv_imag * cos
        tl.store(kv_out + real_off, kv_new_real)
        tl.store(kv_out + imag_off, kv_new_imag)
    else:
        # Q: per-head RMSNorm (no weight) + partial RoPE over a [G, D]
        # tile.  RMSNorm is per-head (reduce along axis=1) so the
        # numerics are bit-identical to the G=1 path.
        g_off = block_h * GROUP_HEADS + tl.arange(0, GROUP_HEADS)
        g_mask = g_off < N_HEADS

        row_base = q_ptr + token * q_stride_n + g_off.to(tl.int64) * q_stride_h
        tile_ptrs = row_base[:, None] + d_off[None, :]
        tile_mask = g_mask[:, None] & d_mask[None, :]
        q_tile = tl.load(tile_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
        q_var = tl.sum(q_tile * q_tile, axis=1) / D  # [G]
        q_inv = tl.rsqrt(q_var + EPS)  # [G]
        q_y = q_tile * q_inv[:, None]  # [G, BLOCK_D] fp32

        # Write NOPE region back (in place, bf16 cast on store).
        nope_tile_mask = g_mask[:, None] & nope_mask[None, :]
        tl.store(tile_ptrs, q_y, mask=nope_tile_mask)

        # RoPE on (real, imag) pairs.  Re-load real/imag from input
        # (the NOPE store left them untouched), multiply by inv to
        # match post-rmsnorm, then rotate.  fp32 throughout — matches
        # the original kernel's "fp32 through RoPE" semantics.
        q_real_ptrs = row_base[:, None] + real_off[None, :]
        q_imag_ptrs = row_base[:, None] + imag_off[None, :]
        q_head_mask = g_mask[:, None]
        q_real = tl.load(q_real_ptrs, mask=q_head_mask, other=0.0).to(tl.float32) * q_inv[:, None]
        q_imag = tl.load(q_imag_ptrs, mask=q_head_mask, other=0.0).to(tl.float32) * q_inv[:, None]
        q_new_real = q_real * cos[None, :] - q_imag * sin[None, :]
        q_new_imag = q_real * sin[None, :] + q_imag * cos[None, :]
        tl.store(q_real_ptrs, q_new_real, mask=q_head_mask)
        tl.store(q_imag_ptrs, q_new_imag, mask=q_head_mask)


# ---------------------------------------------------------------------------
# Shape-keyed launch config dispatch (no env var, no @triton.autotune).
#
# Keys: (n_heads, head_dim, bucket).  Buckets:
#   small  : n_tok <= 32     (decode)
#   mid    : 33 <= n_tok <= 1024
#   large  : n_tok > 1024    (prefill)
#
# Values: (GROUP_HEADS, num_warps, num_stages).
#
# The "*" wildcards form the safe G=1 fallback for unknown shapes.
#
# Populated from the tuning sweep at
# ``test_fused_qkv_rmsnorm_rope_perf.py::test_perf_tune``
# (DSV4_QKV_RMSNORM_ROPE_TUNE=1) on L20D-as-B300 (compute_cap=(10,3),
# HBM peak ≈ 7.6 TB/s).  Bucket cutoffs match B300 wave-fill behaviour:
# 132 SMs × ~8 progs each ≈ 1056 active programs, so once
# ``n_tok × ceil(n_heads/G) >= 1024`` the grid is occupancy-bound
# rather than launch-bound.
#
# Tuning snapshot (event_us is per-call latency including the
# ~29 µs Python+Triton dispatch overhead — see test_perf_tune output):
#
# ## V4-Flash (n_heads=64, D=512)
# | bucket | N_tok | best (G,w,s) | best µs | gain vs (1,4,3) |
# |--------|-------|--------------|---------|------------------|
# | small  | 1     | (2, 8, 3)    | 29.256  | 1.02x            |
# | small  | 8     | (4, 8, 4)    | 29.388  | 1.01x            |
# | small  | 32    | (4, 8, 3)    | 29.207  | 1.01x            |
# | mid    | 128   | (1, 4, 2)    | 29.187  | 1.00x            |
# | mid    | 512   | (4, 2, 2)    | 30.010  | 1.04x            |
# | mid    | 1024  | (2, 1, 2)    | 29.833  | 1.97x            |
# | large  | 4096  | (2, 1, 2)    | 100.904 | 2.26x            |
# | large  | 16384 | (2, 1, 3)    | 360.128 | 2.50x            |
# | large  | 65536 | (4, 1, 2)    | 1387.77 | 2.63x            |
#
# ## V4-Pro (n_heads=128, D=512)
# | bucket | N_tok | best (G,w,s) | best µs | gain vs (1,4,3) |
# |--------|-------|--------------|---------|------------------|
# | small  | 1     | (1, 8, 4)    | 29.154  | 1.00x            |
# | small  | 8     | (4, 8, 3)    | 29.281  | 1.00x            |
# | small  | 32    | (8, 2, 4)    | 29.319  | 1.01x            |
# | mid    | 128   | (8, 8, 4)    | 29.424  | 1.02x            |
# | mid    | 512   | (2, 1, 4)    | 29.993  | 1.93x            |
# | mid    | 1024  | (2, 1, 4)    | 47.039  | 2.46x            |
# | large  | 4096  | (2, 1, 4)    | 174.812 | 2.57x            |
# | large  | 16384 | (2, 1, 3)    | 663.178 | 2.69x            |
# | large  | 65536 | (4, 1, 3)    | 2715.80 | 2.67x            |
#
# Decisions for the table:
#   * small  : all configs land within noise (~29 µs ≈ Python+Triton
#              dispatch).  Pick a moderate G/warps that doesn't hurt
#              when the bucket's N grows toward 32.
#   * mid    : the 128/512 entries are launch-bound; the 1024 entry
#              already shows the (2,1,*) win — bias the bucket toward
#              the large-N config so the cliff at N>1024 is smooth.
#   * large  : strong, consistent win for ``GROUP_HEADS=2,
#              num_warps=1``.  num_stages tuned per variant
#              (V4-Flash s=2, V4-Pro s=4 — the wider head count needs
#              more software pipelining to overlap loads).
# ---------------------------------------------------------------------------
_LAUNCH_CONFIGS = {
    # V4-Flash (n_heads=64, head_dim=512)
    (64, 512, "small"): (4, 4, 2),
    (64, 512, "mid"): (2, 1, 2),
    (64, 512, "large"): (2, 1, 2),
    # V4-Pro (n_heads=128, head_dim=512)
    (128, 512, "small"): (4, 4, 2),
    (128, 512, "mid"): (2, 1, 4),
    (128, 512, "large"): (2, 1, 4),
    # Fallback (always-safe G=1 path).
    ("*", "*", "small"): (1, 2, 2),
    ("*", "*", "mid"): (1, 4, 3),
    ("*", "*", "large"): (1, 4, 3),
}


def _bucket_for(n_tok: int) -> str:
    if n_tok <= 32:
        return "small"
    if n_tok <= 1024:
        return "mid"
    return "large"


def _select_launch(n_tok: int, n_heads: int, head_dim: int) -> Tuple[int, int, int]:
    """Return (GROUP_HEADS, num_warps, num_stages) for the live shape."""
    bucket = _bucket_for(n_tok)
    cfg = _LAUNCH_CONFIGS.get((n_heads, head_dim, bucket))
    if cfg is not None:
        return cfg
    return _LAUNCH_CONFIGS[("*", "*", bucket)]


def fused_q_perhead_norm_qkv_rope(
    q: torch.Tensor,
    kv: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
    kv_out: torch.Tensor | None = None,
    _launch_override: Optional[Tuple[int, int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-wq_b: per-head Q RMSNorm (no weight) + Q-RoPE + KV-RoPE.

    ``q``         — ``[N_tok, n_heads, D]`` or ``[B, S, n_heads, D]``
                    bf16, contiguous along the last 2 dims (head, D);
                    rewritten in place.
    ``kv``        — ``[N_tok, D]`` or ``[B, S, D]`` bf16, **already
                    RMSNormed** by :func:`fused_q_kv_rmsnorm`. Last dim
                    contiguous; leading stride may differ from ``D``.
    ``freqs_cis`` — complex64 with leading shape totaling ``N_tok``
                    elements.
    ``rope_head_dim`` — trailing cols of each row rotated by RoPE.
    ``_launch_override`` — private knob for the tuning sweep
                    (DSV4_QKV_RMSNORM_ROPE_TUNE=1); production callers
                    leave this ``None`` and pick up the shape-keyed
                    dispatch table.

    Returns ``(q, kv_out)``. ``q`` is the same tensor (in-place).
    ``kv_out`` is a fresh ``[..., D]`` tensor (or the caller-supplied
    buffer).
    """
    assert q.is_cuda and kv.is_cuda
    assert q.dtype == kv.dtype == torch.bfloat16
    assert freqs_cis.is_cuda and freqs_cis.is_contiguous()

    D = int(q.shape[-1])
    N_HEADS = int(q.shape[-2])
    RD = int(rope_head_dim)
    assert RD % 2 == 0 and RD <= D
    assert kv.shape[-1] == D, (
        f"q head_dim {D} must match kv last dim {kv.shape[-1]} for V4 "
        "(head_dim == kv_dim)"
    )

    q_flat = q.reshape(-1, N_HEADS, D)
    kv_flat = kv.reshape(-1, D)
    assert q_flat.is_contiguous(), "q must be contiguous (rewritten in place)"
    assert kv_flat.stride(-1) == 1, "kv must be contiguous along the last dim"

    n_tokens = q_flat.shape[0]
    assert kv_flat.shape[0] == n_tokens, (
        f"q token count {n_tokens} must equal kv token count {kv_flat.shape[0]}"
    )

    kv_leading_shape = kv.shape[:-1]
    if kv_out is None:
        kv_out_flat = torch.empty(n_tokens, D, dtype=q.dtype, device=q.device)
    else:
        assert kv_out.shape == kv_leading_shape + (D,) and kv_out.dtype == q.dtype
        assert kv_out.is_cuda and kv_out.is_contiguous()
        kv_out_flat = kv_out.reshape(-1, D)

    if n_tokens == 0:
        return q, kv_out_flat.view(*kv_leading_shape, D)

    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    assert freqs_flat.shape[0] == n_tokens, (
        f"freqs token count {freqs_flat.shape[0]} must equal n_tokens {n_tokens}"
    )
    freqs_ri = torch.view_as_real(freqs_flat)
    assert freqs_ri.shape == (n_tokens, RD // 2, 2)

    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 4096

    if _launch_override is not None:
        group_heads, num_warps, num_stages = _launch_override
    else:
        group_heads, num_warps, num_stages = _select_launch(n_tokens, N_HEADS, D)
    assert group_heads >= 1 and (group_heads & (group_heads - 1)) == 0, (
        f"GROUP_HEADS must be a power of two, got {group_heads}"
    )

    n_blocks_q = (N_HEADS + group_heads - 1) // group_heads
    grid = (n_tokens, n_blocks_q + 1)
    _fused_q_perhead_norm_qkv_rope_kernel[grid](
        q_flat,
        kv_flat,
        kv_out_flat,
        freqs_ri,
        q_flat.stride(0),
        q_flat.stride(1),
        kv_flat.stride(0),
        kv_out_flat.stride(0),
        freqs_ri.stride(0),
        freqs_ri.stride(1),
        N_HEADS=N_HEADS,
        D=D,
        RD_HALF=RD // 2,
        NOPE_OFFSET=D - RD,
        EPS=eps,
        BLOCK_D=BLOCK_D,
        GROUP_HEADS=group_heads,
        N_BLOCKS_Q=n_blocks_q,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q, kv_out_flat.view(*kv_leading_shape, D)
