"""Qwen 4.5-Flash Per-Layer N-gram Embedding (PLE) with FP8 per-tensor table & Triton kernels.

Implements exact Qwen 4.5-Flash Layer 1 PLE math:
  - EOS-aware right-shifted 2-gram / 3-gram hashing (20 heads = 10x 2-gram + 10x 3-gram)
  - FP8 per-tensor sharded/flat table gather supporting GPU tables, CPU pinned UVA tables
    (for the full 179.2B / 167 GiB table on single-node B300), and unpinned CPU host-gather + H2D
  - Fused K/V projection + per-branch GemmaRMSNorm (no bias) + sqrt-signed dot-product gate
  - Depthwise causal ShortConv (k=4, C=20480, residual=False, use_activation=False)
  - Residual injection H + (U + conv1d(U_norm))
"""

from __future__ import annotations

import ctypes
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .contracts import PLELayerState
from .short_conv import (
    Qwen4_5ShortConv,
    causal_dwconv1d_ragged_prefill_with_snapshots_triton,
    causal_dwconv1d_verify_triton,
)

# Real Qwen 4.5-Flash Layer 1 PLE constants
DEFAULT_PLE_LAYER_MULTIPLIERS: Tuple[int, ...] = (
    23703573157769,
    20109073645365,
    8052911324071,
)

# First 20 primes >= 35,000,000 (sum = 700,004,074; padded to mult of 512 = 700,004,352;
# 128 shards x 5,468,784 rows x 256 dim = 179.2B FP8 weights = 166.89 GiB)
DEFAULT_PLE_NGRAM_VOCAB_SIZES: Tuple[int, ...] = (
    35000011,
    35000041,
    35000083,
    35000099,
    35000107,
    35000123,
    35000137,
    35000159,
    35000171,
    35000179,
    35000221,
    35000233,
    35000267,
    35000269,
    35000279,
    35000293,
    35000321,
    35000341,
    35000363,
    35000377,
)

DEFAULT_PLE_WEIGHT_SCALE: float = 0.00090789794921875


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def compute_ngram_vocab_sizes(base: int, num_heads: int) -> Tuple[int, ...]:
    """Return the first ``num_heads`` prime numbers >= ``base``."""
    primes: List[int] = []
    candidate = max(2, int(base))
    while len(primes) < num_heads:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _resolve_cpu_uva_pointers(
    shards: List[torch.Tensor],
    device: torch.device,
    *,
    pin_cpu: bool,
) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
    """Map pinned CPU shards into CUDA UVA space via cudaHostGetDevicePointer."""
    if device.type != "cuda" or torch.version.hip is not None:
        return None
    if not shards:
        return None
    if not all(s.is_pinned() for s in shards) and not pin_cpu:
        return None

    torch.cuda.init()
    runtime = ctypes.CDLL(torch._C.__file__)
    get_ptr = runtime.cudaHostGetDevicePointer
    get_ptr.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    get_ptr.restype = ctypes.c_int

    pinned_shards: List[torch.Tensor] = []
    ptrs: List[int] = []
    with torch.cuda.device(device):
        for shard in shards:
            owner = shard if shard.is_pinned() else shard.pin_memory()
            addr = ctypes.c_void_p()
            rc = get_ptr(ctypes.byref(addr), owner.data_ptr(), 0)
            if rc != 0 or addr.value is None:
                return None
            pinned_shards.append(owner)
            ptrs.append(int(addr.value))
        ptr_table = torch.tensor(ptrs, dtype=torch.int64, device=device)
    return pinned_shards, ptr_table


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["BATCH", "SEQ"])
def _ple_hash_ngram_ids_kernel(
    PREV_CTX_PTR,
    IDS_PTR,
    VOCAB_PTR,
    OFFSETS_PTR,
    MULT_PTR,
    OUT_PTR,
    STAGE_CTX_PTR,
    STAGE_SB: tl.int64,
    STAGE_SV: tl.int64,
    BATCH,
    SEQ,
    EOS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_CTX: tl.constexpr,
    UPDATE_CTX_T1: tl.constexpr,
    WRITE_STAGE_CTX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    total_rows = BATCH.to(tl.int64) * SEQ.to(tl.int64)
    valid_row = row < total_rows
    batch = row // SEQ
    t = row % SEQ

    cur = tl.load(IDS_PTR + row, mask=valid_row, other=EOS)
    prev_from_ids = tl.load(IDS_PTR + row - 1, mask=valid_row & (t >= 1), other=EOS)
    prev2_from_ids = tl.load(IDS_PTR + row - 2, mask=valid_row & (t >= 2), other=EOS)

    if HAS_CTX:
        ctx1 = tl.load(PREV_CTX_PTR + batch * 2 + 1, mask=valid_row & (t < 2), other=EOS)
        ctx0 = tl.load(PREV_CTX_PTR + batch * 2 + 0, mask=valid_row & (t == 0), other=EOS)
        prev = tl.where(t >= 1, prev_from_ids, ctx1)
        prev2_ctx = tl.where(t == 1, ctx1, ctx0)
        prev2 = tl.where(t >= 2, prev2_from_ids, prev2_ctx)
        if UPDATE_CTX_T1:
            tl.store(PREV_CTX_PTR + batch * 2 + 0, ctx1, mask=valid_row)
            tl.store(PREV_CTX_PTR + batch * 2 + 1, cur, mask=valid_row)
    else:
        prev = prev_from_ids
        prev2 = prev2_from_ids

    if WRITE_STAGE_CTX:
        tl.store(STAGE_CTX_PTR + batch * STAGE_SB + t * STAGE_SV + 0, prev, mask=valid_row)
        tl.store(STAGE_CTX_PTR + batch * STAGE_SB + t * STAGE_SV + 1, cur, mask=valid_row)

    # EOS barrier: if previous token is EOS, 2-step history resets to EOS as well.
    prev2 = tl.where(prev == EOS, EOS, prev2)

    m0 = tl.load(MULT_PTR + 0)
    m1 = tl.load(MULT_PTR + 1)
    m2 = tl.load(MULT_PTR + 2)
    mix2 = (cur * m0) ^ (prev * m1)
    mix3 = mix2 ^ (prev2 * m2)

    head = tl.arange(0, HEAD_BLOCK)
    head_mask = head < NUM_HEADS
    mixed = tl.where(head[None, :] < HEADS_PER_NGRAM, mix2[:, None], mix3[:, None])
    vocab = tl.load(VOCAB_PTR + head, mask=head_mask, other=1)
    offsets = tl.load(OFFSETS_PTR + head, mask=head_mask, other=0)

    # Euclidean modulo matching torch.remainder(mixed, vocab) for signed int64
    rem = mixed % vocab[None, :]
    rem = tl.where(rem < 0, rem + vocab[None, :], rem)
    out_ids = rem + offsets[None, :]

    tl.store(
        OUT_PTR + row[:, None] * NUM_HEADS + head[None, :],
        out_ids,
        mask=valid_row[:, None] & head_mask[None, :],
    )


@triton.jit(do_not_specialize=["N"])
def _ple_hash_ngram_ids_ragged_kernel(
    PREV_CTX_PTR,
    IDS_PTR,
    CU_SEQLENS_PTR,
    TOK_REQ_PTR,
    VOCAB_PTR,
    OFFSETS_PTR,
    MULT_PTR,
    OUT_PTR,
    N,
    EOS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_CTX: tl.constexpr,
):
    """Packed ragged 2-gram + 3-gram hashing across variable-length requests."""
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    valid_row = row < N
    b = tl.load(TOK_REQ_PTR + row, mask=valid_row, other=0).to(tl.int64)
    req_start = tl.load(CU_SEQLENS_PTR + b, mask=valid_row, other=0).to(tl.int64)
    t = row - req_start

    cur = tl.load(IDS_PTR + row, mask=valid_row, other=EOS)
    prev_from_ids = tl.load(IDS_PTR + row - 1, mask=valid_row & (t >= 1), other=EOS)
    prev2_from_ids = tl.load(IDS_PTR + row - 2, mask=valid_row & (t >= 2), other=EOS)

    if HAS_CTX:
        ctx1 = tl.load(PREV_CTX_PTR + b * 2 + 1, mask=valid_row & (t < 2), other=EOS)
        ctx0 = tl.load(PREV_CTX_PTR + b * 2 + 0, mask=valid_row & (t == 0), other=EOS)
        prev = tl.where(t >= 1, prev_from_ids, ctx1)
        prev2_ctx = tl.where(t == 1, ctx1, ctx0)
        prev2 = tl.where(t >= 2, prev2_from_ids, prev2_ctx)
    else:
        prev = prev_from_ids
        prev2 = prev2_from_ids

    prev2 = tl.where(prev == EOS, EOS, prev2)

    m0 = tl.load(MULT_PTR + 0)
    m1 = tl.load(MULT_PTR + 1)
    m2 = tl.load(MULT_PTR + 2)
    mix2 = (cur * m0) ^ (prev * m1)
    mix3 = mix2 ^ (prev2 * m2)

    head = tl.arange(0, HEAD_BLOCK)
    head_mask = head < NUM_HEADS
    mixed = tl.where(head[None, :] < HEADS_PER_NGRAM, mix2[:, None], mix3[:, None])
    vocab = tl.load(VOCAB_PTR + head, mask=head_mask, other=1)
    offsets = tl.load(OFFSETS_PTR + head, mask=head_mask, other=0)

    rem = mixed % vocab[None, :]
    rem = tl.where(rem < 0, rem + vocab[None, :], rem)
    out_ids = rem + offsets[None, :]

    tl.store(
        OUT_PTR + row[:, None] * NUM_HEADS + head[None, :],
        out_ids,
        mask=valid_row[:, None] & head_mask[None, :],
    )


@triton.jit
def _scatter_ple_ctx_snapshots_ragged_kernel(
    PREV_CTX_PTR,
    IDS_PTR,
    CU_SEQLENS_PTR,
    SNAP_TOK_PTR,
    SNAP_REQ_PTR,
    SNAP_DST_PTR,
    DST_PTR,
    dst_stride_row: tl.int64,
    dst_rows: tl.int64,
    EOS: tl.constexpr,
    HAS_CTX: tl.constexpr,
    SKIP_INVALID_PHYS: tl.constexpr,
):
    """Scatter boundary 2-token ngram_context snapshots for packed ragged prefill."""
    s_id = tl.program_id(0).to(tl.int64)
    dst_idx = tl.load(SNAP_DST_PTR + s_id).to(tl.int64)
    if SKIP_INVALID_PHYS:
        if dst_idx <= 0 or dst_idx >= dst_rows:
            return
    g_tok = tl.load(SNAP_TOK_PTR + s_id).to(tl.int64)
    b = tl.load(SNAP_REQ_PTR + s_id).to(tl.int64)
    req_start = tl.load(CU_SEQLENS_PTR + b).to(tl.int64)
    local_t = g_tok - req_start

    c1 = tl.load(IDS_PTR + g_tok).to(tl.int64)
    if local_t >= 1:
        c0 = tl.load(IDS_PTR + g_tok - 1).to(tl.int64)
    elif HAS_CTX:
        c0 = tl.load(PREV_CTX_PTR + b * 2 + 1).to(tl.int64)
    else:
        c0 = tl.full((), EOS, tl.int64)

    tl.store(DST_PTR + dst_idx * dst_stride_row + 0, c0)
    tl.store(DST_PTR + dst_idx * dst_stride_row + 1, c1)


@triton.jit(do_not_specialize=["N", "SHARD_ROWS"])
def _ple_gather_shards_kernel(
    IDS_PTR,
    SHARD_PTRS,
    SCALE_PTR,
    OUT_PTR,
    N,
    SHARD_ROWS,
    TOTAL_SHARDS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    IS_FP8: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR).to(tl.int64)
    cols = tl.arange(0, BD).to(tl.int64)
    row_mask = rows < N
    col_mask = cols < D

    ids = tl.load(IDS_PTR + rows, mask=row_mask, other=0).to(tl.int64)
    shard_rows_64 = SHARD_ROWS.to(tl.int64)
    shard_idx = ids // shard_rows_64
    local_row = ids % shard_rows_64
    in_range = row_mask & (ids >= 0) & (shard_idx < TOTAL_SHARDS)
    addr = tl.load(SHARD_PTRS + shard_idx, mask=in_range, other=0)
    valid = in_range & (addr != 0)

    if IS_FP8:
        base = addr.to(tl.pointer_type(tl.uint8))
    elif IS_BF16:
        base = addr.to(tl.pointer_type(tl.bfloat16))
    else:
        base = addr.to(tl.pointer_type(tl.float32))

    d64 = tl.full((), D, tl.int64)
    vals = tl.load(
        base[:, None] + local_row[:, None] * d64 + cols[None, :],
        mask=valid[:, None] & col_mask[None, :],
        other=0,
    )
    if IS_FP8:
        out_ty = OUT_PTR.dtype.element_ty
        scale = tl.load(SCALE_PTR).to(out_ty)
        vals = vals.to(tl.float8e4nv, bitcast=True).to(out_ty) * scale

    tl.store(
        OUT_PTR + rows[:, None] * d64 + cols[None, :],
        vals,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _ple_gate_and_norm_kernel(
    KEY_PTR,
    VAL_PTR,
    HID_PTR,
    GK_PTR,
    GQ_PTR,
    GC_PTR,
    U_PTR,
    UNORM_PTR,
    COMB_SUB_PTR,
    COMB_INJ_PTR,
    COMB_OUT_PTR,
    KEY_STRIDE: tl.constexpr,
    VAL_STRIDE: tl.constexpr,
    HID_STRIDE: tl.constexpr,
    COMB_OUT_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    H: tl.constexpr,
    HC_COUNT: tl.constexpr,
    EPS: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < H
    hc_col = branch * H + col

    # 1. Per-branch GemmaRMSNorm on K and (optionally deferred-combined) H
    k = tl.load(KEY_PTR + row * KEY_STRIDE + hc_col, mask=valid, other=0.0).to(tl.float32)
    gk = tl.load(GK_PTR + hc_col, mask=valid, other=0.0).to(tl.float32)
    kn = (k * 1.0 / tl.sqrt(tl.sum(k * k, axis=0) / H + EPS) * (1.0 + gk)).to(
        KEY_PTR.dtype.element_ty
    )

    q = tl.load(HID_PTR + row * HID_STRIDE + hc_col, mask=valid, other=0.0).to(tl.float32)
    if HAS_COMBINE:
        sub = tl.load(COMB_SUB_PTR + row * H + col, mask=valid, other=0.0).to(tl.float32)
        inj = tl.load(COMB_INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
        q_comb = (q + sub * inj).to(HID_PTR.dtype.element_ty)
        tl.store(COMB_OUT_PTR + row * COMB_OUT_STRIDE + hc_col, q_comb, mask=valid)
        q = q_comb.to(tl.float32)
    gq = tl.load(GQ_PTR + hc_col, mask=valid, other=0.0).to(tl.float32)
    qn = (q * 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / H + EPS) * (1.0 + gq)).to(
        HID_PTR.dtype.element_ty
    )

    # 2. FP32 scaled dot product + signed sqrt gate
    dot = tl.sum(kn.to(tl.float32) * qn.to(tl.float32), axis=0)
    s = dot / tl.sqrt(float(H))
    mag = tl.sqrt(tl.maximum(tl.abs(s), 1e-6))
    sign = tl.where(s > 0.0, 1.0, tl.where(s < 0.0, -1.0, 0.0))
    alpha = tl.sigmoid(mag * sign).to(VAL_PTR.dtype.element_ty).to(tl.float32)

    # 3. U = alpha * V, then per-branch GemmaRMSNorm(U, norm_conv)
    v = tl.load(VAL_PTR + row * VAL_STRIDE + col, mask=valid, other=0.0).to(tl.float32)
    u = (alpha * v).to(U_PTR.dtype.element_ty)
    tl.store(U_PTR + row * WIDTH + hc_col, u, mask=valid)

    u_f = tl.where(valid, u.to(tl.float32), 0.0)
    gc = tl.load(GC_PTR + hc_col, mask=valid, other=0.0).to(tl.float32)
    u_norm = u_f * 1.0 / tl.sqrt(tl.sum(u_f * u_f, axis=0) / H + EPS) * (1.0 + gc)
    tl.store(UNORM_PTR + row * WIDTH + hc_col, u_norm, mask=valid)


@triton.jit(do_not_specialize=["TOTAL"])
def _ple_add3_kernel(
    H_PTR,
    U_PTR,
    CONV_PTR,
    OUT_PTR,
    TOTAL,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < TOTAL
    u = tl.load(U_PTR + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(CONV_PTR + offs, mask=mask, other=0.0).to(tl.float32)
    delta = (u + c).to(OUT_PTR.dtype.element_ty).to(tl.float32)
    h = tl.load(H_PTR + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT_PTR + offs, h + delta, mask=mask)


@triton.jit
def _ple_add3_hc_combine_norm_kernel(
    H_PTR,
    U_PTR,
    CONV_PTR,
    SUB_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    OUT_H_PTR,
    OUT_NORM_PTR,
    WIDTH: tl.constexpr,
    H: tl.constexpr,
    HC_COUNT: tl.constexpr,
    EPS: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < H
    hc_col = branch * H + col
    offs = row * WIDTH + hc_col

    u = tl.load(U_PTR + offs, mask=valid, other=0.0).to(tl.float32)
    c = tl.load(CONV_PTR + offs, mask=valid, other=0.0).to(tl.float32)
    delta = (u + c).to(OUT_H_PTR.dtype.element_ty).to(tl.float32)
    h = tl.load(H_PTR + offs, mask=valid, other=0.0).to(tl.float32)
    comb = (h + delta).to(OUT_H_PTR.dtype.element_ty)
    if HAS_COMBINE:
        sub = tl.load(SUB_PTR + row * H + col, mask=valid, other=0.0).to(tl.float32)
        inj = tl.load(INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
        comb = (comb.to(tl.float32) + sub * inj).to(OUT_H_PTR.dtype.element_ty)
    tl.store(OUT_H_PTR + offs, comb, mask=valid)

    comb_f = tl.where(valid, comb.to(tl.float32), 0.0)
    inv_rms = 1.0 / tl.sqrt(tl.sum(comb_f * comb_f, axis=0) / H + EPS)
    w = tl.load(W_PTR + hc_col, mask=valid, other=0.0).to(tl.float32)
    normed = comb_f * inv_rms * (1.0 + w)
    if HAS_BIAS:
        b = tl.load(B_PTR + hc_col, mask=valid, other=0.0).to(tl.float32)
        normed = normed + b
    tl.store(OUT_NORM_PTR + offs, normed, mask=valid)


# ---------------------------------------------------------------------------
# Functional Triton Wrappers
# ---------------------------------------------------------------------------


def ple_hash_ngram_ids_triton(
    input_ids: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    multipliers: torch.Tensor,
    eos_token_id: int,
    prev_context: Optional[torch.Tensor] = None,
    *,
    update_context_inplace: bool = False,
    stage_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused Triton 2-gram + 3-gram hashing with exact int64 & EOS reset semantics."""
    B, T = input_ids.shape
    num_heads = int(vocab_sizes.numel())
    heads_per_ngram = num_heads // 2
    ids_c = input_ids.to(torch.int64).contiguous()
    out = torch.empty((B, T, num_heads), device=ids_c.device, dtype=torch.int64)
    if B == 0 or T == 0:
        return out

    has_ctx = prev_context is not None
    if has_ctx:
        assert prev_context is not None
        if prev_context.device != ids_c.device or prev_context.dtype != torch.int64:
            raise ValueError("prev_context must be int64 on the same device as input_ids")
        if not prev_context.is_contiguous():
            raise ValueError("prev_context must be contiguous for Triton hash")

    has_stage = stage_context is not None
    update_t1 = has_ctx and update_context_inplace and (T == 1)
    head_block = max(16, triton.next_power_of_2(num_heads))
    block_rows = 1 if (B * T) <= 4 else (16 if (B * T) <= 64 else 64)

    _ple_hash_ngram_ids_kernel[(triton.cdiv(B * T, block_rows),)](
        prev_context if has_ctx else ids_c,
        ids_c,
        vocab_sizes.contiguous(),
        offsets.contiguous(),
        multipliers.contiguous(),
        out,
        stage_context if has_stage else ids_c,
        int(stage_context.stride(0)) if has_stage else 0,
        int(stage_context.stride(1)) if has_stage else 0,
        BATCH=B,
        SEQ=T,
        EOS=eos_token_id,
        NUM_HEADS=num_heads,
        HEADS_PER_NGRAM=heads_per_ngram,
        HEAD_BLOCK=head_block,
        BLOCK_ROWS=block_rows,
        HAS_CTX=has_ctx,
        UPDATE_CTX_T1=update_t1,
        WRITE_STAGE_CTX=has_stage,
        num_warps=4,
    )

    if has_ctx and update_context_inplace and T >= 2:
        assert prev_context is not None
        prev_context.copy_(ids_c[:, -2:])
    return out


def ple_gather_fp8_triton(
    ngram_ids: torch.Tensor,
    pointer_table: torch.Tensor,
    shard_rows: int,
    head_dim: int,
    weight_scale: torch.Tensor,
    *,
    is_fp8: bool = True,
    is_bf16: bool = False,
    is_cpu_uva: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather rows from CUDA or CPU-UVA sharded/flat table and dequantize FP8 -> out_dtype."""
    ids_c = ngram_ids.contiguous()
    total_shards = int(pointer_table.numel())
    n = ids_c.numel()
    out = torch.empty((*ids_c.shape, head_dim), device=ids_c.device, dtype=out_dtype)
    if n == 0:
        return out

    rows_per_cta = 4 if (is_cpu_uva or n <= 64) else 8
    bd = max(16, triton.next_power_of_2(head_dim))
    _ple_gather_shards_kernel[(triton.cdiv(n, rows_per_cta),)](
        ids_c,
        pointer_table,
        weight_scale if is_fp8 else pointer_table,
        out,
        N=n,
        SHARD_ROWS=shard_rows,
        TOTAL_SHARDS=total_shards,
        D=head_dim,
        BD=bd,
        BR=rows_per_cta,
        IS_FP8=is_fp8,
        IS_BF16=is_bf16,
        num_warps=4,
    )
    return out


def ple_gate_and_norm_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key: torch.Tensor,
    norm_query: torch.Tensor,
    norm_conv: torch.Tensor,
    hc_count: int,
    eps: float = 1e-6,
    *,
    combine_sub: Optional[torch.Tensor] = None,
    combine_inj: Optional[torch.Tensor] = None,
    out_combined: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused PLE GemmaRMSNorm(K), GemmaRMSNorm(H), signed-sqrt gate, U=alpha*V, GemmaRMSNorm(U)."""
    B, T, width = hidden.shape
    hidden_size = width // hc_count
    has_combine = combine_sub is not None and combine_inj is not None
    if not hidden.is_cuda or hidden_size > 8192:
        # Pure PyTorch fallback
        hid_eff = hidden
        if has_combine:
            assert combine_sub is not None and combine_inj is not None
            hid_eff = (
                hidden.reshape(B, T, hc_count, hidden_size).float()
                + combine_sub.reshape(B, T, 1, hidden_size).float()
                * combine_inj.reshape(B, T, hc_count, 1).float()
            ).flatten(-2).to(hidden.dtype)
            if out_combined is not None:
                out_combined.copy_(hid_eff)
        k_4 = key.reshape(B, T, hc_count, hidden_size).float()
        kn = (
            k_4
            * torch.rsqrt(k_4.square().mean(dim=-1, keepdim=True) + eps)
            * (1.0 + norm_key.reshape(1, 1, hc_count, hidden_size).float())
        ).to(key.dtype)
        h_4 = hid_eff.reshape(B, T, hc_count, hidden_size).float()
        qn = (
            h_4
            * torch.rsqrt(h_4.square().mean(dim=-1, keepdim=True) + eps)
            * (1.0 + norm_query.reshape(1, 1, hc_count, hidden_size).float())
        ).to(hidden.dtype)
        s = (kn.float() * qn.float()).sum(dim=-1, keepdim=True) / math.sqrt(hidden_size)
        alpha = torch.sigmoid(s.abs().clamp_min(1e-6).sqrt() * s.sign()).to(value.dtype)
        u = (alpha * value.unsqueeze(-2)).flatten(-2)
        u_4 = u.reshape(B, T, hc_count, hidden_size).float()
        u_norm = (
            u_4
            * torch.rsqrt(u_4.square().mean(dim=-1, keepdim=True) + eps)
            * (1.0 + norm_conv.reshape(1, 1, hc_count, hidden_size).float())
        ).flatten(-2).to(hidden.dtype)
        return u, u_norm

    # Flatten [B, T, ...] -> [B*T, ...] while preserving row strides for zero-copy split K/V
    if key.stride(-1) == 1 and (B == 1 or key.stride(0) == T * key.stride(1)):
        key_2d = key.view(B * T, width)
    else:
        key_2d = key.contiguous().view(B * T, width)
    if value.stride(-1) == 1 and (B == 1 or value.stride(0) == T * value.stride(1)):
        val_2d = value.view(B * T, hidden_size)
    else:
        val_2d = value.contiguous().view(B * T, hidden_size)
    hid_c = hidden.contiguous()
    hid_2d = hid_c.view(B * T, width)

    u = torch.empty_like(hid_c)
    u_norm = torch.empty_like(hid_c)
    if B * T == 0:
        if has_combine and out_combined is not None:
            out_combined.zero_()
        return u, u_norm

    if has_combine:
        assert combine_sub is not None and combine_inj is not None
        sub_2d = combine_sub.contiguous().view(B * T, hidden_size)
        inj_2d = combine_inj.contiguous().view(B * T, hc_count)
        comb_target = out_combined if out_combined is not None else torch.empty_like(hid_c)
        comb_2d = comb_target.view(B * T, width)
    else:
        sub_2d = hid_2d
        inj_2d = hid_2d
        comb_2d = hid_2d

    block = max(128, triton.next_power_of_2(hidden_size))
    num_warps = 4
    _ple_gate_and_norm_kernel[(B * T, hc_count)](
        key_2d,
        val_2d,
        hid_2d,
        norm_key.contiguous(),
        norm_query.contiguous(),
        norm_conv.contiguous(),
        u.view(B * T, width),
        u_norm.view(B * T, width),
        sub_2d,
        inj_2d,
        comb_2d,
        KEY_STRIDE=int(key_2d.stride(0)),
        VAL_STRIDE=int(val_2d.stride(0)),
        HID_STRIDE=int(hid_2d.stride(0)),
        COMB_OUT_STRIDE=int(comb_2d.stride(0)),
        WIDTH=width,
        H=hidden_size,
        HC_COUNT=hc_count,
        EPS=eps,
        HAS_COMBINE=has_combine,
        BLOCK=block,
        num_warps=num_warps,
    )
    return u, u_norm


def ple_add3_triton(
    hidden: torch.Tensor,
    u: torch.Tensor,
    conv_out: torch.Tensor,
) -> torch.Tensor:
    """Fused H + (U + conv_out) with exact intermediate BF16 rounding."""
    if not hidden.is_cuda:
        return hidden + (u + conv_out)
    h_c = hidden.contiguous()
    u_c = u.contiguous()
    c_c = conv_out.contiguous()
    out = torch.empty_like(h_c)
    total = h_c.numel()
    if total == 0:
        return out
    block = 1024
    _ple_add3_kernel[(triton.cdiv(total, block),)](
        h_c,
        u_c,
        c_c,
        out,
        TOTAL=total,
        BLOCK=block,
        num_warps=4,
    )
    return out


def ple_add3_and_hc_norm_triton(
    hidden: torch.Tensor,
    u: torch.Tensor,
    conv_out: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: Optional[torch.Tensor] = None,
    hc_count: int = 4,
    eps: float = 1e-6,
    *,
    combine_sub: Optional[torch.Tensor] = None,
    combine_inj: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused H + (U + conv_out) [+ post inj * sub] + next-HC per-branch GemmaRMSNorm."""
    width = int(hidden.shape[-1])
    if width % hc_count != 0:
        raise ValueError(f"Last dim {width} is not divisible by hc_count={hc_count}")
    hidden_size = width // hc_count
    has_combine = combine_sub is not None and combine_inj is not None

    if not hidden.is_cuda or hidden_size > 8192:
        comb = hidden + (u + conv_out)
        if has_combine:
            assert combine_sub is not None and combine_inj is not None
            comb = (
                comb.unflatten(-1, (hc_count, hidden_size)).float()
                + combine_sub.unsqueeze(-2).float() * combine_inj.unsqueeze(-1).float()
            ).flatten(-2).to(hidden.dtype)
        streams = comb.unflatten(-1, (hc_count, hidden_size)).float()
        normed_f = streams * torch.rsqrt(streams.square().mean(dim=-1, keepdim=True) + eps)
        normed = normed_f.flatten(-2) * (1.0 + norm_weight.float())
        if norm_bias is not None:
            normed = normed + norm_bias.float()
        return comb, normed.to(hidden.dtype)

    h_c = hidden.contiguous()
    u_c = u.contiguous()
    c_c = conv_out.contiguous()
    w_c = norm_weight.contiguous()
    b_c = norm_bias.contiguous() if norm_bias is not None else w_c
    rows = h_c.numel() // width
    out_h = torch.empty_like(h_c)
    out_norm = torch.empty_like(h_c)
    if rows == 0:
        return out_h, out_norm

    if has_combine:
        assert combine_sub is not None and combine_inj is not None
        sub_c = combine_sub.contiguous().view(rows, hidden_size)
        inj_c = combine_inj.contiguous().view(rows, hc_count)
    else:
        sub_c = h_c
        inj_c = h_c

    block = max(128, triton.next_power_of_2(hidden_size))
    num_warps = 8 if block >= 4096 else 4
    _ple_add3_hc_combine_norm_kernel[(rows, hc_count)](
        h_c,
        u_c,
        c_c,
        sub_c,
        inj_c,
        w_c,
        b_c,
        out_h,
        out_norm,
        WIDTH=width,
        H=hidden_size,
        HC_COUNT=hc_count,
        EPS=eps,
        HAS_COMBINE=has_combine,
        HAS_BIAS=norm_bias is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out_h, out_norm


# ---------------------------------------------------------------------------
# Module: Qwen4_5PLELayer
# ---------------------------------------------------------------------------


class Qwen4_5PLELayer(nn.Module):
    """Qwen 4.5-Flash Layer 1 Per-Layer N-gram Embedding (PLE) module."""

    def __init__(
        self,
        hidden_size: int = 5120,
        hc_count: int = 4,
        ple_embed_dim: int = 5120,
        ngram_size: int = 3,
        heads_per_ngram: int = 10,
        conv_kernel_size: int = 4,
        split_ngram_parts: int = 128,
        eos_token_id: int = 248044,
        eps: float = 1e-6,
        ngram_vocab_size_base: int = 35000000,
        layer_multipliers: Optional[Sequence[int]] = None,
        ngram_heads_vocab_sizes: Optional[Sequence[int]] = None,
        ngram_table: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        weight_scale: float = DEFAULT_PLE_WEIGHT_SCALE,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
    ) -> None:
        super().__init__()
        if ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {ngram_size}")
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_hidden_size = hc_count * hidden_size
        self.ple_embed_dim = ple_embed_dim
        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.ngram_heads = (ngram_size - 1) * heads_per_ngram
        if ple_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                f"ple_embed_dim ({ple_embed_dim}) must be divisible by "
                f"ngram_heads ({self.ngram_heads})"
            )
        self.head_dim = ple_embed_dim // self.ngram_heads
        self.conv_kernel_size = conv_kernel_size
        self.split_ngram_parts = split_ngram_parts
        self.eos_token_id = eos_token_id
        self.eps = eps

        if layer_multipliers is None:
            if ngram_size != len(DEFAULT_PLE_LAYER_MULTIPLIERS):
                raise ValueError(
                    "layer_multipliers must be provided when ngram_size != 3"
                )
            mult_values = DEFAULT_PLE_LAYER_MULTIPLIERS
        else:
            mult_values = tuple(int(v) for v in layer_multipliers)
            if len(mult_values) < ngram_size:
                raise ValueError("layer_multipliers length must be >= ngram_size")

        if ngram_heads_vocab_sizes is None:
            if (
                ngram_vocab_size_base == 35000000
                and self.ngram_heads == len(DEFAULT_PLE_NGRAM_VOCAB_SIZES)
            ):
                vocab_values = DEFAULT_PLE_NGRAM_VOCAB_SIZES
            else:
                vocab_values = compute_ngram_vocab_sizes(
                    int(ngram_vocab_size_base), self.ngram_heads
                )
        else:
            vocab_values = tuple(int(v) for v in ngram_heads_vocab_sizes)
            if len(vocab_values) != self.ngram_heads:
                raise ValueError(
                    f"ngram_heads_vocab_sizes len {len(vocab_values)} != {self.ngram_heads}"
                )

        offsets_list: List[int] = []
        running = 0
        for sz in vocab_values:
            offsets_list.append(running)
            running += sz
        self.total_vocab_size = running

        self.register_buffer(
            "layer_multipliers",
            torch.tensor(mult_values[:ngram_size], dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(vocab_values, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets_list, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.tensor(weight_scale, dtype=dtype, device=device),
        )

        # Projections, per-branch GemmaRMSNorms (no bias), and depthwise ShortConv (k=4)
        self.key_proj = nn.Linear(
            ple_embed_dim, self.hc_hidden_size, bias=False, dtype=dtype, device=device
        )
        self.value_proj = nn.Linear(
            ple_embed_dim, hidden_size, bias=False, dtype=dtype, device=device
        )
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)

        self.norm_key = nn.Parameter(
            torch.zeros(self.hc_hidden_size, dtype=dtype, device=device)
        )
        self.norm_query = nn.Parameter(
            torch.zeros(self.hc_hidden_size, dtype=dtype, device=device)
        )
        self.norm_conv = nn.Parameter(
            torch.zeros(self.hc_hidden_size, dtype=dtype, device=device)
        )
        self.sp_size = max(1, int(sp_size))
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group
        self.dp_attn = bool(dp_attn)
        self.conv1d = Qwen4_5ShortConv(
            channels=self.hc_hidden_size,
            kernel_size=conv_kernel_size,
            residual=False,
            use_activation=False,
            dtype=dtype,
            device=device,
            sp_size=self.sp_size,
            sp_rank=self.sp_rank,
            sp_group=self.sp_group,
            dp_attn=self.dp_attn,
        )

        # External n-gram table storage (intentionally NOT registered as Parameter/Buffer
        # so .to("cuda") keeps a 167 GiB CPU table safely on host RAM!)
        self._ngram_shards: Optional[List[torch.Tensor]] = None
        self._shard_rows: int = 0
        self._shard_offset: int = 0
        self._total_shards: int = 0
        self._ple_ep_group: Any = None
        self._table_pin_cpu: bool = False
        self._table_enable_cpu_uva: bool = True
        self._pointer_cache_device: Optional[torch.device] = None
        self._pointer_table: Optional[torch.Tensor] = None
        self._weight_scale_cuda: Optional[torch.Tensor] = None
        self._cpu_uva_active: bool = False

        self.tp_size = max(1, int(tp_size))
        self.tp_rank = int(tp_rank)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce

        # Inference cache for fused [key_proj.weight; value_proj.weight]
        self._fused_kv_key: Optional[Tuple[int, int, int, int]] = None
        self._fused_kv_weight: Optional[torch.Tensor] = None

        if ngram_table is not None:
            self.set_ngram_table(ngram_table)

    @property
    def ngram_table(self) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        if self._ngram_shards is None:
            return None
        if len(self._ngram_shards) == 1:
            return self._ngram_shards[0]
        return self._ngram_shards

    def set_ngram_table(
        self,
        table: Union[torch.Tensor, Sequence[torch.Tensor]],
        weight_scale: Optional[Union[float, torch.Tensor]] = None,
        *,
        pin_cpu: bool = False,
        enable_cpu_uva: bool = True,
        shard_offset: int = 0,
        total_shards: Optional[int] = None,
        ep_group: Any = None,
        fast_all_reduce: Any = None,
    ) -> None:
        """Attach a flat [total_rows, head_dim] or sharded list[Tensor] FP8/BF16 table.

        When sharded across EP ranks (``total_shards > len(shards)``), this rank
        owns global shards ``[shard_offset, shard_offset + len(shards))``. Remote
        shards map to pointer 0 (producing exact 0.0 in the gather kernel) and are
        combined via ``all_reduce(SUM)`` over ``ep_group``.
        """
        if isinstance(table, torch.Tensor):
            shards = [table.contiguous()]
        else:
            shards = [t.contiguous() for t in table]
        if not shards:
            raise ValueError("ngram_table must be non-empty")

        ref_shape = shards[0].shape
        ref_dtype = shards[0].dtype
        ref_device = shards[0].device
        if len(ref_shape) != 2 or ref_shape[1] != self.head_dim:
            raise ValueError(
                f"Expected shard shape [rows, {self.head_dim}], got {tuple(ref_shape)}"
            )
        if ref_dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported ngram_table dtype {ref_dtype}")
        for s in shards:
            if s.shape != ref_shape or s.dtype != ref_dtype or s.device != ref_device:
                raise ValueError("All ngram_table shards must share shape, dtype, and device")
        eff_total_shards = len(shards) if total_shards is None else int(total_shards)
        if shard_offset < 0 or shard_offset + len(shards) > eff_total_shards:
            raise ValueError(
                f"Invalid shard range [{shard_offset}, {shard_offset + len(shards)}) "
                f"for total_shards={eff_total_shards}"
            )
        capacity = ref_shape[0] * eff_total_shards
        if capacity < self.total_vocab_size:
            raise ValueError(
                f"ngram_table holds {capacity} rows, need at least {self.total_vocab_size}"
            )

        self._ngram_shards = shards
        self._shard_rows = int(ref_shape[0])
        self._shard_offset = int(shard_offset)
        self._total_shards = eff_total_shards
        self._ple_ep_group = ep_group
        if fast_all_reduce is not None:
            self.fast_all_reduce = fast_all_reduce
        self._table_pin_cpu = pin_cpu
        self._table_enable_cpu_uva = enable_cpu_uva
        self._pointer_cache_device = None
        self._pointer_table = None
        self._cpu_uva_active = False

        if weight_scale is not None:
            with torch.no_grad():
                if isinstance(weight_scale, torch.Tensor):
                    self.weight_scale.copy_(
                        weight_scale.to(
                            device=self.weight_scale.device,
                            dtype=self.weight_scale.dtype,
                        ).reshape(())
                    )
                else:
                    self.weight_scale.fill_(float(weight_scale))

    def alloc_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> PLELayerState:
        dev = device if device is not None else self.key_proj.weight.device
        dt = dtype if dtype is not None else self.key_proj.weight.dtype
        return PLELayerState(
            ngram_context=torch.full(
                (batch_size, self.context_len),
                self.eos_token_id,
                dtype=torch.int64,
                device=dev,
            ),
            conv_state=torch.zeros(
                (batch_size, self.hc_hidden_size, self.conv_kernel_size - 1),
                dtype=dt,
                device=dev,
            ),
        )

    def _ensure_pointer_table(self, target_device: torch.device) -> Optional[torch.Tensor]:
        assert self._ngram_shards is not None
        if self._pointer_cache_device == target_device:
            return self._pointer_table

        shards = self._ngram_shards
        self._pointer_cache_device = target_device
        self._pointer_table = None
        self._weight_scale_cuda = None
        self._cpu_uva_active = False

        if target_device.type != "cuda":
            return None
        self._weight_scale_cuda = self.weight_scale.to(device=target_device)

        table_device = shards[0].device
        if table_device == target_device:
            if self._total_shards > len(shards):
                ptrs = [0] * self._total_shards
                for idx, s in enumerate(shards):
                    ptrs[self._shard_offset + idx] = int(s.data_ptr())
            else:
                ptrs = [int(s.data_ptr()) for s in shards]
            self._pointer_table = torch.tensor(ptrs, dtype=torch.int64, device=target_device)
            return self._pointer_table

        if table_device.type == "cpu" and self._table_enable_cpu_uva:
            resolved = _resolve_cpu_uva_pointers(
                shards, target_device, pin_cpu=self._table_pin_cpu
            )
            if resolved is not None:
                self._ngram_shards, self._pointer_table = resolved
                self._cpu_uva_active = True
                return self._pointer_table
        return None

    def _get_fused_kv_weight(self) -> torch.Tensor:
        wk = self.key_proj.weight
        wv = self.value_proj.weight
        if torch.is_grad_enabled() and (wk.requires_grad or wv.requires_grad):
            return torch.cat([wk, wv], dim=0)
        key = (wk.data_ptr(), 0 if wk.is_inference() else wk._version, wv.data_ptr(), 0 if wv.is_inference() else wv._version)
        if self._fused_kv_key != key or self._fused_kv_weight is None:
            self._fused_kv_weight = torch.cat([wk, wv], dim=0).contiguous()
            self._fused_kv_key = key
        return self._fused_kv_weight

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        """Right-shift token sequence without crossing eos_token_id boundaries."""
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.int64)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        prev_eos_incl = torch.cummax(eos_positions, dim=1).values
        prev_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), prev_eos_incl[:, :-1]], dim=1
        )
        segment_start = prev_eos + 1
        pos_in_segment = positions.unsqueeze(0) - segment_start
        src_pos = positions - shift
        gather_pos = src_pos.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_pos)
        valid = (pos_in_segment >= shift) & (src_pos.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def compute_ngram_ids_ref(
        self,
        input_ids: torch.Tensor,
        state: Optional[PLELayerState] = None,
        *,
        update_state: bool = True,
    ) -> torch.Tensor:
        """Exact pure-PyTorch n-gram hashing via _shift_right_ignore_eos."""
        ids = input_ids.to(torch.int64)
        B, T = ids.shape
        if state is not None:
            prev_ctx = state.ngram_context.to(device=ids.device, dtype=torch.int64)
        else:
            prev_ctx = ids.new_full((B, self.context_len), self.eos_token_id)

        full_tokens = torch.cat([prev_ctx, ids], dim=1)  # [B, context_len + T]
        if state is not None and update_state and T > 0:
            state.ngram_context.copy_(full_tokens[:, -self.context_len :])

        shifted = [
            self._shift_right_ignore_eos(full_tokens, shift)[:, -T:]
            for shift in range(self.ngram_size)
        ]

        mult = self.layer_multipliers.to(device=ids.device)
        vocab = self.ngram_heads_vocab_sizes.to(device=ids.device)
        offsets = self.ngram_heads_offsets.to(device=ids.device)

        head_ids: List[torch.Tensor] = []
        running_mix = shifted[0] * mult[0]
        for order_idx in range(1, self.ngram_size):
            running_mix = running_mix ^ (shifted[order_idx] * mult[order_idx])
            start_h = (order_idx - 1) * self.heads_per_ngram
            end_h = order_idx * self.heads_per_ngram
            v_slice = vocab[start_h:end_h]
            o_slice = offsets[start_h:end_h]
            ids_order = torch.remainder(running_mix.unsqueeze(-1), v_slice) + o_slice
            head_ids.append(ids_order)
        return torch.cat(head_ids, dim=-1)

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        state: Optional[PLELayerState] = None,
        *,
        update_state: bool = True,
    ) -> torch.Tensor:
        """Compute [B, T, ngram_heads] hashed indices (Triton fast path on CUDA for 3-gram)."""
        if input_ids.is_cuda and self.ngram_size == 3:
            prev_ctx = state.ngram_context if state is not None else None
            return ple_hash_ngram_ids_triton(
                input_ids,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
                self.layer_multipliers,
                self.eos_token_id,
                prev_context=prev_ctx,
                update_context_inplace=(state is not None and update_state),
            )
        return self.compute_ngram_ids_ref(input_ids, state=state, update_state=update_state)

    def gather_embeddings(
        self,
        ngram_ids: torch.Tensor,
        *,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Gather [B, T, ngram_heads] -> [B, T, ple_embed_dim] from CUDA or CPU table."""
        if self._ngram_shards is None:
            raise RuntimeError(
                "Qwen4_5PLELayer.ngram_table is not set; call set_ngram_table() first."
            )
        B, T, H = ngram_ids.shape
        target_device = ngram_ids.device
        dtype = out_dtype if out_dtype is not None else self.key_proj.weight.dtype
        shards = self._ngram_shards
        is_fp8 = shards[0].dtype == torch.float8_e4m3fn
        is_bf16 = shards[0].dtype == torch.bfloat16

        ptr_table = self._ensure_pointer_table(target_device)
        if ptr_table is not None:
            scale = (
                self._weight_scale_cuda
                if self._weight_scale_cuda is not None
                and self._weight_scale_cuda.device == target_device
                else self.weight_scale.to(device=target_device)
            )
            gathered = ple_gather_fp8_triton(
                ngram_ids,
                ptr_table,
                self._shard_rows,
                self.head_dim,
                scale,
                is_fp8=is_fp8,
                is_bf16=is_bf16,
                is_cpu_uva=self._cpu_uva_active,
                out_dtype=dtype,
            )
            out = gathered.reshape(B, T, self.ple_embed_dim)
            if self._total_shards > len(shards):
                out = self._reduce_sharded_embeddings(out)
            return out

        # Fallback: host-side gather (keeps compact 1-byte FP8 over H2D) or generic device gather
        table_device = shards[0].device
        flat_ids = ngram_ids.reshape(-1)
        ids_on_table = flat_ids.to(table_device, non_blocking=False)
        if len(shards) == 1 and self._total_shards == 1:
            rows = shards[0].index_select(0, ids_on_table)
        else:
            shard_idx = ids_on_table // self._shard_rows
            local_row = ids_on_table % self._shard_rows
            rows = torch.zeros(
                (ids_on_table.numel(), self.head_dim),
                dtype=shards[0].dtype,
                device=table_device,
            )
            shard_end = self._shard_offset + len(shards)
            for s_id in shard_idx.unique().tolist():
                if self._shard_offset <= s_id < shard_end:
                    mask = shard_idx == s_id
                    rows[mask] = shards[s_id - self._shard_offset].index_select(
                        0, local_row[mask]
                    )

        if rows.device != target_device:
            rows = rows.to(target_device, non_blocking=True)
        if is_fp8:
            scale = self.weight_scale.to(device=target_device, dtype=dtype)
            rows = rows.to(dtype) * scale
        elif rows.dtype != dtype:
            rows = rows.to(dtype)
        out = rows.reshape(B, T, self.ple_embed_dim)
        if self._total_shards > len(shards):
            out = self._reduce_sharded_embeddings(out)
        return out

    def _reduce_sharded_embeddings(self, out: torch.Tensor) -> torch.Tensor:
        reduce_group = self._ple_ep_group if self._ple_ep_group is not None else self.tp_group
        if self.fast_all_reduce is not None:
            ar_group = getattr(self.fast_all_reduce, "group", reduce_group)
            if reduce_group is None or ar_group is reduce_group:
                return self.fast_all_reduce(out)
        if reduce_group is not None:
            torch.distributed.all_reduce(
                out, op=torch.distributed.ReduceOp.SUM, group=reduce_group
            )
        return out

    def _project_kv_fused(self, emb: torch.Tensor, *, use_fused_cuda: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_fused_cuda and self.hidden_size <= 8192:
            w_kv = self._get_fused_kv_weight()
            kv = F.linear(emb, w_kv)
            return kv[..., : self.hc_hidden_size], kv[..., self.hc_hidden_size :]
        return self.key_proj(emb), self.value_proj(emb)

    def forward_verify(
        self,
        hidden_states: torch.Tensor,
        input_ids_2d: torch.Tensor,
        init_ngram_ctx: torch.Tensor,
        init_conv_state: torch.Tensor,
        stage_ctx: Optional[torch.Tensor] = None,
        stage_conv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched V-step verify forward writing per-step rollback states in one pass."""
        B, V, W = hidden_states.shape
        if W != self.hc_hidden_size:
            raise ValueError(
                f"Expected hidden_states width {self.hc_hidden_size}, got {W}"
            )
        if B == 0 or V == 0:
            return hidden_states.clone()

        if input_ids_2d.is_cuda and self.ngram_size == 3:
            ngram_ids = ple_hash_ngram_ids_triton(
                input_ids_2d,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
                self.layer_multipliers,
                self.eos_token_id,
                prev_context=init_ngram_ctx,
                update_context_inplace=False,
                stage_context=stage_ctx,
            )
        else:
            ngram_ids = self.compute_ngram_ids_ref(
                input_ids_2d,
                state=PLELayerState(ngram_context=init_ngram_ctx.clone(), conv_state=init_conv_state),
                update_state=False,
            )
            if stage_ctx is not None:
                full = torch.cat([init_ngram_ctx, input_ids_2d.to(torch.int64)], dim=1)
                for step in range(V):
                    stage_ctx[:, step].copy_(full[:, step + 1 : step + 3])

        emb = self.gather_embeddings(ngram_ids, out_dtype=hidden_states.dtype)
        key, value = self._project_kv_fused(emb, use_fused_cuda=hidden_states.is_cuda)
        u, u_norm = ple_gate_and_norm_triton(
            key,
            value,
            hidden_states,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.hc_count,
            self.eps,
        )
        conv_out = causal_dwconv1d_verify_triton(
            u_norm,
            self.conv1d.weight,
            init_conv_state,
            stage_conv,
            residual=False,
            use_activation=False,
        )
        return ple_add3_triton(hidden_states, u, conv_out)

    def forward_packed_prefill(
        self,
        hidden_states: torch.Tensor,
        flat_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tok_req: torch.Tensor,
        tile_req: torch.Tensor,
        tile_t0: torch.Tensor,
        *,
        prev_ngram_ctx: Optional[torch.Tensor] = None,
        init_conv_state: Optional[torch.Tensor] = None,
        snap_tok_idx: Optional[torch.Tensor] = None,
        snap_req_idx: Optional[torch.Tensor] = None,
        conv_dst_pool: Optional[torch.Tensor] = None,
        conv_snap_phys: Optional[torch.Tensor] = None,
        ctx_dst_pool: Optional[torch.Tensor] = None,
        ctx_snap_phys: Optional[torch.Tensor] = None,
        skip_invalid_phys: bool = True,
        block_t: int = 16,
    ) -> torch.Tensor:
        """Whole-batch packed ragged prefill with boundary snapshot scatter."""
        ple_out, _ = self.forward_packed_prefill_with_hc(
            hidden_states,
            flat_input_ids,
            cu_seqlens,
            tok_req,
            tile_req,
            tile_t0,
            prev_ngram_ctx=prev_ngram_ctx,
            init_conv_state=init_conv_state,
            snap_tok_idx=snap_tok_idx,
            snap_req_idx=snap_req_idx,
            conv_dst_pool=conv_dst_pool,
            conv_snap_phys=conv_snap_phys,
            ctx_dst_pool=ctx_dst_pool,
            ctx_snap_phys=ctx_snap_phys,
            skip_invalid_phys=skip_invalid_phys,
            block_t=block_t,
        )
        return ple_out

    def forward_packed_prefill_with_hc(
        self,
        hidden_states: torch.Tensor,
        flat_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tok_req: torch.Tensor,
        tile_req: torch.Tensor,
        tile_t0: torch.Tensor,
        *,
        prev_ngram_ctx: Optional[torch.Tensor] = None,
        init_conv_state: Optional[torch.Tensor] = None,
        snap_tok_idx: Optional[torch.Tensor] = None,
        snap_req_idx: Optional[torch.Tensor] = None,
        conv_dst_pool: Optional[torch.Tensor] = None,
        conv_snap_phys: Optional[torch.Tensor] = None,
        ctx_dst_pool: Optional[torch.Tensor] = None,
        ctx_snap_phys: Optional[torch.Tensor] = None,
        skip_invalid_phys: bool = True,
        block_t: int = 16,
        pre_combine_sub: Optional[torch.Tensor] = None,
        pre_combine_inj: Optional[torch.Tensor] = None,
        post_combine_sub: Optional[torch.Tensor] = None,
        post_combine_inj: Optional[torch.Tensor] = None,
        hc_norm_weight: Optional[torch.Tensor] = None,
        hc_norm_bias: Optional[torch.Tensor] = None,
        hc_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Whole-batch packed ragged prefill optionally fused with pre/post HC ops."""
        N, W = hidden_states.shape
        if W != self.hc_hidden_size:
            raise ValueError(
                f"Expected hidden_states width {self.hc_hidden_size}, got {W}"
            )
        if N == 0:
            h_empty = hidden_states.clone()
            n_empty = torch.empty_like(h_empty) if hc_norm_weight is not None else None
            return h_empty, n_empty

        if flat_input_ids.is_cuda and self.ngram_size == 3:
            ids_c = flat_input_ids.to(torch.int64).contiguous()
            ngram_ids = torch.empty(
                (N, self.ngram_heads), device=ids_c.device, dtype=torch.int64
            )
            has_ctx = prev_ngram_ctx is not None
            head_block = max(16, triton.next_power_of_2(self.ngram_heads))
            block_rows = 1 if N <= 4 else (16 if N <= 64 else 64)
            _ple_hash_ngram_ids_ragged_kernel[(triton.cdiv(N, block_rows),)](
                prev_ngram_ctx.contiguous() if has_ctx else ids_c,
                ids_c,
                cu_seqlens,
                tok_req,
                self.ngram_heads_vocab_sizes.contiguous(),
                self.ngram_heads_offsets.contiguous(),
                self.layer_multipliers.contiguous(),
                ngram_ids,
                N=N,
                EOS=self.eos_token_id,
                NUM_HEADS=self.ngram_heads,
                HEADS_PER_NGRAM=self.heads_per_ngram,
                HEAD_BLOCK=head_block,
                BLOCK_ROWS=block_rows,
                HAS_CTX=has_ctx,
                num_warps=4,
            )
            if (
                snap_tok_idx is not None
                and snap_req_idx is not None
                and ctx_dst_pool is not None
                and ctx_snap_phys is not None
                and int(snap_tok_idx.numel()) > 0
            ):
                _scatter_ple_ctx_snapshots_ragged_kernel[(int(snap_tok_idx.numel()),)](
                    prev_ngram_ctx.contiguous() if has_ctx else ids_c,
                    ids_c,
                    cu_seqlens,
                    snap_tok_idx,
                    snap_req_idx,
                    ctx_snap_phys,
                    ctx_dst_pool,
                    int(ctx_dst_pool.stride(0)),
                    int(ctx_dst_pool.shape[0]),
                    EOS=self.eos_token_id,
                    HAS_CTX=has_ctx,
                    SKIP_INVALID_PHYS=skip_invalid_phys,
                    num_warps=1,
                )
        else:
            cu_list = cu_seqlens.tolist()
            ng_chunks = []
            for b in range(len(cu_list) - 1):
                s, e = int(cu_list[b]), int(cu_list[b + 1])
                if e <= s:
                    continue
                st_b = (
                    PLELayerState(
                        ngram_context=prev_ngram_ctx[b : b + 1].clone(),
                        conv_state=torch.empty(0),
                    )
                    if prev_ngram_ctx is not None
                    else None
                )
                ng_chunks.append(
                    self.compute_ngram_ids_ref(
                        flat_input_ids[s:e].unsqueeze(0), state=st_b, update_state=False
                    ).squeeze(0)
                )
            ngram_ids = torch.cat(ng_chunks, dim=0)
            if (
                snap_tok_idx is not None
                and snap_req_idx is not None
                and ctx_dst_pool is not None
                and ctx_snap_phys is not None
            ):
                for idx_s in range(int(snap_tok_idx.numel())):
                    phys = int(ctx_snap_phys[idx_s].item())
                    if skip_invalid_phys and (phys <= 0 or phys >= int(ctx_dst_pool.shape[0])):
                        continue
                    g_tok = int(snap_tok_idx[idx_s].item())
                    b = int(snap_req_idx[idx_s].item())
                    req_start = int(cu_list[b])
                    local_t = g_tok - req_start
                    c1 = int(flat_input_ids[g_tok].item())
                    if local_t >= 1:
                        c0 = int(flat_input_ids[g_tok - 1].item())
                    elif prev_ngram_ctx is not None:
                        c0 = int(prev_ngram_ctx[b, 1].item())
                    else:
                        c0 = self.eos_token_id
                    ctx_dst_pool[phys, 0] = c0
                    ctx_dst_pool[phys, 1] = c1

        emb = self.gather_embeddings(
            ngram_ids.unsqueeze(0), out_dtype=hidden_states.dtype
        )
        key, value = self._project_kv_fused(emb, use_fused_cuda=hidden_states.is_cuda)
        h_3d = hidden_states.unsqueeze(0)
        has_pre = pre_combine_sub is not None and pre_combine_inj is not None
        h_eff_3d = torch.empty_like(h_3d) if has_pre else h_3d
        u, u_norm = ple_gate_and_norm_triton(
            key,
            value,
            h_3d,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.hc_count,
            self.eps,
            combine_sub=pre_combine_sub.unsqueeze(0) if has_pre else None,
            combine_inj=pre_combine_inj.unsqueeze(0) if has_pre else None,
            out_combined=h_eff_3d if has_pre else None,
        )
        conv_out = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
            u_norm.squeeze(0),
            self.conv1d.weight,
            cu_seqlens,
            tile_req,
            tile_t0,
            init_state=init_conv_state,
            snap_tok_idx=snap_tok_idx,
            snap_req_idx=snap_req_idx,
            dst_pool=conv_dst_pool,
            snap_dst_idx=conv_snap_phys,
            skip_invalid_phys=skip_invalid_phys,
            residual=False,
            use_activation=False,
            block_t=block_t,
        )
        if hc_norm_weight is not None:
            has_post = post_combine_sub is not None and post_combine_inj is not None
            ple_out_3d, hc_normed_3d = ple_add3_and_hc_norm_triton(
                h_eff_3d,
                u,
                conv_out.unsqueeze(0),
                hc_norm_weight,
                hc_norm_bias,
                hc_count=self.hc_count,
                eps=hc_eps,
                combine_sub=post_combine_sub.unsqueeze(0) if has_post else None,
                combine_inj=post_combine_inj.unsqueeze(0) if has_post else None,
            )
            return ple_out_3d.squeeze(0), hc_normed_3d.squeeze(0)
        return ple_add3_triton(h_eff_3d, u, conv_out.unsqueeze(0)).squeeze(0), None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        state: Optional[PLELayerState] = None,
    ) -> torch.Tensor:
        """Compute H + delta where delta = U + conv1d(gemma_rmsnorm(U, norm_conv))."""
        B, T, W = hidden_states.shape
        if W != self.hc_hidden_size:
            raise ValueError(
                f"Expected hidden_states width {self.hc_hidden_size}, got {W}"
            )
        if T == 0:
            return hidden_states.clone()

        # 1. Hash token n-grams & gather FP8 embeddings
        ngram_ids = self.compute_ngram_ids(input_ids, state=state, update_state=True)
        emb = self.gather_embeddings(ngram_ids, out_dtype=hidden_states.dtype)

        # 2. Fused K/V projection (single GEMM, zero-copy strided split consumed by Triton gate)
        key, value = self._project_kv_fused(emb, use_fused_cuda=hidden_states.is_cuda)

        # 3. Fused norm_key + norm_query + signed-sqrt dot gate + U + norm_conv
        u, u_norm = ple_gate_and_norm_triton(
            key,
            value,
            hidden_states,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.hc_count,
            self.eps,
        )

        # 4. Depthwise causal conv1d (k=4, residual=False, use_activation=False)
        conv_out = self.conv1d(u_norm, state=state)

        # 5. Return H + (U + conv_out)
        return ple_add3_triton(hidden_states, u, conv_out)

    def forward_ref(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        state: Optional[PLELayerState] = None,
    ) -> torch.Tensor:
        """Exact pure-PyTorch FP32 reference matching the Qwen 4.5-Flash PLE specification."""
        B, T, _ = hidden_states.shape
        out_dtype = hidden_states.dtype
        if T == 0:
            return hidden_states.clone()

        ngram_ids = self.compute_ngram_ids_ref(input_ids, state=state, update_state=True)
        # Reference gather in PyTorch
        assert self._ngram_shards is not None
        shards = self._ngram_shards
        table_device = shards[0].device
        flat_ids = ngram_ids.reshape(-1).to(table_device)
        if len(shards) == 1:
            rows = shards[0].index_select(0, flat_ids)
        else:
            s_idx = flat_ids // self._shard_rows
            l_row = flat_ids % self._shard_rows
            rows = torch.empty(
                (flat_ids.numel(), self.head_dim),
                dtype=shards[0].dtype,
                device=table_device,
            )
            for s_id in s_idx.unique().tolist():
                mask = s_idx == s_id
                rows[mask] = shards[s_id].index_select(0, l_row[mask])
        rows = rows.to(hidden_states.device)
        if rows.dtype == torch.float8_e4m3fn:
            emb = (rows.to(out_dtype) * self.weight_scale.to(device=hidden_states.device, dtype=out_dtype)).reshape(
                B, T, self.ple_embed_dim
            )
        else:
            emb = rows.to(out_dtype).reshape(B, T, self.ple_embed_dim)

        K = F.linear(emb.float(), self.key_proj.weight.float()).to(out_dtype).reshape(
            B, T, self.hc_count, self.hidden_size
        )
        V = F.linear(emb.float(), self.value_proj.weight.float()).to(out_dtype)

        def _gemma_rms(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            t_f = t.float()
            n = t_f * torch.rsqrt(t_f.square().mean(dim=-1, keepdim=True) + self.eps)
            return (n * (1.0 + w.reshape(1, 1, self.hc_count, self.hidden_size).float())).to(out_dtype)

        Kn = _gemma_rms(K, self.norm_key)
        Qn = _gemma_rms(hidden_states.reshape(B, T, self.hc_count, self.hidden_size), self.norm_query)

        s = (Kn.float() * Qn.float()).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        alpha = torch.sigmoid(s.abs().clamp_min(1e-6).sqrt() * s.sign()).to(out_dtype)
        U = (alpha * V.unsqueeze(-2)).flatten(-2)
        U_norm = _gemma_rms(
            U.reshape(B, T, self.hc_count, self.hidden_size), self.norm_conv
        ).flatten(-2)

        conv_out = self.conv1d.forward_ref(U_norm, state=state)
        delta = U + conv_out
        return hidden_states + delta
