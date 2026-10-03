"""M3.1 raw Gemma norm and cached NeoX RoPE, in-place.

Selected by M3.1 attention when all four raw checkpoint norms are available.
FP32 normalization continues through rotation; only the final store is BF16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_norm_rope(
    QKV,
    IQ,
    IK,
    WQ,
    WK,
    WIQ,
    WIK,
    POS,
    CACHE,
    QS: tl.constexpr,
    IQS: tl.constexpr,
    IKS: tl.constexpr,
    PS: tl.constexpr,
    CS: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HI: tl.constexpr,
    EPS: tl.constexpr,
):
    # CP4 at 1M tokens has ~250K rows. The fused projection row stride is
    # 9856 elements, so row * stride crosses INT32 at row 217886. Widen BEFORE
    # multiplication for every projection/position pointer, not afterwards.
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    if head < HQ:
        ptr = QKV + row * QS + head * 128 + d
        w = tl.load(WQ + d).to(tl.float32)
    elif head < HQ + HK:
        ptr = QKV + row * QS + head * 128 + d
        w = tl.load(WK + d).to(tl.float32)
    elif head < HQ + HK + HI:
        ptr = IQ + row * IQS + (head - HQ - HK) * 128 + d
        w = tl.load(WIQ + d).to(tl.float32)
    else:
        ptr = IK + row * IKS + d
        w = tl.load(WIK + d).to(tl.float32)
    x = tl.load(ptr).to(tl.float32)
    # Match the demo's lane assignment, including its non-rotary adjacent
    # pairs and descending XOR butterfly. A generic 128-element tl.sum has
    # a different FP32 addition tree at rounding boundaries.
    lane = tl.arange(0, 32)
    x0 = tl.gather(x, lane, 0)
    x1 = tl.gather(x, lane + 32, 0)
    x2 = tl.gather(x, 2 * lane + 64, 0)
    x3 = tl.gather(x, 2 * lane + 65, 0)
    rotary = x0 * x0 + x1 * x1
    plain = x2 * x2 + x3 * x3
    for shift in tl.static_range(5):
        mask = 16 >> shift
        rotary = rotary + tl.gather(rotary, lane ^ mask, 0)
        plain = plain + tl.gather(plain, lane ^ mask, 0)
    totals = rotary + plain
    total = tl.gather(totals, d % 32, 0)
    n = (x * tl.rsqrt(tl.fma(total, 1.0 / 128.0, EPS))) * (1.0 + w)
    other = tl.gather(n, tl.where(d < 32, d + 32, tl.where(d < 64, d - 32, d)), 0)
    position = tl.load(POS + row * PS).to(tl.int64)
    c = tl.load(CACHE + position * CS + d % 32)
    s = tl.load(CACHE + position * CS + 32 + d % 32)
    out = tl.where(
        d < 32, tl.fma(n, c, -other * s), tl.where(d < 64, tl.fma(n, c, other * s), n)
    )
    tl.store(ptr, out)


def minimax_m31_gemma_norm_rope_(
    qkv,
    index_q,
    index_k,
    raw_weights,
    positions,
    cos_sin_cache,
    *,
    num_q_heads,
    num_kv_heads,
    num_index_heads,
    eps=1e-6
):
    """Normalize/rotate projected BF16 rows; leave V untouched.

    QKV is [T,(Hq+2*Hkv)*128], index Q/K are [T,Hi*128]/[T,128].
    All four weights are raw checkpoint [128] BF16 (not pre-added gamma).
    Reuses the existing FP32 [positions,64] cos/sin cache. No output/workspace
    allocation, no host read/sync; caller must validate position bounds before
    capture, and rewrite projections before every replay.
    """
    heads = (num_q_heads, num_kv_heads, num_index_heads)
    if any(not isinstance(h, int) or h <= 0 for h in heads) or eps <= 0:
        raise ValueError("positive integer head counts and positive eps required")
    if qkv.ndim != 2:
        raise ValueError("QKV must be 2-D")
    rows = qkv.shape[0]
    shapes = (
        (rows, (num_q_heads + 2 * num_kv_heads) * 128),
        (rows, num_index_heads * 128),
        (rows, 128),
    )
    for x, shape in zip((qkv, index_q, index_k), shapes):
        if x.shape != shape or x.dtype != torch.bfloat16 or x.stride(-1) != 1:
            raise ValueError("projected BF16 shapes/contiguous channels required")
        if rows and x.stride(0) < shape[1]:
            raise ValueError("projection rows must not overlap")
    if len(raw_weights) != 4 or any(
        w.shape != (128,) or w.dtype != torch.bfloat16 or not w.is_contiguous()
        for w in raw_weights
    ):
        raise ValueError("four raw contiguous BF16 norm weights required")
    if positions.shape != (rows,) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("integer positions [T] required")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("FP32 NeoX cos/sin cache [N,64] required")
    tensors = (qkv, index_q, index_k, *raw_weights, positions, cos_sin_cache)
    if not qkv.is_cuda or any(x.device != qkv.device for x in tensors):
        raise ValueError("all tensors must share a CUDA device")
    if rows:
        _gemma_norm_rope[(rows, sum(heads) + 1)](
            qkv,
            index_q,
            index_k,
            *raw_weights,
            positions,
            cos_sin_cache,
            qkv.stride(0),
            index_q.stride(0),
            index_k.stride(0),
            positions.stride(0),
            cos_sin_cache.stride(0),
            *heads,
            eps,
            num_warps=4,
            enable_fp_fusion=False
        )
