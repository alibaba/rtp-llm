"""M3.1 raw Gemma norm and cached NeoX RoPE, in-place.

Selected by M3.1 attention when all four raw checkpoint norms are available.
FP32 normalization continues through rotation; only the final store is BF16.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
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
    ROWS,
    GROUP: tl.constexpr,
    OQ=None,
    OK=None,
    OIQ=None,
    OIK=None,
    WRITE_CONTIGUOUS: tl.constexpr = False,
    Q8=None,
    IQ8=None,
    WRITE_Q8: tl.constexpr = False,
):
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    # Amortize CTA overhead only for large prefill batches. Keep the original
    # one-dimensional gather and FP32 reduction/FMA order for each row.
    for offset in tl.static_range(GROUP):
        row = tl.program_id(0).to(tl.int64) * GROUP + offset
        # Live token count changes on every mixed-length Prefill batch. It is
        # only a bounds mask, not tile geometry: share the compiled kernel
        # across counts while retaining GROUP/head/stride specialization.
        valid = row < ROWS
        # CP4 at 1M tokens has ~250K rows. The fused projection row stride is
        # 9856 elements, so row * stride crosses INT32 at row 217886. Widen BEFORE
        # multiplication for every projection/position pointer, not afterwards.
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
        x = tl.load(ptr, mask=valid, other=0).to(tl.float32)
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
        position = tl.load(POS + row * PS, mask=valid, other=0).to(tl.int64)
        c = tl.load(CACHE + position * CS + d % 32, mask=valid, other=0)
        s = tl.load(CACHE + position * CS + 32 + d % 32, mask=valid, other=0)
        out = tl.where(
            d < 32,
            tl.fma(n, c, -other * s),
            tl.where(d < 64, tl.fma(n, c, other * s), n),
        )
        tl.store(ptr, out, mask=valid)
        if WRITE_Q8:
            # Match the existing BF16 store followed by scale-one E4M3 cast.
            # Casting FP32 directly would skip a rounding boundary.
            rounded = out.to(tl.bfloat16).to(tl.float8e4nv)
            if head < HQ:
                tl.store(Q8 + row * (HQ * 128) + head * 128 + d, rounded, mask=valid)
            elif head >= HQ + HK and head < HQ + HK + HI:
                tl.store(
                    IQ8 + row * (HI * 128) + (head - HQ - HK) * 128 + d,
                    rounded,
                    mask=valid,
                )
        if WRITE_CONTIGUOUS:
            if head < HQ:
                dst = OQ + row * (HQ * 128) + head * 128 + d
            elif head < HQ + HK:
                dst = OK + row * (HK * 128) + (head - HQ) * 128 + d
            elif head < HQ + HK + HI:
                dst = OIQ + row * (HI * 128) + (head - HQ - HK) * 128 + d
            else:
                dst = OIK + row * 128 + d
            tl.store(dst, out, mask=valid)


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
    eps=1e-6,
    contiguous_outputs=None,
    query_fp8_outputs=None,
):
    """Normalize/rotate projected BF16 rows; leave V untouched.

    QKV is [T,(Hq+2*Hkv)*128], index Q/K are [T,Hi*128]/[T,128].
    All four weights are raw checkpoint [128] BF16 (not pre-added gamma).
    Reuses the existing FP32 [positions,64] cos/sin cache. The optional caller-
    owned contiguous Q/K/index-Q/index-K outputs replace subsequent copies;
    original projections are still mutated identically and V is untouched.
    No internal allocation or host read/sync. Caller validates position bounds
    before capture and rewrites projections before every replay.
    Optional independent E4M3 Q/index-Q outputs fuse the existing scale-one
    cast after BF16 rounding; K/index-K/V and in-place projections are unchanged.
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
    if query_fp8_outputs is not None:
        if contiguous_outputs is not None:
            raise ValueError("choose contiguous BF16 outputs or Q8 outputs, not both")
        output_shapes = ((rows, num_q_heads, 128), (rows, num_index_heads, 128))
        if len(query_fp8_outputs) != 2 or any(
            x.shape != shape
            or x.dtype != torch.float8_e4m3fn
            or x.device != qkv.device
            or not x.is_contiguous()
            for x, shape in zip(query_fp8_outputs, output_shapes)
        ):
            raise ValueError("two contiguous E4M3 Q/index-Q outputs required")
        inputs = {x.untyped_storage().data_ptr() for x in tensors}
        outputs = [x.untyped_storage().data_ptr() for x in query_fp8_outputs]
        if rows and (len(set(outputs)) != 2 or inputs.intersection(outputs)):
            raise ValueError("Q8 outputs must have independent non-input storage")
    if contiguous_outputs is not None:
        output_shapes = (
            (rows, num_q_heads, 128),
            (rows, num_kv_heads, 128),
            (rows, num_index_heads, 128),
            (rows, 1, 128),
        )
        if len(contiguous_outputs) != 4 or any(
            x.shape != shape
            or x.dtype != torch.bfloat16
            or x.device != qkv.device
            or not x.is_contiguous()
            for x, shape in zip(contiguous_outputs, output_shapes)
        ):
            raise ValueError(
                "four contiguous BF16 Q/K/index-Q/index-K outputs required"
            )
        # Independent storage prevents cross-head CTAs from overwriting inputs
        # or other outputs before their readers complete.
        inputs = {x.untyped_storage().data_ptr() for x in tensors}
        outputs = [x.untyped_storage().data_ptr() for x in contiguous_outputs]
        if rows and (len(set(outputs)) != 4 or inputs.intersection(outputs)):
            raise ValueError(
                "contiguous outputs must have independent non-input storage"
            )
    if rows:
        # In-place Prefill producers benefit from grouping at medium row
        # counts too. Keep small decode/verify and contiguous-output producers
        # on their existing threshold; grouping needs no extra tensors.
        group = (
            8 if rows >= 32768 or (rows >= 2048 and contiguous_outputs is None) else 1
        )
        # One warp avoids cross-warp gather/layout conversions for grouped
        # 128-channel heads. Keep small decode/verify and contiguous stores on
        # their existing launch configuration.
        num_warps = 1 if group == 8 and contiguous_outputs is None else 4
        _gemma_norm_rope[(triton.cdiv(rows, group), sum(heads) + 1)](
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
            rows,
            group,
            OQ=contiguous_outputs[0] if contiguous_outputs is not None else None,
            OK=contiguous_outputs[1] if contiguous_outputs is not None else None,
            OIQ=contiguous_outputs[2] if contiguous_outputs is not None else None,
            OIK=contiguous_outputs[3] if contiguous_outputs is not None else None,
            WRITE_CONTIGUOUS=contiguous_outputs is not None,
            Q8=query_fp8_outputs[0] if query_fp8_outputs is not None else None,
            IQ8=query_fp8_outputs[1] if query_fp8_outputs is not None else None,
            WRITE_Q8=query_fp8_outputs is not None,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
