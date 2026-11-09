"""PPU Prefill FP4 scores with native BF16 product and head reduction.

The 64 MiB bound applies to dot/product scratch; caller chunking bounds logits.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.ppu.moe.ppu_mxfp4 import (
    gemm_fp4_fp4_bf16_nt,
    prepare_fp4_weight_scale_mxfp4,
)

DOT_BYTE_LIMIT = 64 * 1024**2


@triton.jit
def _relu_weight(dot, weight, numel, columns: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    valid = idx < numel
    value = tl.load(dot + idx, valid, other=0).to(tl.float32)
    scale = tl.load(weight + idx // columns, valid, other=0).to(tl.float32)
    # Clamp before multiply. The store rounds the product to BF16 before sum.
    clamped = tl.where(value < 0.0, 0.0, value)
    tl.store(dot + idx, clamped * scale, valid)


def _relu_weight_(dot, weight):
    assert dot.ndim == 3 and dot.is_contiguous()
    assert tuple(weight.shape) == tuple(dot.shape[:2]) and weight.is_contiguous()
    assert str(dot.dtype) == "torch.bfloat16" and weight.dtype == dot.dtype
    assert weight.device == dot.device
    block = 1024
    _relu_weight[(triton.cdiv(dot.numel(), block),)](
        dot, weight, dot.numel(), dot.shape[-1], block
    )


@triton.jit
def _store_masked_sum(
    scores,
    result,
    starts,
    ends,
    rows,
    keys,
    width,
    start_stride,
    end_stride,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = idx // keys
    col = idx % keys
    inside = row < rows
    begin = tl.load(starts + row * start_stride, inside, other=0)
    end = tl.load(ends + row * end_stride, inside, other=0)
    valid = inside & (col < width) & (col >= begin) & (col < end)
    value = tl.load(scores + row * width + col, valid, other=0)
    masked = tl.where(valid, value, tl.full((BLOCK,), float("-inf"), tl.bfloat16))
    tl.store(result + idx, masked, inside)


def _rows(rows, heads, keys, scratch_bytes):
    if not 0 < scratch_bytes <= DOT_BYTE_LIMIT:
        raise ValueError("scratch_bytes must be positive and at most64MiB")
    cap = scratch_bytes // (2 * heads * keys)
    if cap < 1:
        raise ValueError("One full-head/full-K query exceeds bounded FP4 score scratch")
    return min(rows, cap)


def _prepared(packed, scale):
    if (
        packed.dtype != torch.int8
        or scale.dtype != torch.int32
        or packed.shape[-1] != 64
    ):
        raise ValueError(
            "FP4 score requires original int8 packed K128 and int32 UE8M0 scales"
        )
    if (
        not packed.is_contiguous()
        or not scale.is_contiguous()
        or tuple(scale.shape) != tuple(packed.shape[:-1]) + (1,)
    ):
        raise ValueError("FP4 score payload/scale layout mismatch")
    rows = packed.numel() // 64
    data = packed.view(torch.uint8).reshape(rows, 64)
    raw = scale.view(torch.uint8).reshape(rows, 4)
    return (data, prepare_fp4_weight_scale_mxfp4(raw))


def _plan_causal_tiles(ends, heads, keys, scratch_bytes=DOT_BYTE_LIMIT, quantum=4):
    """Increase query tiles at narrow causal widths, retaining the scratch bound."""
    if keys <= 0 or keys % 1024 or heads not in (32, 64):
        raise ValueError("unqualified geometry")
    if any(x < 0 or x > keys for x in ends) or any(
        a > b for a, b in zip(ends, ends[1:])
    ):
        raise ValueError("requires monotone request-local causal ends")
    if scratch_bytes < 2 * heads * keys:
        raise ValueError("scratch cannot hold one full-width query")
    result = []
    start = 0
    while start < len(ends):
        lo, hi = start + 1, len(ends)
        while lo <= hi:
            mid = (lo + hi) // 2
            width = ((ends[mid - 1] + 1023) // 1024) * 1024
            if (mid - start) * 2 * heads * width <= scratch_bytes:
                lo = mid + 1
            else:
                hi = mid - 1
        count = hi - start
        if count >= quantum:
            count = (count // quantum) * quantum
        end = start + count
        width = ((ends[end - 1] + 1023) // 1024) * 1024
        assert end > start and (end - start) * heads * width * 2 <= scratch_bytes
        result.append((start, end, width))
        start = end
    return result


@triton.jit
def _q3_bf16_product(dot, weight, row, head, col, valid, WIDTH: tl.constexpr):
    # The old graph casts weights before the multiply and rounds each BF16
    # product before the FP32 sum. ReLU preserves input signed zero and NaN.
    x = tl.load(dot + (row * 64 + head) * WIDTH + col, valid, other=0).to(tl.float32)
    w = tl.load(weight + row * 64 + head).to(tl.float32)
    w = w.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    x = tl.where(x < 0.0, 0.0, x)
    return (x * w).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _fused_q3_score(
    dot,
    weight,
    starts,
    ends,
    out,
    start_stride,
    end_stride,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1) * BLOCK
    col = tile + tl.arange(0, BLOCK)
    start = tl.load(starts + row * start_stride)
    end = tl.load(ends + row * end_stride)
    valid = (col >= start) & (col < end)
    if (tile < end) & (tile + BLOCK > start) & (end > start):
        # Torch Reduce.cuh: four warps y, each processes head y + 4*i.
        # vt0=4 interleaves four FP32 accumulators in each warp, then
        # combines the accumulators left-to-right. Do not use tl.sum here.
        warp = tl.arange(0, 4)[:, None]
        keys = col[None, :]
        active = valid[None, :]
        zero = tl.full((4, BLOCK), 0.0, tl.float32)
        v0 = zero + _q3_bf16_product(dot, weight, row, warp, keys, active, WIDTH)
        v1 = zero + _q3_bf16_product(dot, weight, row, warp + 4, keys, active, WIDTH)
        v2 = zero + _q3_bf16_product(dot, weight, row, warp + 8, keys, active, WIDTH)
        v3 = zero + _q3_bf16_product(dot, weight, row, warp + 12, keys, active, WIDTH)
        for j in tl.static_range(1, 4):
            base = warp + j * 16
            v0 = v0 + _q3_bf16_product(dot, weight, row, base, keys, active, WIDTH)
            v1 = v1 + _q3_bf16_product(dot, weight, row, base + 4, keys, active, WIDTH)
            v2 = v2 + _q3_bf16_product(dot, weight, row, base + 8, keys, active, WIDTH)
            v3 = v3 + _q3_bf16_product(dot, weight, row, base + 12, keys, active, WIDTH)
        partial = ((v0 + v1) + v2) + v3
        # block_y_reduce applies offsets 2, then 1: (w0+w2)+(w1+w3).
        even, odd = tl.split(tl.reshape(tl.trans(partial), (BLOCK, 2, 2)))
        w0, w2 = tl.split(even)
        w1, w3 = tl.split(odd)
        value = (w0 + w2) + (w1 + w3)
        masked = tl.where(valid, value, float("-inf")).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        tl.store(out + row * WIDTH + col, masked)
    else:
        tl.store(out + row * WIDTH + col, float("-inf"))


def _score_grouped_q3_out(dot, weight, starts, ends, out):
    """Only the sealed grouped request-local Q3 [3,64,12288] geometry."""
    assert dot.is_cuda and torch.cuda.get_device_name(dot.device) == "ZW-M890P"
    rows, heads, width = 3, 64, 12288
    assert (
        dot.dtype == torch.bfloat16
        and dot.shape == (rows, heads, width)
        and dot.is_contiguous()
    )
    assert weight.shape == (rows, heads) and weight.is_contiguous()
    assert weight.dtype in (torch.float32, torch.bfloat16)
    assert all(
        bound.dtype == torch.int32 and bound.shape == (rows,)
        for bound in (starts, ends)
    )
    assert (
        out.dtype == torch.bfloat16
        and out.shape == (rows, width)
        and out.is_contiguous()
    )
    assert all(tensor.device == dot.device for tensor in (weight, starts, ends, out))
    block = 128
    _fused_q3_score[(rows, width // block)](
        dot,
        weight,
        starts,
        ends,
        out,
        starts.stride(0),
        ends.stride(0),
        width,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def fp8_fp4_mqa_logits(
    q,
    kv,
    weights,
    cu_seq_len_k_start,
    cu_seq_len_k_end,
    clean_logits=False,
    max_seqlen_k=0,
    logits_dtype=torch.float32,
    *,
    scratch_bytes=DOT_BYTE_LIMIT,
    trim_causal_prefix=False,
    out=None,
    _prepared_k=False,
    _prepared_q=None
):
    """Compute nonpaged BF16 MQA scores from the original FP4 payload.

    Invalid columns are initialized -inf, a stronger defined contract than old
    clean_logits=False. Callers compare only valid columns against old DG output.
    """
    if logits_dtype != torch.bfloat16 or max_seqlen_k not in (None, 0):
        raise ValueError("FP4 score requires BF16 uncompressed/full-width output")
    (packed, scale) = q
    (kpacked, kscale) = kv
    if packed.ndim != 3 or kpacked.ndim != 2 or weights.shape != packed.shape[:2]:
        raise ValueError("FP4 score Q/K/weight shape mismatch")
    (rows, heads, _) = packed.shape
    keys = kpacked.shape[0]
    if (
        heads not in (32, 64)
        or keys <= 0
        or weights.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("Unsupported FP4 score shape/dtype")
    if any(
        (
            v.device != packed.device
            for v in [
                scale,
                kpacked,
                kscale,
                weights,
                cu_seq_len_k_start,
                cu_seq_len_k_end,
            ]
        )
    ):
        raise ValueError("FP4 score device mismatch")
    if any(
        (
            v.dtype != torch.int32 or tuple(v.shape) != (rows,)
            for v in [cu_seq_len_k_start, cu_seq_len_k_end]
        )
    ):
        raise ValueError("FP4 score per-query bounds must be int32[rows]")
    if out is None:
        result = torch.empty((rows, keys), dtype=torch.bfloat16, device=packed.device)
    else:
        if (
            trim_causal_prefix
            or keys % 2
            or out.shape != (rows, keys)
            or out.dtype != torch.bfloat16
            or out.device != packed.device
            or not out.is_contiguous()
            or not packed.is_cuda
            or torch.cuda.get_device_name(packed.device) != "ZW-M890P"
        ):
            raise ValueError(
                "FP4 decode score output must be contiguous BF16 at even full width on M890P"
            )
        result = out
    if rows == 0:
        return result
    # The installed no-bias FP4 epilogue requires even physical N.
    physical_keys = keys + keys % 2
    if physical_keys != keys:
        kpacked = torch.cat((kpacked, torch.zeros_like(kpacked[:1])), dim=0)
        kscale = torch.cat((kscale, kscale[:1]), dim=0)
    step = _rows(rows, heads, physical_keys, scratch_bytes)
    if _prepared_k:
        assert (rows, heads, keys) == (3, 64, 12288) and out is not None
        assert not trim_causal_prefix and physical_keys == keys
        assert kpacked.dtype == torch.int8 and kpacked.is_contiguous()
        assert kscale.dtype == torch.uint16 and kscale.shape == (keys, 2)
        assert kscale.stride() == (1, keys)
        prepared_keys = {physical_keys: (kpacked.view(torch.uint8), kscale)}
    else:
        prepared_keys = {physical_keys: _prepared(kpacked, kscale)}
    fused_mask = out is not None or (
        trim_causal_prefix and torch.cuda.get_device_name(packed.device) == "ZW-M890P"
    )
    columns = None if fused_mask else torch.arange(keys, device=packed.device)
    host_ends = None
    if (
        trim_causal_prefix
        and keys > 1024
        and rows > step
        and not torch.cuda.is_current_stream_capturing()
    ):
        # One bounded readback per caller chunk. Bucket the widths to limit
        # DeepGEMM JIT shapes while keeping the original score/TopK dimensions.
        host_ends = cu_seq_len_k_end.tolist()
    tiles = None
    if host_ends is not None and fused_mask and keys % 1024 == 0:
        try:
            candidate_tiles = _plan_causal_tiles(host_ends, heads, keys, scratch_bytes)
        except ValueError:
            pass
        else:
            if len(candidate_tiles) <= triton.cdiv(rows, step):
                tiles = candidate_tiles
    if tiles is None:
        tiles = []
        for start in range(0, rows, step):
            end = min(rows, start + step)
            width = physical_keys
            if host_ends is not None:
                max_visible = max(0, max(host_ends[start:end]))
                width = min(physical_keys, ((max_visible + 1023) // 1024) * 1024)
            tiles.append((start, end, width))
    for start, end, width in tiles:
        if width == 0:
            result[start:end].fill_(float("-inf"))
            continue
        if width not in prepared_keys:
            prepared_keys[width] = _prepared(kpacked[:width], kscale[:width])
        if _prepared_q is not None:
            assert (start, end, heads) == (0, 3, 64)
            assert _prepared_q.shape == (192, 2) and _prepared_q.stride() == (1, 192)
            prepared_q = (packed.view(torch.uint8).reshape(192, 64), _prepared_q)
        else:
            prepared_q = _prepared(packed[start:end], scale[start:end])
        dot = gemm_fp4_fp4_bf16_nt(prepared_q, prepared_keys[width]).view(
            end - start, heads, width
        )
        if (
            out is not None
            and weights.is_contiguous()
            and dot.is_contiguous()
            and (rows, end - start, heads, width) == (3, 3, 64, 12288)
        ):
            _score_grouped_q3_out(
                dot,
                weights[start:end],
                cu_seq_len_k_start[start:end],
                cu_seq_len_k_end[start:end],
                result[start:end],
            )
            continue
        # Keep the BF16 product before Torch accumulates the head sum.
        weight = weights[start:end].to(torch.bfloat16)
        if trim_causal_prefix and dot.numel() >= 1024 * 1024:
            _relu_weight_(dot, weight.contiguous())
        else:
            dot.relu_().mul_(weight.unsqueeze(-1))
        visible_width = min(keys, width)
        visible = result[start:end, :visible_width]
        summed = dot.sum(dim=1)
        del dot, prepared_q
        if fused_mask:
            # Keep Torch's BF16 reduction; fuse only its output copy and bounds mask.
            _store_masked_sum[(triton.cdiv((end - start) * keys, 256),)](
                summed,
                result[start:end],
                cu_seq_len_k_start[start:end],
                cu_seq_len_k_end[start:end],
                end - start,
                keys,
                width,
                cu_seq_len_k_start.stride(0),
                cu_seq_len_k_end.stride(0),
                256,
            )
        else:
            visible.copy_(summed[:, :visible_width])
            valid = (
                columns[None, :visible_width] >= cu_seq_len_k_start[start:end, None]
            ) & (columns[None, :visible_width] < cu_seq_len_k_end[start:end, None])
            visible.masked_fill_(~valid, float("-inf"))
            if visible_width != keys:
                result[start:end, visible_width:].fill_(float("-inf"))
    return result


def native_topk(scores, starts, ends, out=None, *, topk=512, _fuse_restore=False):
    """Native Torch topk order, then bounds masking and caller-width -1 padding."""
    if scores.ndim != 2 or scores.dtype != torch.bfloat16 or topk <= 0:
        raise ValueError("Native Prefill topk requires BF16[M,T] and positive K")
    (rows, keys) = scores.shape
    if out is None:
        out = torch.empty((rows, topk), device=scores.device, dtype=torch.int32)
    if (
        out.dtype != torch.int32
        or tuple(out.shape) != (rows, topk)
        or out.device != scores.device
    ):
        raise ValueError("Native topk output must be int32[M,K] on score device")
    if any(
        (
            v.dtype != torch.int32
            or tuple(v.shape) != (rows,)
            or v.device != scores.device
            for v in [starts, ends]
        )
    ):
        raise ValueError("Native topk bounds must be int32[M] on score device")
    if _fuse_restore:
        from rtp_llm.models_py.kernels.ppu.attention.ppu_fp4_topk_finish import (
            native_topk_restore,
        )

        return native_topk_restore(scores, starts, ends, out, topk=topk)
    count = min(topk, keys)
    out.fill_(-1)
    if not count or not rows:
        return out
    # Preserve native TopK tie order and its [1, rows, keys] call shape.
    selected = scores.unsqueeze(0).topk(count, dim=-1).indices[0]
    valid = (selected >= starts[:, None]) & (selected < ends[:, None])
    out[:, :count].copy_(selected.masked_fill(~valid, -1))
    return out
