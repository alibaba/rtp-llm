"""CUDA kernels with a fixed reduction order for each token."""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "x_scale_row_stride", "x_scale_col_stride"])
def _linear(
    x,
    weight,
    bias,
    output,
    rows,
    cols: tl.constexpr,
    inner: tl.constexpr,
    weight_row_stride: tl.constexpr,
    weight_col_stride: tl.constexpr,
    bias_stride: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    x_scales=None,
    weight_scales=None,
    x_scale_row_stride=0,
    x_scale_col_stride=0,
    weight_scale_row_stride: tl.constexpr = 0,
    weight_scale_col_stride: tl.constexpr = 0,
    packed_scales: tl.constexpr = False,
):
    m = tl.program_id(0).to(tl.int64) * block_m + tl.arange(0, block_m)
    n = tl.program_id(1).to(tl.int64) * block_n + tl.arange(0, block_n)
    k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), tl.float32)
    # Every output tile reduces the complete K dimension in the same order.
    for tile in range(tl.cdiv(inner, block_k)):
        index = tile * block_k + k
        a = tl.load(
            x + m[:, None] * inner + index[None, :],
            (m[:, None] < rows) & (index[None, :] < inner),
            other=0.0,
        )
        b = tl.load(
            weight
            + n[None, :] * weight_row_stride
            + index[:, None] * weight_col_stride,
            (n[None, :] < cols) & (index[:, None] < inner),
            other=0.0,
        )
        if x_scales is not None:
            group = tile
            scale_col = group // 4 if packed_scales else group
            scale_row = n if packed_scales else n // 128
            sx = tl.load(
                x_scales + m * x_scale_row_stride + scale_col * x_scale_col_stride,
                m < rows,
                other=0,
            )
            sw = tl.load(
                weight_scales
                + scale_row * weight_scale_row_stride
                + scale_col * weight_scale_col_stride,
                n < cols,
                other=0,
            )
            if packed_scales:
                shift = (group % 4) * 8
                sx = (((sx >> shift) & 0xFF) << 23).to(tl.float32, bitcast=True)
                sw = (((sw >> shift) & 0xFF) << 23).to(tl.float32, bitcast=True)
            partial = tl.dot(a.to(tl.float16), b.to(tl.float16))
            acc += partial * (sx[:, None] * sw[None, :])
        else:
            acc = tl.dot(a, b, acc, input_precision="ieee")
    if has_bias:
        acc += tl.load(bias + n * bias_stride, n < cols, other=0)[None, :]
    tl.store(
        output + m[:, None] * cols + n[None, :],
        acc,
        (m[:, None] < rows) & (n[None, :] < cols),
    )


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply an [out_features, in_features] weight to independent input rows."""
    if not x.is_cuda or weight.device != x.device:
        raise ValueError(
            "batch-invariant linear requires tensors on the same CUDA device"
        )
    if x.dtype != weight.dtype or x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("batch-invariant linear requires matching FP16, BF16 or FP32")
    inner = weight.size(1)
    if inner <= 0 or x.shape[-1] != inner:
        raise ValueError("batch-invariant linear input and weight dimensions differ")
    return _launch_linear(x, weight, bias, out_dtype or x.dtype)


def _launch_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    x_scales: torch.Tensor | None = None,
    weight_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    cols, inner = weight.shape
    if bias is not None:
        if bias.numel() != cols or bias.device != x.device:
            raise ValueError("batch-invariant linear bias must match the weight output")
        bias = bias.reshape(cols)
    rows = x.numel() // inner
    output = torch.empty((*x.shape[:-1], cols), device=x.device, dtype=out_dtype)
    if rows and cols:
        block_m, block_n = 16, 64
        block_k = 128 if x_scales is not None else 32
        scale_options = (
            {
                "x_scales": x_scales,
                "weight_scales": weight_scales,
                "x_scale_row_stride": x_scales.stride(0),
                "x_scale_col_stride": x_scales.stride(1),
                "weight_scale_row_stride": weight_scales.stride(0),
                "weight_scale_col_stride": weight_scales.stride(1),
                "packed_scales": weight_scales.dtype == torch.int32,
            }
            if x_scales is not None
            else {}
        )
        with torch.cuda.device(x.device) if x.is_cuda else nullcontext():
            _linear[(triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))](
                x.contiguous(),
                weight,
                bias,
                output,
                rows,
                cols,
                inner,
                weight.stride(0),
                weight.stride(1),
                bias.stride(0) if bias is not None else 0,
                bias is not None,
                block_m,
                block_n,
                block_k,
                num_warps=4,
                **scale_options,
            )
    return output


def fp8_per_tensor_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # Static scaling is shared by all rows. Dynamic scaling must derive its
    # maximum from each token independently.
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if weight_scale.numel() != 1 or (
        input_scale is not None and input_scale.numel() != 1
    ):
        raise ValueError("per-tensor FP8 linear requires scalar checkpoint scales")
    if input_scale is None:
        input_scale = (
            rows.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 448
        )
    quantized = (rows.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    # FP8 values are exactly representable in FP16. Keep the FP32 accumulator.
    output = linear(quantized.half(), weight.half(), out_dtype=torch.float32)
    output *= input_scale * weight_scale
    if bias is not None:
        output += bias
    return output.to(out_dtype or x.dtype).reshape(*x.shape[:-1], weight.size(0))


def quantize_fp8_blocks(
    rows: torch.Tensor, packed: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )
    from rtp_llm.ops.compute_ops import per_token_group_quant_fp8_v2

    rows = rows.contiguous()
    if rows.size(1) % 128 != 0:
        raise ValueError("batch-invariant block FP8 requires K divisible by 128")
    quantized = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    scales = create_per_token_group_quant_fp8_output_scale(
        rows.shape,
        rows.device,
        128,
        column_major_scales=packed,
        scale_tma_aligned=packed,
        scale_ue8m0=packed,
    )
    # Use the same quantizer for every batch size. Each scale sees one token.
    if rows.size(0):
        per_token_group_quant_fp8_v2(
            rows, quantized, scales, 128, 1e-4, -448.0, 448.0, packed, False, None
        )
    return quantized, scales


def fp8_per_block_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape[-1] != weight.size(1):
        raise ValueError("batch-invariant linear input and weight dimensions differ")
    rows = x.reshape(-1, x.shape[-1])
    quantized, scales = quantize_fp8_blocks(rows, weight_scales.dtype == torch.int32)
    output = _launch_linear(quantized, weight, bias, x.dtype, scales, weight_scales)
    return output.reshape(*x.shape[:-1], weight.size(0))


def lm_head(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None
) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise ValueError("batch-invariant FP8 LM head requires a weight scale")
        return fp8_per_tensor_linear(x, weight, scale, out_dtype=torch.float32)
    return linear(x.to(weight.dtype), weight, out_dtype=torch.float32)


@triton.jit
def _rms_norm(
    x,
    weight,
    output,
    size: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, block)
    values = tl.load(x + row * size + col, col < size, other=0).to(tl.float32)
    gamma = tl.load(weight + col, col < size, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(values * values, 0) / size + eps)
    tl.store(output + row * size + col, values * scale * gamma, col < size)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.contiguous()
    if x.numel() == 0:
        return x
    size = x.size(-1)
    output = torch.empty_like(x)
    _rms_norm[(x.numel() // size,)](
        x, weight.contiguous(), output, size, eps, triton.next_power_of_2(size)
    )
    return output


@triton.jit(do_not_specialize=["batch", "rows", "table_stride", "table_width"])
def _prepare_attention_tokens(
    query_offsets,
    prefixes_or_lengths,
    block_table,
    layout,
    rows,
    batch,
    table_stride,
    table_width,
    page_size: tl.constexpr,
    prefill: tl.constexpr,
    paged: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0) * block + tl.arange(0, block)
    if prefill:
        lo = tl.full((block,), 0, tl.int32)
        hi = tl.full((block,), batch, tl.int32)
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            end = tl.load(query_offsets + mid + 1, lo < hi, other=0)
            advance = (lo < hi) & (end <= token)
            lo = tl.where(advance, mid + 1, lo)
            hi = tl.where((lo < hi) & ~advance, mid, hi)
        sequence = lo
        active = (token < rows) & (sequence < batch)
        start = tl.load(query_offsets + sequence, active, other=0)
        end = tl.load(query_offsets + sequence + 1, active, other=0)
        prefix = tl.load(prefixes_or_lengths + sequence, active, other=0)
        position = prefix + token - start
    else:
        sequence = token
        active = (token < rows) & (sequence < batch)
        start, end = token, token + 1
        position = tl.load(prefixes_or_lengths + sequence, active, other=0) - 1

    page = tl.full((block,), 0, tl.int32)
    if paged:
        page = tl.load(
            block_table + sequence.to(tl.int64) * table_stride + position // page_size,
            active & (position >= 0) & (position // page_size < table_width),
            other=0,
        )
        # RTP reserves page zero for padding and fake streams.
        active = active & (page > 0)
    tl.store(layout + token * 5, sequence, token < rows)
    tl.store(layout + token * 5 + 1, start, token < rows)
    tl.store(layout + token * 5 + 2, end, token < rows)
    tl.store(layout + token * 5 + 3, tl.where(active, position, -1), token < rows)
    tl.store(layout + token * 5 + 4, page, token < rows)


def attention_token_layout(
    query_offsets: torch.Tensor | None,
    prefixes_or_lengths: torch.Tensor,
    block_table: torch.Tensor | None,
    rows: int,
    page_size: int,
) -> torch.Tensor:
    """Map packed queries to their request, query range, position and cache page."""
    layout = prefixes_or_lengths.new_empty((rows, 5))
    if rows:
        _prepare_attention_tokens[(triton.cdiv(rows, 128),)](
            query_offsets,
            prefixes_or_lengths,
            block_table,
            layout,
            rows,
            prefixes_or_lengths.numel(),
            block_table.stride(0) if block_table is not None else 0,
            block_table.size(1) if block_table is not None else 0,
            page_size,
            query_offsets is not None,
            block_table is not None,
            128,
        )
    return layout


@triton.jit
def _rope(
    tensor,
    layout,
    cos_sin,
    position_ids,
    row_stride: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    has_position_ids: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    position = tl.load(layout + token * 5 + 3)
    if position >= 0:
        if has_position_ids:
            position = tl.load(position_ids + token)
        index = tl.arange(0, block)
        valid = index < half_dim
        cache_offset = position.to(tl.int64) * (2 * half_dim) + index
        cos = tl.load(cos_sin + cache_offset, valid, other=0)
        sin = tl.load(cos_sin + cache_offset + half_dim, valid, other=0)
        offset = token.to(tl.int64) * row_stride + tl.program_id(1) * head_dim + index
        first = tl.load(tensor + offset, valid, other=0).to(tl.float32)
        second = tl.load(tensor + offset + half_dim, valid, other=0).to(tl.float32)
        tl.store(tensor + offset, first * cos - second * sin, valid)
        tl.store(tensor + offset + half_dim, second * cos + first * sin, valid)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: torch.Tensor,
    cos_sin: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> None:
    rows = query.size(0)
    if position_ids is not None and position_ids.numel() == 0:
        position_ids = None
    if position_ids is not None and position_ids.numel() != rows:
        raise ValueError("batch-invariant RoPE requires one position ID per token")
    if rows:
        half_dim = cos_sin.size(1) // 2
        for tensor in (query, key):
            _rope[(rows, tensor.size(1))](
                tensor,
                layout,
                cos_sin,
                position_ids,
                tensor.stride(0),
                tensor.size(2),
                half_dim,
                position_ids is not None,
                triton.next_power_of_2(half_dim),
            )


@triton.jit
def _append_kv(
    key,
    value,
    cache,
    scales,
    layout,
    key_row_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    page_stride: tl.constexpr,
    kv_stride: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    kv_heads: tl.constexpr,
    store_scale: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    position = tl.load(layout + token * 5 + 3)
    if position >= 0:
        page = tl.load(layout + token * 5 + 4).to(tl.int64)
        dim = tl.arange(0, block)
        cache_offset = (
            page * page_stride + (head * page_size + position % page_size) * head_dim
        )
        k = tl.load(
            key + token.to(tl.int64) * key_row_stride + head * head_dim + dim,
            dim < head_dim,
            other=0,
        )
        v = tl.load(
            value + token.to(tl.int64) * value_row_stride + head * head_dim + dim,
            dim < head_dim,
            other=0,
        )
        tl.store(cache + cache_offset + dim, k, dim < head_dim)
        tl.store(cache + cache_offset + kv_stride + dim, v, dim < head_dim)
        if store_scale:
            scale_offset = (
                page * 2 * kv_heads + head
            ) * page_size + position % page_size
            tl.store(scales + scale_offset, 1.0)
            tl.store(scales + scale_offset + kv_heads * page_size, 1.0)


def append_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    layout: torch.Tensor,
    scales: torch.Tensor | None = None,
) -> None:
    rows, heads, head_dim = key.shape
    if rows:
        _append_kv[(rows, heads)](
            key,
            value,
            cache,
            scales,
            layout,
            key.stride(0),
            value.stride(0),
            cache.stride(0),
            cache.stride(1),
            head_dim,
            cache.size(3),
            heads,
            scales is not None,
            triton.next_power_of_2(head_dim),
        )


@triton.jit(do_not_specialize=["table_stride"])
def _attention(
    query,
    key,
    value,
    cache,
    block_table,
    layout,
    output,
    table_stride,
    query_row_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    page_stride: tl.constexpr,
    kv_stride: tl.constexpr,
    query_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    scale: tl.constexpr,
    causal: tl.constexpr,
    paged: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_head = tl.program_id(2) * block_h + tl.arange(0, block_h)
    q_head = kv_head * group_size + group_head
    dim = tl.arange(0, block_d)
    valid_output = (group_head[:, None] < group_size) & (dim[None, :] < head_dim)
    out = (
        output
        + (token.to(tl.int64) * query_heads + q_head[:, None]) * head_dim
        + dim[None, :]
    )
    position = tl.load(layout + token * 5 + 3)
    if position < 0:
        tl.store(out, 0, valid_output)
    else:
        sequence = tl.load(layout + token * 5)
        start = tl.load(layout + token * 5 + 1)
        if causal:
            length = position + 1
        else:
            length = tl.load(layout + token * 5 + 2) - start
        q = tl.load(
            query
            + token.to(tl.int64) * query_row_stride
            + q_head[:, None] * head_dim
            + dim[None, :],
            valid_output,
            other=0,
        )
        maximum = tl.full((block_h,), float("-inf"), tl.float32)
        total = tl.zeros((block_h,), tl.float32)
        acc = tl.zeros((block_h, block_d), tl.float32)
        # Use the same sequence-aligned tiles for prefill and decode. No
        # split-KV or merge depends on other requests or the query chunk size.
        for tile in range(tl.cdiv(length, block_n)):
            index = tile * block_n + tl.arange(0, block_n)
            valid = (index[:, None] < length) & (dim[None, :] < head_dim)
            if paged:
                page = tl.load(
                    block_table
                    + sequence.to(tl.int64) * table_stride
                    + index // page_size,
                    index < length,
                    other=0,
                ).to(tl.int64)
                offset = (
                    page * page_stride
                    + (kv_head * page_size + index % page_size) * head_dim
                )
                k = tl.load(cache + offset[:, None] + dim[None, :], valid, other=0.0)
                v = tl.load(
                    cache + offset[:, None] + kv_stride + dim[None, :], valid, other=0.0
                )
            else:
                row = (start + index).to(tl.int64)
                k = tl.load(
                    key
                    + row[:, None] * key_row_stride
                    + kv_head * head_dim
                    + dim[None, :],
                    valid,
                    other=0,
                )
                v = tl.load(
                    value
                    + row[:, None] * value_row_stride
                    + kv_head * head_dim
                    + dim[None, :],
                    valid,
                    other=0,
                )
            k = k.to(q.dtype)
            v = v.to(q.dtype)
            score = tl.dot(q, tl.trans(k)) * scale
            score = tl.where(index[None, :] < length, score, float("-inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, 1))
            correction = tl.exp(maximum - next_maximum)
            probability = tl.exp(score - next_maximum[:, None])
            total = total * correction + tl.sum(probability, 1)
            acc = tl.dot(probability.to(v.dtype), v, acc * correction[:, None])
            maximum = next_maximum
        tl.store(out, acc / total[:, None], valid_output)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: torch.Tensor,
    cache: torch.Tensor | None,
    block_table: torch.Tensor | None,
    *,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    rows, query_heads, head_dim = query.shape
    kv_heads = key.size(1)
    group_size = query_heads // kv_heads
    output = torch.empty(
        (rows, query_heads * head_dim), device=query.device, dtype=query.dtype
    )
    if rows:
        block_h = 16
        _attention[(rows, kv_heads, triton.cdiv(group_size, block_h))](
            query,
            key,
            value,
            cache,
            block_table,
            layout,
            output,
            block_table.stride(0) if block_table is not None else 0,
            query.stride(0),
            key.stride(0),
            value.stride(0),
            cache.stride(0) if cache is not None else 0,
            cache.stride(1) if cache is not None else 0,
            query_heads,
            group_size,
            head_dim,
            cache.size(3) if cache is not None else 0,
            scale,
            causal,
            cache is not None,
            block_h,
            32,
            max(32, triton.next_power_of_2(head_dim)),
            num_warps=4,
        )
    return output
