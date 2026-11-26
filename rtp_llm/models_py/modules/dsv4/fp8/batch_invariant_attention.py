"""Fixed-order sliding-window and compressed attention for DeepSeek V4."""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_cache_tile(pool, slots, dim, valid, stride, page_size, entry_bytes):
    if entry_bytes == 0:
        dense_row = (pool + slots.to(tl.int64) * stride).to(
            tl.pointer_type(tl.bfloat16)
        )
        result = tl.load(dense_row[None, :] + dim[:, None], valid[None, :], other=0)
    else:
        page = slots // page_size
        row = slots % page_size
        block = pool + page.to(tl.int64) * stride
        if entry_bytes == 528:
            data = block + row * 528
            raw = tl.load(data[None, :] + dim[:, None], valid[None, :], other=0)
            exponent = tl.load(
                data[None, :] + 512 + dim[:, None] // 32, valid[None, :], other=127
            )
            value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            result = (value * tl.exp2(exponent.to(tl.float32) - 127.0)).to(tl.bfloat16)
        elif entry_bytes == 288:
            data = block + row * 288
            raw = tl.load(data[None, :] + dim[:, None] // 2, valid[None, :], other=0)
            code = (raw >> ((dim[:, None] & 1) * 4)) & 15
            magnitude = code & 7
            normal = tl.exp2((magnitude >> 1).to(tl.float32) - 1.0)
            normal *= 1.0 + (magnitude & 1).to(tl.float32) * 0.5
            value = tl.where(magnitude < 2, magnitude.to(tl.float32) * 0.5, normal)
            value = tl.where((code & 8) != 0, -value, value)
            scale = tl.load(
                data[None, :] + 256 + dim[:, None] // 16, valid[None, :], other=0
            )
            scale = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            result = (value * scale).to(tl.bfloat16)
        else:
            # V4 pages store 576 data bytes per token followed by an 8-byte
            # scale plane per token.
            data = block + row * 576
            scales = block + page_size * 576 + row * 8
            nope = (dim[:, None] < 448) & valid[None, :]
            rope = (dim[:, None] >= 448) & valid[None, :]
            raw = tl.load(data[None, :] + dim[:, None], nope, other=0)
            exponent = tl.load(scales[None, :] + dim[:, None] // 64, nope, other=127)
            scale = tl.exp2(exponent.to(tl.float32) - 127.0)
            latent = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
            position = tl.load(
                (data + 448).to(tl.pointer_type(tl.bfloat16))[None, :]
                + (dim[:, None] - 448),
                rope,
                other=0,
            )
            result = tl.where(dim[:, None] < 448, latent.to(tl.bfloat16), position)
    return result


@triton.jit(
    do_not_specialize=[
        "primary_width",
        "extra_width",
        "query_per_request",
        "pool_stride",
        "extra_stride",
        "page_size",
        "extra_page_size",
        "entry_bytes",
        "extra_entry_bytes",
        "has_lengths",
        "has_extra_lengths",
    ]
)
def _indexed_attention(
    query,
    pool,
    indices,
    lengths,
    extra_pool,
    extra_indices,
    extra_lengths,
    sink,
    output,
    heads: tl.constexpr,
    primary_width,
    extra_width,
    query_per_request,
    pool_stride,
    extra_stride,
    page_size,
    extra_page_size,
    entry_bytes,
    extra_entry_bytes,
    scale: tl.constexpr,
    has_lengths,
    has_extra_lengths,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * block_h + tl.arange(0, block_h)
    dim = tl.arange(0, 512)
    vector_offset = (token * heads + head[:, None]) * 512 + dim[None, :]
    q = tl.load(query + vector_offset, head[:, None] < heads, other=0)

    request = token // query_per_request
    primary_count = tl.minimum(
        tl.load(lengths + request, has_lengths != 0, other=primary_width),
        primary_width,
    )
    extra_count = tl.minimum(
        tl.load(extra_lengths + request, has_extra_lengths != 0, other=extra_width),
        extra_width,
    )

    # The learned sink contributes a zero value with its own score.
    if sink is not None:
        maximum = tl.load(sink + head, head < heads, other=0).to(tl.float32)
        total = tl.full((block_h,), 1.0, tl.float32)
    else:
        maximum = tl.full((block_h,), float("-inf"), tl.float32)
        total = tl.zeros((block_h,), tl.float32)
    accumulator = tl.zeros((block_h, 512), tl.float32)
    for tile in tl.range(0, primary_count + extra_count, block_n):
        offset = tile + tl.arange(0, block_n)
        # Match the prefill workspace's [compressed | sliding-window] order,
        # including tiles which straddle the two pools.
        primary_offset = offset - extra_count
        primary_valid = (primary_offset >= 0) & (primary_offset < primary_count)
        primary_slot = tl.load(
            indices + token * primary_width + primary_offset,
            primary_valid,
            other=-1,
        )
        primary_valid = primary_valid & (primary_slot >= 0)

        key = _load_cache_tile(
            pool,
            primary_slot,
            dim,
            primary_valid,
            pool_stride,
            page_size,
            entry_bytes,
        )
        if extra_count > 0:
            extra_valid = offset < extra_count
            extra_slot = tl.load(
                extra_indices + token * extra_width + offset,
                extra_valid,
                other=-1,
            )
            extra_valid = extra_valid & (extra_slot >= 0)
            extra_key = _load_cache_tile(
                extra_pool,
                extra_slot,
                dim,
                extra_valid,
                extra_stride,
                extra_page_size,
                extra_entry_bytes,
            )
            key = tl.where(extra_valid[None, :], extra_key, key)
            primary_valid = primary_valid | extra_valid
        score = tl.dot(q, key) * scale
        score = tl.where(primary_valid[None, :], score, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=1))
        base = tl.where(next_max == float("-inf"), 0.0, next_max)
        correction = tl.exp(maximum - base)
        prob = tl.exp(score - base[:, None])
        total = total * correction + tl.sum(prob, axis=1)
        accumulator *= correction[:, None]
        accumulator += tl.dot(prob.to(key.dtype), tl.trans(key))
        maximum = next_max
    result = accumulator / tl.where(total > 0, total, 1.0)[:, None]
    tl.store(output + vector_offset, result, head[:, None] < heads)


def indexed_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    sink: torch.Tensor | None,
    scale: float,
    *,
    extra_kv: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accept a BF16 prefill workspace or the typed V4/V4.1 decode pools."""
    heads, dim = query.shape[-2:]
    query_per_request = query.size(1) if query.dim() == 4 else 1
    query = query.contiguous()
    rows = query.numel() // (heads * dim)
    primary_width = indices.size(-1)
    packed = kv.dtype == torch.uint8
    if (
        dim != 512
        or query.dtype != torch.bfloat16
        or (packed and kv.shape[-1] not in (528, 584))
        or (extra_kv is not None and extra_kv.shape[-1] not in (288, 584))
        or (not packed and (kv.dtype != query.dtype or kv.shape[-1] != dim))
    ):
        raise ValueError("unsupported batch-invariant DeepSeek V4 attention layout")
    output = torch.empty_like(query)

    if rows:
        # Keep the contraction in one specialization. Layout is a runtime
        # property of the byte-addressed loads, even for BF16 workspaces.
        indices = indices.to(dtype=torch.int32).contiguous()
        pool = kv.view(torch.uint8)
        extra_pool = extra_kv if extra_kv is not None else pool
        _indexed_attention[(rows, triton.cdiv(heads, 16))](
            query,
            pool,
            indices,
            (
                lengths.to(dtype=torch.int32).contiguous()
                if lengths is not None
                else indices
            ),
            extra_pool,
            (
                extra_indices.to(dtype=torch.int32).contiguous()
                if extra_indices is not None
                else indices
            ),
            (
                extra_lengths.to(dtype=torch.int32).contiguous()
                if extra_lengths is not None
                else indices
            ),
            sink,
            output,
            heads,
            primary_width,
            extra_indices.size(-1) if extra_indices is not None else 0,
            query_per_request,
            pool.stride(0),
            extra_pool.stride(0),
            kv.size(1) if packed else 1,
            extra_pool.size(1),
            kv.size(-1) if packed else 0,
            extra_kv.size(-1) if extra_kv is not None else 0,
            scale,
            int(lengths is not None),
            int(extra_lengths is not None),
            16,
            32,
            num_warps=4,
        )
    return output
