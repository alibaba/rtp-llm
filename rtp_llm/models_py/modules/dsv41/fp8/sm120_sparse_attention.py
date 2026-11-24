"""Bounded SM120 sparse attention using the existing V4.1 cache codecs.

Keep the joint online softmax used by the Flash SM120 chunk fallback, while
reading V4.1's token-interleaved 528-byte SWA and 288-byte GLOBAL rows.
"""

import torch


def _lengths(lengths, rows, width, device):
    if lengths is None:
        return torch.full((rows,), width, dtype=torch.int64, device=device)
    result = lengths.to(device=device, dtype=torch.int64).reshape(-1)
    if result.numel() != rows:
        if not result.numel() or rows % result.numel():
            raise ValueError("Sparse lengths do not match the query rows")
        result = result.repeat_interleave(rows // result.numel())
    return result.clamp(0, width)


def _attention(query, sinks, pools, scale, value_dim=512):
    rows, heads, dim = query.shape
    if dim != 512 or not 0 < value_dim <= dim:
        raise ValueError("SM120 V4.1 attention requires 512-dimensional keys")
    output = torch.empty(
        (rows, heads, value_dim), dtype=query.dtype, device=query.device
    )
    maximum = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
    lse = torch.empty_like(maximum)
    if sinks is None:
        sink = torch.full((1, heads), -torch.inf, device=query.device)
    else:
        sink = sinks.to(device=query.device, dtype=torch.float32).reshape(1, heads)
    # Bound both the gathered KV tensor and attention logits independently of
    # the total prompt length. Loop bounds depend only on graph-static shapes.
    for start in range(0, rows, 32):
        end = min(start + 32, rows)
        q = query[start:end].float()
        m = torch.maximum(sink, torch.zeros_like(sink)).expand(end - start, -1)
        raw_max = torch.full_like(m, -torch.inf)
        total = torch.zeros_like(m)
        acc = torch.zeros(
            (end - start, heads, value_dim),
            dtype=torch.float32,
            device=query.device,
        )
        for indices, lengths, gather in pools:
            width = indices.shape[1]
            for column in range(0, width, 128):
                last = min(column + 128, width)
                slots = indices[start:end, column:last]
                valid = (slots >= 0) & (
                    torch.arange(column, last, device=query.device)[None, :]
                    < lengths[start:end, None]
                )
                # A masked slot can alias uninitialized/NaN backing storage;
                # zero it before either contraction (0 * NaN is still NaN).
                kv = torch.where(valid[..., None], gather(slots), 0.0).float()
                scores = torch.einsum("rhd,rkd->rhk", q, kv) * scale
                scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
                block_max = scores.amax(-1)
                raw_max = torch.maximum(raw_max, block_max)
                next_m = torch.maximum(m, block_max)
                old_scale = torch.exp(m - next_m)
                probability = torch.exp(scores - next_m[..., None])
                acc = acc * old_scale[..., None] + torch.einsum(
                    "rhk,rkd->rhd", probability, kv[..., :value_dim]
                )
                total = total * old_scale + probability.sum(-1)
                m = next_m
        denominator = total + torch.exp(sink - m)
        output[start:end].copy_(
            torch.where(
                denominator[..., None] > 0,
                acc / denominator[..., None],
                0.0,
            ).to(query.dtype)
        )
        maximum[start:end].copy_(raw_max)
        # FlashMLA's LSE excludes the attention sink and marks empty rows +inf.
        lse[start:end].copy_(torch.where(total > 0, m + torch.log(total), torch.inf))
    return output, maximum, lse


def sparse_prefill(q, kv, indices, scale, value_dim, sinks, lengths):
    if q.ndim != 3 or kv.shape[-1] != 512:
        raise ValueError("SM120 sparse prefill requires [rows,heads,512] queries")
    rows = q.shape[0]
    slots = indices.reshape(rows, -1).to(torch.int64)
    lens = _lengths(lengths, rows, slots.shape[1], q.device)
    keys = kv.reshape(-1, 512)

    def gather(selected):
        return keys[selected.clamp(0, max(keys.shape[0] - 1, 0))]

    return _attention(q, sinks, [(slots, lens, gather)], scale, value_dim)


def sparse_decode(
    q,
    sinks,
    cache,
    indices,
    scale,
    lengths=None,
    extra_cache=None,
    extra_indices=None,
    extra_lengths=None,
):
    from ._v41_fp4_triton import dequantize_k_cache_slots_fp4
    from ._v41_swa_triton import dequantize_swa_k_cache

    batch, span, heads, dim = q.shape
    rows = batch * span
    pools = []
    for pool, slots, lens in (
        (cache, indices, lengths),
        (extra_cache, extra_indices, extra_lengths),
    ):
        if pool is None:
            if slots is not None:
                raise ValueError("Sparse indices require a corresponding cache")
            continue
        if slots is None:
            raise ValueError("Sparse cache requires corresponding indices")
        if pool.ndim == 4 and pool.shape[-2] == 1:
            pool = pool.squeeze(-2)
        if pool.ndim != 3 or pool.shape[-1] not in (528, 288):
            raise ValueError("SM120 V4.1 requires SWA 528B or GLOBAL 288B rows")
        flat_slots = slots.reshape(rows, -1).to(torch.int64)
        counts = _lengths(lens, rows, flat_slots.shape[1], q.device)

        def gather(selected, pool=pool):
            if pool.shape[-1] == 288:
                return dequantize_k_cache_slots_fp4(pool, selected.reshape(-1)).view(
                    *selected.shape, 512
                )
            return dequantize_swa_k_cache(pool, selected.reshape(-1)).view(
                *selected.shape, 512
            )

        pools.append((flat_slots, counts, gather))
    output, _, lse = _attention(q.reshape(rows, heads, dim), sinks, pools, scale)
    return output.view_as(q), lse.view(batch, span, heads).transpose(1, 2).contiguous()
