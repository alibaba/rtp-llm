"""In-place RoPE and direct packed cache stores for ordinary PPU Decode Graph."""

from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41.ppu._quant import (
    FP4_GLOBAL_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    INDEXER_K_ENTRY_BYTES,
    KV_DIM,
    SWA_KV_ENTRY_BYTES,
)
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import _pack_e2m1_payload

    @triton.jit
    def rotate_pairs(real, imag, cosine, sine):
        # Use the PPU complex-multiply contraction from fused_ops. The sine
        # products are rounded before the explicit FP32 FMAs.
        return (
            tl.fma(real, cosine, -(imag * sine)),
            tl.fma(imag, cosine, real * sine),
        )

    @triton.jit
    def rotate_heads(
        q, token, tile, cosine, sine, active, HEADS: tl.constexpr, DIM: tl.constexpr
    ):
        heads = tile * 8 + tl.arange(0, 8)
        pair = tl.arange(0, 32)
        row = (token * HEADS + heads[:, None]) * DIM
        address = q + row + (DIM - 64) + 2 * pair[None, :]
        live = active & (heads[:, None] < HEADS)
        real = tl.load(address, live, other=0).to(tl.float32)
        imag = tl.load(address + 1, live, other=0).to(tl.float32)
        real, imag = rotate_pairs(real, imag, cosine[None, :], sine[None, :])
        tl.store(address, real, heads[:, None] < HEADS)
        tl.store(address + 1, imag, heads[:, None] < HEADS)

    @triton.jit
    def pack_swa(values):
        # Match the PPU torch codec, including amax/448 rounding and the
        # scale byte cast before decoding. The input has landed in BF16.
        groups = values.to(tl.float32).reshape((16, 32))
        has_nan = tl.sum((groups != groups).to(tl.int32), 1) > 0
        maximum = tl.maximum(tl.max(tl.abs(groups), 1), 1.0e-4)
        maximum = tl.where(has_nan, float("nan"), maximum)
        raw_bits = tl.div_rn(maximum, 448.0).to(tl.uint32, bitcast=True)
        encoded = (((raw_bits >> 23) & 255) + ((raw_bits & 0x7FFFFF) != 0)).to(
            tl.uint8
        )
        scale = (encoded.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        scale = tl.where(encoded == 0, 2.0**-127, scale)
        scale = tl.where(encoded == 255, float("nan"), scale)
        scaled = tl.div_rn(groups, scale[:, None])
        clipped = tl.clamp(scaled, -448.0, 448.0)
        clipped = tl.where(scaled != scaled, scaled, clipped)
        payload = clipped.to(tl.float8e4nv, fp_downcast_rounding="rtne")
        return payload.to(tl.uint8, bitcast=True).reshape((512,)), encoded

    @triton.jit
    def rotate_q_store_swa(
        q,
        kv,
        frequencies,
        slots,
        pool,
        HEADS: tl.constexpr,
        ENTRIES: tl.constexpr,
        POOL_BLOCKS: tl.constexpr,
        BLOCK_STRIDE: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        pair = tl.arange(0, 32)
        cosine = tl.load(frequencies + token * 64 + 2 * pair)
        sine = tl.load(frequencies + token * 64 + 2 * pair + 1)
        if tile < tl.cdiv(HEADS, 8):
            rotate_heads(q, token, tile, cosine, sine, True, HEADS, 512)
        else:
            slot = tl.load(slots + token).to(tl.int64)
            block = slot // ENTRIES
            if (slot >= 0) & (block > 0) & (block < POOL_BLOCKS):
                column = tl.arange(0, 512)
                values = tl.load(kv + token * 512 + column)
                real = tl.gather(values, 448 + 2 * pair, 0).to(tl.float32)
                imag = tl.gather(values, 449 + 2 * pair, 0).to(tl.float32)
                real, imag = rotate_pairs(real, imag, cosine, sine)
                tail = tl.interleave(real, imag).to(tl.bfloat16)
                rotated = tl.where(
                    column < 448,
                    values,
                    tl.gather(tail, tl.maximum(column - 448, 0), 0),
                )
                payload, scale = pack_swa(rotated)
                base = pool + block * BLOCK_STRIDE + (slot % ENTRIES) * 528
                tl.store(base + column, payload)
                tl.store(base + 512 + tl.arange(0, 16), scale)

    @triton.jit
    def rope_inplace(
        q, frequencies, valid, HEADS: tl.constexpr, DIM: tl.constexpr, INVERSE: tl.constexpr
    ):
        token = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        pair = tl.arange(0, 32)
        cosine = tl.load(frequencies + token * 64 + 2 * pair)
        sine = tl.load(frequencies + token * 64 + 2 * pair + 1)
        if INVERSE:
            sine = -sine
        if valid is None:
            active = True
        else:
            active = tl.load(valid + token)
            if not active:
                heads = tile * 8 + tl.arange(0, 8)
                columns = tl.arange(0, DIM)
                row = (token * HEADS + heads[:, None]) * DIM
                tl.store(
                    q + row + columns[None, :],
                    0,
                    (heads[:, None] < HEADS) & (columns[None, :] < DIM - 64),
                )
        # Rotate zero for a padded row, matching the original mask then RoPE.
        rotate_heads(q, token, tile, cosine, sine, active, HEADS, DIM)

    @triton.jit
    def write_fp4(
        kv,
        slots,
        pool,
        DIM: tl.constexpr,
        ROW_STRIDE: tl.constexpr,
        ENTRIES: tl.constexpr,
        POOL_BLOCKS: tl.constexpr,
        BLOCK_STRIDE: tl.constexpr,
        INDEXER: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        slot = tl.load(slots + token).to(tl.int64)
        block = slot // ENTRIES
        if (slot >= 0) & (block > 0) & (block < POOL_BLOCKS):
            offset = slot % ENTRIES
            base = pool + block * BLOCK_STRIDE
            group: tl.constexpr = 32 if INDEXER else 16
            values = tl.load(kv + token * ROW_STRIDE + tl.arange(0, DIM))
            values = values.to(tl.float32).reshape((DIM // group, group))
            has_nan = tl.sum((values != values).to(tl.int32), 1) > 0
            maximum = tl.max(tl.abs(values), 1)
            if INDEXER:
                # Match the PPU MX codec's reciprocal multiplication before
                # exponent extraction. Reconstruct the scale before truncating
                # the exponent to a byte, including the NaN and Inf cases.
                maximum = tl.maximum(maximum, 6.0 * (2.0**-126))
                maximum = tl.where(has_nan, float("nan"), maximum)
                bits = (maximum * (1.0 / 6.0)).to(tl.uint32, bitcast=True)
                exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)
                scale = (exponent << 23).to(tl.float32, bitcast=True)
                scale = tl.where(exponent >= 255, float("inf"), scale)
                encoded = exponent.to(tl.uint8)
                normalized = tl.div_rn(values, scale[:, None])
                payload_base = base + offset * 64
                scale_base = base + ENTRIES * 64 + offset * 4
            else:
                raw_scale = tl.div_rn(maximum, 6.0)
                raw_scale = tl.clamp(raw_scale, 2.0**-9, 448.0)
                raw_scale = tl.where(has_nan, float("nan"), raw_scale)
                scale_fp8 = raw_scale.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                encoded = scale_fp8.to(tl.uint8, bitcast=True)
                # PPU GLOBAL uses reciprocal then multiply. A single division
                # can change FP4 midpoint ties when the E4M3 scale is inexact.
                inverse = tl.div_rn(1.0, scale_fp8.to(tl.float32))
                normalized = values * inverse[:, None]
                payload_base = base + offset * 288
                scale_base = payload_base + 256
            payload = _pack_e2m1_payload(normalized, DIM)
            tl.store(payload_base + tl.arange(0, DIM // 2), payload)
            tl.store(scale_base + tl.arange(0, DIM // group), encoded)

    @triton.jit
    def write_csa(
        values,
        scores,
        slots,
        pool,
        VALUE_STRIDE: tl.constexpr,
        SCORE_STRIDE: tl.constexpr,
        ENTRIES: tl.constexpr,
        POOL_BLOCKS: tl.constexpr,
        BLOCK_STRIDE: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        slot = tl.load(slots + token).to(tl.int64)
        block = slot // ENTRIES
        if (slot >= 0) & (block > 0) & (block < POOL_BLOCKS):
            column = tl.arange(0, 512)
            base = pool + block * BLOCK_STRIDE + (slot % ENTRIES) * 1024
            value = tl.load(values + token * VALUE_STRIDE + column)
            score = tl.load(scores + token * SCORE_STRIDE + column)
            tl.store(base + column, value.to(tl.float32))
            tl.store(base + 512 + column, score.to(tl.float32))

    return rotate_q_store_swa, rope_inplace, write_fp4, write_csa


def _can_rotate(q, freqs):
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.ndim in (2, 3)
        and q.shape[0] > 0
        and q.shape[-2] > 0
        and q.shape[-1] in (128, 512)
        and q.is_contiguous()
        and not q.requires_grad
        and freqs.device == q.device
        and freqs.dtype == torch.complex64
        and freqs.shape == (q.shape[0], 32)
        and freqs.is_contiguous()
        and not freqs.is_conj()
        and not freqs.requires_grad
    )


def _native_pool(cache, layer_id, tag, device, row_width, dtype=torch.uint8):
    pool = cache.pool(layer_id, tag)
    entries = cache.entries[cache._table_tag(layer_id, tag)]
    if not (
        pool.device == device
        and pool.dtype == dtype
        and pool.ndim == 2
        and pool.stride(1) == 1
        and entries > 0
        and pool.shape[1] >= entries * row_width
        and pool.stride(0) >= pool.shape[1]
    ):
        return None
    return pool, entries


@lru_cache(maxsize=1)
def _csa_kernels():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def slot(row, position, active, table, LAYOUT: tl.constexpr, RATIO: tl.constexpr):
        page = position // LAYOUT[3]
        mapped = active & (position >= 0) & (page < LAYOUT[2])
        block = tl.load(
            table + row * LAYOUT[0] + page * LAYOUT[1], mapped, other=0
        ).to(tl.int64)
        if RATIO == 1:
            offset = position % LAYOUT[4]
        else:
            offset = (position % LAYOUT[3]) // RATIO
        mapped = mapped & (block > 0) & (block < LAYOUT[5]) & (offset < LAYOUT[4])
        return tl.where(mapped, block * LAYOUT[4] + offset, -1)

    @triton.jit
    def prepare_csa(
        positions, valid, freqs, state_table, main_table, index_table, slots, rope,
        ROWS: tl.constexpr, FREQ_ROWS: tl.constexpr,
        STATE: tl.constexpr, MAIN: tl.constexpr, INDEX: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        position = tl.load(positions + row).to(tl.int64)
        active = tl.load(valid + row)
        commit = active & ((position & 1) == 1)
        previous = position - 1
        tl.store(slots + row, slot(row, position, active, state_table, STATE, 1))
        tl.store(slots + ROWS + row, slot(row, previous, commit, state_table, STATE, 1))
        tl.store(slots + 2 * ROWS + row, slot(row, previous, commit, main_table, MAIN, 2))
        tl.store(slots + 3 * ROWS + row, slot(row, previous, commit, index_table, INDEX, 2))
        columns = tl.arange(0, 64)
        phase = tl.minimum(tl.maximum(previous, 0), FREQ_ROWS - 1)
        tl.store(rope + row * 64 + columns, tl.load(freqs + phase * 64 + columns))

    @triton.jit
    def square_sum_512(values, ROWS: tl.constexpr):
        # Follow the CUDA V4.1 compressor's ATen reduction order. The B8
        # PPU mean also launches 64 lanes per row with four vector accumulators.
        LANES: tl.constexpr = 128 if ROWS < 8 else (64 if ROWS < 16 else 32)
        lane = tl.arange(0, LANES)
        sums0 = tl.full((LANES,), 0, tl.float32)
        sums1 = tl.full((LANES,), 0, tl.float32)
        sums2 = tl.full((LANES,), 0, tl.float32)
        sums3 = tl.full((LANES,), 0, tl.float32)
        squares = values * values
        for chunk in tl.static_range(128 // LANES):
            first = (lane + chunk * LANES) * 4
            sums0 = sums0 + tl.gather(squares, first, 0)
            sums1 = sums1 + tl.gather(squares, first + 1, 0)
            sums2 = sums2 + tl.gather(squares, first + 2, 0)
            sums3 = sums3 + tl.gather(squares, first + 3, 0)
        partials = ((sums0 + sums1) + sums2) + sums3
        if LANES == 128:
            lower = tl.arange(0, 64)
            partials = tl.gather(partials, lower, 0) + tl.gather(partials, lower + 64, 0)
        if LANES >= 64:
            lower = tl.arange(0, 32)
            partials = tl.gather(partials, lower, 0) + tl.gather(partials, lower + 32, 0)
        return tl.sum(partials, 0)

    @triton.jit
    def compress_csa(
        values, scores, gamma, previous_slots, pool, output,
        ROWS: tl.constexpr, VALUE_STRIDE: tl.constexpr, SCORE_STRIDE: tl.constexpr,
        ENTRIES: tl.constexpr, POOL_BLOCKS: tl.constexpr, BLOCK_STRIDE: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, 512)
        previous_slot = tl.load(previous_slots + row)
        block = previous_slot // ENTRIES
        live = (previous_slot >= 0) & (block > 0) & (block < POOL_BLOCKS)
        base = pool + block * BLOCK_STRIDE + (previous_slot % ENTRIES) * 1024
        previous = tl.load(base + columns, live, other=0)
        previous_score = tl.load(base + 512 + columns, live, other=0)
        current = tl.load(values + row * VALUE_STRIDE + columns)
        current_score = tl.load(scores + row * SCORE_STRIDE + columns)
        maximum = tl.maximum(previous_score, current_score)
        previous_exp = libdevice.exp(previous_score - maximum)
        current_exp = libdevice.exp(current_score - maximum)
        denominator = previous_exp + current_exp
        pooled = previous * tl.div_rn(previous_exp, denominator) + (
            current * tl.div_rn(current_exp, denominator)
        )
        # The PPU torch compressor rounds the pooled values before RMSNorm.
        pooled = pooled.to(tl.bfloat16).to(tl.float32)
        if gamma is not None:
            inverse = tl.rsqrt(square_sum_512(pooled, ROWS) / 512 + EPS)
            weight = tl.load(gamma + columns).to(tl.float32)
            pooled = pooled * inverse * weight
        tl.store(output + row * 512 + columns, pooled)

    return prepare_csa, compress_csa


def try_compress_csa(values, scores, norm, eps, freqs, positions, valid, cache, layer_id, shared):
    """Pool from prior FP32 state and prepare slots without modifying any cache."""
    rows = positions.numel()
    if not (
        type(cache) is NativeGraphCache
        and rows > 0
        and positions.is_cuda
        and positions.shape == valid.shape == (rows,)
        and positions.dtype in (torch.int32, torch.int64)
        and valid.dtype == torch.bool
        and positions.device == valid.device == values.device == scores.device == freqs.device
        and positions.is_contiguous()
        and valid.is_contiguous()
        and values.shape == scores.shape == (rows, KV_DIM)
        and values.dtype == scores.dtype == torch.float32
        and values.stride(1) == scores.stride(1) == 1
        and min(values.stride(0), scores.stride(0)) >= KV_DIM
        and freqs.ndim == 2
        and freqs.shape[0] > 0
        and freqs.shape[1] == 32
        and freqs.dtype == torch.complex64
        and freqs.is_contiguous()
        and not freqs.is_conj()
        and not torch.is_grad_enabled()
    ):
        return None
    if norm is not None and (
        norm.dtype not in (torch.bfloat16, torch.float32)
        or norm.shape != (KV_DIM,)
        or norm.device != values.device
        or not norm.is_contiguous()
    ):
        return None

    target = _native_pool(
        cache, layer_id, "csa_state", values.device, 2 * KV_DIM, torch.float32
    )
    if target is None:
        return None
    pool, entries = target
    tables, layouts = [], []
    for tag in ("csa_state", "global_kv_2", "indexer_kv"):
        domain = cache._table_tag(layer_id, tag)
        table = cache.tables[domain]
        if (
            table.device != values.device
            or table.dtype not in (torch.int32, torch.int64)
            or table.ndim != 2
            or table.shape[0] < rows
            or not table.shape[1]
        ):
            return None
        tables.append(table)
        layouts.append((
            table.stride(0), table.stride(1), table.shape[1],
            cache.spans[domain], cache.entries[domain], cache.pool(layer_id, tag).shape[0],
        ))

    prepare_csa, compress_csa = _csa_kernels()
    key = ("packed_csa", id(freqs), tuple(id(table) for table in tables), tuple(layouts))
    if key not in shared:
        slots = torch.empty((4, rows), device=values.device, dtype=torch.int64)
        rope = torch.empty((rows, 32), device=values.device, dtype=torch.complex64)
        prepare_csa[(rows,)](
            positions, valid, torch.view_as_real(freqs), *tables, slots,
            torch.view_as_real(rope), rows, freqs.shape[0], *layouts, num_warps=4,
        )
        shared[key] = (*slots.unbind(), rope)
    current, previous, main, index, rope = shared[key]
    latent = torch.empty((rows, KV_DIM), device=values.device, dtype=torch.bfloat16)
    compress_csa[(rows,)](
        values, scores, norm, previous, pool, latent, rows, values.stride(0),
        scores.stride(0), entries, pool.shape[0], pool.stride(0), eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return latent, current, main, index, rope


def try_rotate_q_and_write_swa(q, kv, freqs, cache, layer_id, slots):
    """Rotate an owned Q in place and quantize normalized KV into native slots.

    Q must be the fresh projection result with no live aliases. KV and the
    gathered RoPE rows are read-only. The caller supplies ordinary Decode
    slots, with at most one live writer per slot. Unsupported layouts return
    None before any write.
    """
    if not _can_rotate(q, freqs) or not (
        q.ndim == 3
        and q.shape[-1] == 512
        and kv.shape == (q.shape[0], 512)
        and kv.device == q.device
        and kv.dtype == torch.bfloat16
        and kv.is_contiguous()
        and not kv.requires_grad
        and slots.shape == (q.shape[0],)
        and slots.device == q.device
        and slots.dtype == torch.int64
        and slots.is_contiguous()
    ):
        return None
    target = _native_pool(cache, layer_id, "swa_kv", q.device, SWA_KV_ENTRY_BYTES)
    if target is None:
        return None
    pool, entries = target
    rotate_q_store_swa, _, _, _ = _kernels()
    heads = q.shape[1]
    rotate_q_store_swa[(q.shape[0], (heads + 7) // 8 + 1)](
        q,
        kv,
        torch.view_as_real(freqs),
        slots,
        pool,
        heads,
        entries,
        pool.shape[0],
        pool.stride(0),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q


def try_rope_(output, freqs, *, inverse=False, valid_mask=None):
    """Rotate an owned tensor, optionally zeroing padded rows before RoPE."""
    if not _can_rotate(output, freqs) or (
        valid_mask is not None
        and (
            valid_mask.shape != (output.shape[0],)
            or valid_mask.dtype != torch.bool
            or valid_mask.device != output.device
            or not valid_mask.is_contiguous()
        )
    ):
        return None
    _, rope_inplace, _, _ = _kernels()
    heads = output.shape[1] if output.ndim == 3 else 1
    rope_inplace[(output.shape[0], (heads + 7) // 8)](
        output,
        torch.view_as_real(freqs),
        valid_mask,
        heads,
        output.shape[-1],
        inverse,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def try_write_packed_kv(kv, cache, layer_id, tag, slots):
    """Encode BF16 keys directly in native slots, or decline before writing.

    The caller supplies already rotated keys and unique live ordinary Decode
    slots. The GLOBAL pool uses 288-byte interleaved rows. The INDEXER pool
    stores all 64-byte payloads before the 4-byte scales for each block.
    """
    indexer = tag == "indexer_kv"
    if not indexer and tag not in ("global_kv_1", "global_kv_2"):
        return False
    dim = INDEXER_HEAD_DIM if indexer else KV_DIM
    entry_bytes = INDEXER_K_ENTRY_BYTES if indexer else FP4_GLOBAL_ENTRY_BYTES
    if not (
        kv.is_cuda
        and kv.dtype == torch.bfloat16
        and kv.ndim == 2
        and kv.shape[1] == dim
        and kv.stride(1) == 1
        and kv.stride(0) >= dim
        and not kv.requires_grad
        and slots.shape == (kv.shape[0],)
        and slots.device == kv.device
        and slots.dtype == torch.int64
        and slots.is_contiguous()
    ):
        return False
    target = _native_pool(cache, layer_id, tag, kv.device, entry_bytes)
    if target is None:
        return False
    pool, entries = target
    if kv.shape[0]:
        _, _, write_fp4, _ = _kernels()
        write_fp4[(kv.shape[0],)](
            kv,
            slots,
            pool,
            dim,
            kv.stride(0),
            entries,
            pool.shape[0],
            pool.stride(0),
            indexer,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return True


def try_write_csa_state(values, scores, cache, layer_id, slots):
    """Store ordinary ratio-2 projections as one FP32 state row."""
    if not (
        values.is_cuda
        and values.dtype == scores.dtype
        and values.dtype in (torch.bfloat16, torch.float32)
        and values.ndim == 2
        and values.shape[1] == KV_DIM
        and values.shape == scores.shape
        and values.device == scores.device
        and values.stride(1) == scores.stride(1) == 1
        and min(values.stride(0), scores.stride(0)) >= KV_DIM
        and not values.requires_grad
        and not scores.requires_grad
        and slots.shape == (values.shape[0],)
        and slots.device == values.device
        and slots.dtype == torch.int64
        and slots.is_contiguous()
    ):
        return False
    target = _native_pool(
        cache, layer_id, "csa_state", values.device, 2 * KV_DIM, torch.float32
    )
    if target is None:
        return False
    pool, entries = target
    if values.shape[0]:
        _, _, _, write_csa = _kernels()
        write_csa[(values.shape[0],)](
            values,
            scores,
            slots,
            pool,
            values.stride(0),
            scores.stride(0),
            entries,
            pool.shape[0],
            pool.stride(0),
            num_warps=4,
        )
    return True
