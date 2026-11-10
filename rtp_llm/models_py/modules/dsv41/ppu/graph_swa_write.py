"""In-place RoPE and packed cache or journal stores for PPU Decode Graph."""

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
    from rtp_llm.models_py.modules.dsv41.ppu.fused_ops import _complex_multiply

    complex_multiply = _complex_multiply()

    @triton.jit
    def rotate_pairs(real, imag, cosine, sine, SIGNED_ZERO: tl.constexpr):
        if SIGNED_ZERO:
            return complex_multiply(real, imag, cosine, sine)
        # Use the PPU complex-multiply contraction from fused_ops. The sine
        # products are rounded before the explicit FP32 FMAs.
        return (
            tl.fma(real, cosine, -(imag * sine)),
            tl.fma(imag, cosine, real * sine),
        )

    @triton.jit
    def rotate_heads(
        q, token, tile, cosine, sine, active, HEADS: tl.constexpr, DIM: tl.constexpr,
        SIGNED_ZERO: tl.constexpr,
    ):
        heads = tile * 8 + tl.arange(0, 8)
        pair = tl.arange(0, 32)
        row = (token * HEADS + heads[:, None]) * DIM
        address = q + row + (DIM - 64) + 2 * pair[None, :]
        live = active & (heads[:, None] < HEADS)
        real = tl.load(address, live, other=0).to(tl.float32)
        imag = tl.load(address + 1, live, other=0).to(tl.float32)
        real, imag = rotate_pairs(real, imag, cosine[None, :], sine[None, :], SIGNED_ZERO)
        tl.store(address, real, heads[:, None] < HEADS)
        tl.store(address + 1, imag, heads[:, None] < HEADS)

    @triton.jit
    def rotate_kv(values, cosine, sine, DIM: tl.constexpr, SIGNED_ZERO: tl.constexpr):
        pair = tl.arange(0, 32)
        column = tl.arange(0, DIM)
        real = tl.gather(values, DIM - 64 + 2 * pair, 0).to(tl.float32)
        imag = tl.gather(values, DIM - 63 + 2 * pair, 0).to(tl.float32)
        real, imag = rotate_pairs(real, imag, cosine, sine, SIGNED_ZERO)
        tail = tl.interleave(real, imag).to(tl.bfloat16)
        return tl.where(
            column < DIM - 64,
            values,
            tl.gather(tail, tl.maximum(column - (DIM - 64), 0), 0),
        )

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
        generations,
        HEADS: tl.constexpr,
        ENTRIES: tl.constexpr,
        POOL_BLOCKS: tl.constexpr,
        BLOCK_STRIDE: tl.constexpr,
        JOURNAL: tl.constexpr,
        SIGNED_ZERO: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        pair = tl.arange(0, 32)
        cosine = tl.load(frequencies + token * 64 + 2 * pair)
        sine = tl.load(frequencies + token * 64 + 2 * pair + 1)
        if tile < tl.cdiv(HEADS, 8):
            rotate_heads(q, token, tile, cosine, sine, True, HEADS, 512, SIGNED_ZERO)
        else:
            if slots is None:
                writable = True
                base = pool + token * BLOCK_STRIDE
            elif JOURNAL:
                # Preserve the raw generation and write every payload row.
                # Padded rows have their own journal storage too.
                slot = tl.load(slots + token).to(tl.int64)
                tl.store(generations + token, slot)
                writable = True
                base = pool + token * BLOCK_STRIDE
            else:
                slot = tl.load(slots + token).to(tl.int64)
                block = slot // ENTRIES
                writable = (slot >= 0) & (block > 0) & (block < POOL_BLOCKS)
                base = pool + block * BLOCK_STRIDE + (slot % ENTRIES) * 528
            if writable:
                column = tl.arange(0, 512)
                values = tl.load(kv + token * 512 + column)
                payload, scale = pack_swa(rotate_kv(values, cosine, sine, 512, SIGNED_ZERO))
                tl.store(base + column, payload)
                tl.store(base + 512 + tl.arange(0, 16), scale)

    @triton.jit
    def rope_inplace(
        q, frequencies, valid, HEADS: tl.constexpr, DIM: tl.constexpr, INVERSE: tl.constexpr,
        SIGNED_ZERO: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        pair = tl.arange(0, 32)
        cosine = tl.load(frequencies + token * 64 + 2 * pair)
        sine = tl.load(frequencies + token * 64 + 2 * pair + 1)
        if INVERSE and SIGNED_ZERO:
            sine = (sine.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)
        elif INVERSE:
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
        rotate_heads(q, token, tile, cosine, sine, active, HEADS, DIM, SIGNED_ZERO)

    @triton.jit
    def write_fp4(
        kv,
        frequencies,
        slots,
        pool,
        generations,
        DIM: tl.constexpr,
        ROW_STRIDE: tl.constexpr,
        ENTRIES: tl.constexpr,
        POOL_BLOCKS: tl.constexpr,
        BLOCK_STRIDE: tl.constexpr,
        INDEXER: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        slot = tl.load(slots + token).to(tl.int64)
        if generations is not None:
            tl.store(generations + token, slot)
            base = pool + token * BLOCK_STRIDE
            offset = 0
            writable = True
        else:
            block = slot // ENTRIES
            offset = slot % ENTRIES
            base = pool + block * BLOCK_STRIDE
            writable = (slot >= 0) & (block > 0) & (block < POOL_BLOCKS)
        if writable:
            group: tl.constexpr = 32 if INDEXER else 16
            values = tl.load(kv + token * ROW_STRIDE + tl.arange(0, DIM))
            if frequencies is not None:
                pair = tl.arange(0, 32)
                cosine = tl.load(frequencies + token * 64 + 2 * pair)
                sine = tl.load(frequencies + token * 64 + 2 * pair + 1)
                values = rotate_kv(values, cosine, sine, DIM, True)
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
                if generations is not None:
                    payload_base = base
                    scale_base = base + 64
                else:
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
        generations,
        VALUE_STRIDE: tl.constexpr,
        SCORE_STRIDE: tl.constexpr,
        ENTRIES: tl.constexpr,
        POOL_BLOCKS: tl.constexpr,
        BLOCK_STRIDE: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        slot = tl.load(slots + token).to(tl.int64)
        if generations is not None:
            tl.store(generations + token, slot)
            base = pool + token * BLOCK_STRIDE
            writable = True
        else:
            block = slot // ENTRIES
            base = pool + block * BLOCK_STRIDE + (slot % ENTRIES) * 1024
            writable = (slot >= 0) & (block > 0) & (block < POOL_BLOCKS)
        if writable:
            column = tl.arange(0, 512)
            value = tl.load(values + token * VALUE_STRIDE + column)
            score = tl.load(scores + token * SCORE_STRIDE + column)
            tl.store(base + column, value.to(tl.float32))
            tl.store(base + 512 + column, score.to(tl.float32))

    @triton.jit
    def write_rotary_swa(
        kv, frequencies, slots, pool, generations, ENTRIES: tl.constexpr,
        POOL_BLOCKS: tl.constexpr, BLOCK_STRIDE: tl.constexpr,
        JOURNAL: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        slot = tl.load(slots + row)
        if JOURNAL:
            tl.store(generations + row, slot)
            writable = slot >= 0
            base = pool + row * BLOCK_STRIDE
        else:
            block = slot // ENTRIES
            writable = (slot >= 0) & (block > 0) & (block < POOL_BLOCKS)
            base = pool + block * BLOCK_STRIDE + (slot % ENTRIES) * 528
        if writable:
            pair = tl.arange(0, 32)
            columns = tl.arange(0, 512)
            cosine = tl.load(frequencies + row * 64 + 2 * pair)
            sine = tl.load(frequencies + row * 64 + 2 * pair + 1)
            values = tl.load(kv + row * 512 + columns)
            payload, scale = pack_swa(rotate_kv(values, cosine, sine, 512, True))
            tl.store(base + columns, payload)
            tl.store(base + 512 + tl.arange(0, 16), scale)

    return rotate_q_store_swa, rope_inplace, write_fp4, write_csa, write_rotary_swa


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


def _journal_pool(cache, layer_id, tag, slots, row_width, dtype=torch.uint8):
    key = (layer_id, tag)
    pool, generations = cache.payloads[key], cache.generations[key]
    if not (
        pool.shape == (slots.shape[0], row_width)
        and pool.device == generations.device == slots.device
        and pool.dtype == dtype
        and pool.is_contiguous()
        and generations.shape == slots.shape
        and generations.dtype == torch.int64
        and generations.is_contiguous()
    ):
        return None
    return pool, generations


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

    @triton.jit
    def prepare_verify_pairs(
        values, scores, raw_slots, requests, starts, prefixes, valid,
        generations, payloads, table, pool, paired_values, paired_scores,
        ROWS: tl.constexpr, VALUE_STRIDE: tl.constexpr, SCORE_STRIDE: tl.constexpr,
        POOL_STRIDE: tl.constexpr, LAYOUT: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, 512)
        raw = tl.load(raw_slots + row)
        request = tl.load(requests + row)
        offset = raw - tl.load(prefixes + request)
        candidate = tl.minimum(tl.maximum(tl.load(starts + request) + offset, 0), ROWS - 1)
        pending = (
            (raw >= 0) & (offset >= 0) & tl.load(valid + candidate)
            & (tl.load(requests + candidate) == request)
            & (tl.load(generations + candidate) == raw)
        )
        if pending:
            previous = payloads + candidate * 1024
            previous_value = tl.load(previous + column)
            previous_score = tl.load(previous + 512 + column)
        else:
            previous_slot = slot(row, raw, True, table, LAYOUT, 1)
            block = previous_slot // LAYOUT[4]
            previous = pool + block * POOL_STRIDE + (previous_slot % LAYOUT[4]) * 1024
            previous_value = tl.load(previous + column, previous_slot >= 0, other=0)
            previous_score = tl.load(previous + 512 + column, previous_slot >= 0, other=0)
        output = row * 1024 + column
        tl.store(paired_values + output, previous_value)
        tl.store(paired_scores + output, previous_score)
        tl.store(paired_values + output + 512, tl.load(values + row * VALUE_STRIDE + column))
        tl.store(paired_scores + output + 512, tl.load(scores + row * SCORE_STRIDE + column))

    return prepare_csa, compress_csa, prepare_verify_pairs


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

    prepare_csa, compress_csa, _ = _csa_kernels()
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


def try_prepare_verify_csa_pairs(values, scores, cache, layer_id, slots):
    """Read previous Verify state and form pairs for the original Torch compressor."""
    rows = slots.numel()
    if not (
        rows > 0
        and values.is_cuda
        and values.dtype == scores.dtype == torch.float32
        and values.shape == scores.shape == (rows, KV_DIM)
        and values.stride(1) == scores.stride(1) == 1
        and min(values.stride(0), scores.stride(0)) >= KV_DIM
        and not (values.requires_grad or scores.requires_grad)
        and slots.shape == (rows,)
        and slots.dtype == torch.int64
        and slots.is_contiguous()
        and slots.device == values.device == scores.device
    ):
        return None
    tag = "csa_state"
    native = cache.native_cache
    domain = native._table_tag(layer_id, tag)
    table = cache.row_tables[domain]
    requests, starts, prefixes, valid = cache.requests, cache.starts, cache.prefixes, cache.valid
    if not (
        requests.shape == valid.shape == slots.shape
        and starts.ndim == prefixes.ndim == 1
        and starts.numel() == prefixes.numel() > 0
        and valid.dtype == torch.bool
        and all(tensor.dtype in (torch.int32, torch.int64) for tensor in (requests, starts, prefixes))
        and all(tensor.device == slots.device and tensor.is_contiguous()
                for tensor in (requests, starts, prefixes, valid))
        and table.device == slots.device
        and table.dtype in (torch.int32, torch.int64)
        and table.ndim == 2
        and table.shape[0] == rows
        and table.shape[1] > 0
    ):
        return None
    journal = _journal_pool(cache, layer_id, tag, slots, 2 * KV_DIM, torch.float32)
    target = _native_pool(native, layer_id, tag, values.device, 2 * KV_DIM, torch.float32)
    if journal is None or target is None:
        return None
    payloads, generations = journal
    pool, entries = target
    layout = (
        table.stride(0), table.stride(1), table.shape[1],
        native.spans[domain], entries, pool.shape[0],
    )
    paired_values = values.new_empty((rows, 2, KV_DIM))
    paired_scores = scores.new_empty((rows, 2, KV_DIM))
    _, _, prepare_verify_pairs = _csa_kernels()
    prepare_verify_pairs[(rows,)](
        values, scores, slots, requests, starts, prefixes, valid, generations,
        payloads, table, pool, paired_values, paired_scores, rows, values.stride(0),
        scores.stride(0), pool.stride(0), layout, num_warps=4,
    )
    return paired_values, paired_scores


def _can_rotate_q_and_kv(q, kv, freqs):
    return _can_rotate(q, freqs) and (
        q.ndim == 3
        and q.shape[-1] == KV_DIM
        and kv.shape == (q.shape[0], KV_DIM)
        and kv.device == q.device
        and kv.dtype == torch.bfloat16
        and kv.is_contiguous()
        and not kv.requires_grad
    )


def _launch_rotate_q_and_store(q, kv, freqs, slots, pool, generations, entries, journal):
    rotate_q_store_swa, _, _, _, _ = _kernels()
    heads = q.shape[1]
    rotate_q_store_swa[(q.shape[0], (heads + 7) // 8 + 1)](
        q, kv, torch.view_as_real(freqs), slots, pool, generations, heads, entries,
        pool.shape[0], pool.stride(0), journal, journal or slots is None,
        num_warps=4, enable_fp_fusion=False,
    )


def try_rotate_q_and_pack_swa(q, kv, freqs):
    """Rotate proposal Q and encode the whole ephemeral query block."""
    if not _can_rotate_q_and_kv(q, kv, freqs):
        return None
    packed = torch.empty((q.shape[0], SWA_KV_ENTRY_BYTES), device=q.device, dtype=torch.uint8)
    _launch_rotate_q_and_store(q, kv, freqs, None, packed, None, 1, False)
    return q, packed


def try_write_rotary_swa(kv, freqs, cache, layer_id, slots, *, journal=False):
    """Write native or row-private commit bytes through the same RoPE and codec."""
    if not (
        _can_rotate(kv, freqs)
        and kv.ndim == 2
        and kv.shape[-1] == KV_DIM
        and slots.shape == (kv.shape[0],)
        and slots.device == kv.device
        and slots.dtype == torch.int64
        and slots.is_contiguous()
    ):
        return False
    target = (
        _journal_pool(cache, layer_id, "swa_kv", slots, SWA_KV_ENTRY_BYTES)
        if journal else _native_pool(cache, layer_id, "swa_kv", kv.device, SWA_KV_ENTRY_BYTES)
    )
    if target is None:
        return False
    if journal:
        pool, generations = target
        entries = 1
    else:
        pool, entries = target
        generations = slots
    _, _, _, _, write_rotary_swa = _kernels()
    write_rotary_swa[(kv.shape[0],)](
        kv, torch.view_as_real(freqs), slots, pool, generations, entries,
        pool.shape[0], pool.stride(0), journal,
        num_warps=4, enable_fp_fusion=False,
    )
    return True


@lru_cache(maxsize=1)
def _draft_swa_kernels():
    import triton
    import triton.language as tl

    from rtp_llm.models_py.modules.dsv41.ppu._quant import _triton_kernels

    ue8m0_to_fp32 = _triton_kernels().ue8m0_to_fp32

    @triton.jit
    def draft_indices(
        prefixes, active, table, slots, indices, lengths,
        ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr, TABLE_COLS: tl.constexpr,
        ENTRIES: tl.constexpr, SPAN: tl.constexpr, POOL_BLOCKS: tl.constexpr,
        WINDOW: tl.constexpr, WIDTH: tl.constexpr, PAD: tl.constexpr,
    ):
        request = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, PAD)
        enabled = tl.load(active + request)
        raw = tl.load(prefixes + request).to(tl.int64) - WINDOW + column
        page = raw // SPAN
        mapped = enabled & (column < WINDOW) & (raw >= 0) & (page < TABLE_COLS)
        block = tl.load(
            table + request * ROW_STRIDE + page * COL_STRIDE, mapped, other=0
        ).to(tl.int64)
        mapped = mapped & (block > 0) & (block < POOL_BLOCKS)
        tl.store(
            slots + request * WINDOW + column,
            tl.where(mapped, block * ENTRIES + raw % ENTRIES, -1),
            column < WINDOW,
        )
        bank_width = WINDOW + WIDTH
        fresh = enabled & (column >= WINDOW) & (column < bank_width)
        # An inactive request keeps its own zero-valued first bank row.
        live = mapped | fresh | (~enabled & (column == 0))
        prefix = tl.cumsum(live.to(tl.int32), 0)
        length = tl.sum(live.to(tl.int32), 0)
        dest = tl.where(live, prefix - 1, length + column - prefix)
        ordinal = tl.where(live, request * bank_width + column, -1)
        for query in tl.static_range(WIDTH):
            row = request * WIDTH + query
            tl.store(indices + row * PAD + dest, ordinal)
            tl.store(lengths + row, length)

    @triton.jit
    def draft_bank(
        pool, fresh, slots, bank, ENTRIES: tl.constexpr,
        POOL_STRIDE: tl.constexpr, WINDOW: tl.constexpr, WIDTH: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        request, column = row // (WINDOW + WIDTH), row % (WINDOW + WIDTH)
        channels = tl.arange(0, 512)
        if column >= WINDOW:
            source = fresh + (request * WIDTH + column - WINDOW) * 528
            read = column >= WINDOW
        else:
            slot = tl.load(slots + request * WINDOW + column)
            source = pool + (slot // ENTRIES) * POOL_STRIDE + (slot % ENTRIES) * 528
            read = slot >= 0
        if read:
            payload = tl.load(source + channels).to(tl.float32)
            scale = tl.load(source.to(tl.pointer_type(tl.uint8)) + 512 + channels // 32)
            values = (payload * ue8m0_to_fp32(scale)).to(tl.bfloat16)
        else:
            values = tl.full((512,), 0, tl.bfloat16)
        tl.store(bank + row * 512 + channels, values)

    @triton.jit
    def last_slot_writer(slots, unique, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        other = tl.arange(0, BLOCK)
        slot = tl.load(slots + row)
        later = tl.load(slots + other, (other > row) & (other < ROWS), other=-1)
        superseded = tl.sum((later == slot).to(tl.int32), 0) > 0
        tl.store(unique + row, tl.where(superseded, -1, slot))

    return triton, draft_indices, draft_bank, last_slot_writer


def try_prepare_draft_swa(cache, layer_id, prefixes, active, *, window, width):
    """Map committed rows and compact noncausal indices once per draft geometry."""
    batch = prefixes.numel()
    if not (
        type(cache) is NativeGraphCache
        and window == 128
        and 0 < width <= 16
        and batch > 0
        and prefixes.is_cuda
        and prefixes.dtype in (torch.int32, torch.int64)
        and prefixes.shape == active.shape == (batch,)
        and active.dtype == torch.bool
        and active.device == prefixes.device
        and active.is_contiguous()
        and prefixes.is_contiguous()
    ):
        return None
    tag = "swa_kv"
    domain = cache._table_tag(layer_id, tag)
    table = cache.tables[domain]
    target = _native_pool(cache, layer_id, tag, prefixes.device, SWA_KV_ENTRY_BYTES)
    if target is None or not (
        table.device == prefixes.device
        and table.dtype in (torch.int32, torch.int64)
        and table.ndim == 2
        and table.shape[0] >= batch
        and table.shape[1] > 0
    ):
        return None
    pool, entries = target
    pad = 256  # 128-aligned candidate width for window 128 plus the query block.
    slots = torch.empty((batch, window), device=prefixes.device, dtype=torch.int64)
    indices = torch.empty((batch * width, 1, pad), device=prefixes.device, dtype=torch.int32)
    lengths = torch.empty(batch * width, device=prefixes.device, dtype=torch.int32)
    _, draft_indices, _, _ = _draft_swa_kernels()
    draft_indices[(batch,)](
        prefixes, active, table, slots, indices, lengths,
        table.stride(0), table.stride(1), table.shape[1], entries, cache.spans[domain],
        pool.shape[0], window, width, pad, num_warps=4,
    )
    return slots, indices, lengths


def dequantize_draft_swa(cache, layer_id, slots, fresh, *, width):
    """Decode committed and fresh rows into the original BF16 MLA bank order."""
    pool, entries = _native_pool(cache, layer_id, "swa_kv", fresh.device, SWA_KV_ENTRY_BYTES)
    batch, window = slots.shape
    bank = torch.empty(
        (batch * (window + width), 1, KV_DIM), device=fresh.device, dtype=torch.bfloat16
    )
    _, _, draft_bank, _ = _draft_swa_kernels()
    draft_bank[(bank.shape[0],)](
        pool.view(torch.float8_e4m3fn), fresh.view(torch.float8_e4m3fn),
        slots, bank, entries, pool.stride(0), window, width, num_warps=4,
    )
    return bank


def unique_commit_swa_slots(slots):
    """Keep the last feature row for each physical slot."""
    if slots.numel() <= 1:
        return slots
    unique = torch.empty_like(slots)
    if slots.is_cuda and slots.numel() <= 512:
        triton, _, _, last_slot_writer = _draft_swa_kernels()
        last_slot_writer[(slots.numel(),)](
            slots, unique, slots.numel(), triton.next_power_of_2(slots.numel()), num_warps=4
        )
    else:
        ordered, order = slots.sort(stable=True)
        ordered[:-1].masked_fill_(ordered[:-1] == ordered[1:], -1)
        unique.scatter_(0, order, ordered)
    return unique


def try_rotate_q_and_write_swa(q, kv, freqs, cache, layer_id, slots, *, journal=False):
    """Rotate an owned Q and write normalized KV into native slots or a journal.

    Q must be the fresh projection result with no live aliases. KV and the
    gathered RoPE rows are read-only. Native slots have at most one live writer.
    A Verify journal receives raw generations and one payload per query row.
    Unsupported layouts return None before any write.
    """
    if not _can_rotate_q_and_kv(q, kv, freqs) or not (
        slots.shape == (q.shape[0],)
        and slots.device == q.device
        and slots.dtype == torch.int64
        and slots.is_contiguous()
    ):
        return None
    generations = None
    if journal:
        target = _journal_pool(cache, layer_id, "swa_kv", slots, SWA_KV_ENTRY_BYTES)
        if target is None:
            return None
        pool, generations = target
        entries = 1
    else:
        target = _native_pool(cache, layer_id, "swa_kv", q.device, SWA_KV_ENTRY_BYTES)
        if target is None:
            return None
        pool, entries = target
    _launch_rotate_q_and_store(q, kv, freqs, slots, pool, generations, entries, journal)
    return q


def try_rope_(output, freqs, *, inverse=False, valid_mask=None, preserve_signed_zero=False):
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
    _, rope_inplace, _, _, _ = _kernels()
    heads = output.shape[1] if output.ndim == 3 else 1
    rope_inplace[(output.shape[0], (heads + 7) // 8)](
        output,
        torch.view_as_real(freqs),
        valid_mask,
        heads,
        output.shape[-1],
        inverse,
        preserve_signed_zero,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def try_write_packed_kv(kv, cache, layer_id, tag, slots, *, journal=False, freqs=None):
    """Encode BF16 keys directly into native slots or Verify journal rows.

    Native Decode takes rotated keys and unique live slots. A Verify journal
    optionally applies RoPE before packing and stores all rows, including
    invalid rows. The native Indexer scale plane is interleaved in the journal.
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
    if freqs is not None and not (journal and _can_rotate(kv, freqs)):
        return False
    generations = None
    if journal:
        target = _journal_pool(cache, layer_id, tag, slots, entry_bytes)
        if target is None:
            return False
        pool, generations = target
        entries = 1
    else:
        target = _native_pool(cache, layer_id, tag, kv.device, entry_bytes)
        if target is None:
            return False
        pool, entries = target
    if kv.shape[0]:
        _, _, write_fp4, _, _ = _kernels()
        write_fp4[(kv.shape[0],)](
            kv,
            None if freqs is None else torch.view_as_real(freqs),
            slots,
            pool,
            generations,
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


def try_write_csa_state(values, scores, cache, layer_id, slots, *, journal=False):
    """Store ratio-2 projections as one FP32 state row."""
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
    generations = None
    if journal:
        target = _journal_pool(cache, layer_id, "csa_state", slots, 2 * KV_DIM, torch.float32)
        if target is None:
            return False
        pool, generations = target
        entries = 1
    else:
        target = _native_pool(
            cache, layer_id, "csa_state", values.device, 2 * KV_DIM, torch.float32
        )
        if target is None:
            return False
        pool, entries = target
    if values.shape[0]:
        _, _, _, write_csa, _ = _kernels()
        write_csa[(values.shape[0],)](
            values,
            scores,
            slots,
            pool,
            generations,
            values.stride(0),
            scores.stride(0),
            entries,
            pool.shape[0],
            pool.stride(0),
            num_warps=4,
        )
    return True
