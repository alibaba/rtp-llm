"""PPU K RMSNorm, RoPE and cache encoding in one launch.

The input is the *unnormalized* BF16 projection. FP32 RMS statistics and
weight multiplication round to BF16 before RoPE; the rotated tail rounds to
BF16 again before the existing PPU codec. The reduction tree differs from
FlashInfer RMSNorm, so callers must opt in until numerical validation.

Native writes require unique live physical slots; block zero is reserved.
Journal slots are raw generations, not physical addresses: every row gets
a private payload, including rows whose generation is -1. Callers must put
-1 in slots for invalid rows, as with the existing graph cache writers.
No tensor values are inspected on the host and no allocation survives a call.
"""

import math
from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41.ppu._quant import (
    INDEXER_HEAD_DIM,
    INDEXER_K_ENTRY_BYTES,
    KV_DIM,
    SWA_KV_ENTRY_BYTES,
    is_ppu,
)


def _raw(value):
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
    )


def _inputs_supported(raw, weight, eps, freqs, dim, *, indexed_freqs=False):
    if not (
        _raw(raw)
        and raw.is_cuda
        and raw.dtype == torch.bfloat16
        and raw.ndim == 2
        and raw.shape[1] == dim
        and raw.stride(1) == 1
        and raw.stride(0) >= dim
        and _raw(freqs)
        and freqs.device == raw.device
        and freqs.dtype == torch.complex64
        and freqs.ndim == 2
        and freqs.shape[1] == 32
        and (freqs.shape[0] > 0 if indexed_freqs else freqs.shape[0] == raw.shape[0])
        and freqs.is_contiguous()
        and isinstance(eps, (int, float))
        and math.isfinite(eps)
        and eps >= 0
        and is_ppu()
    ):
        return False
    return weight is None or (
        _raw(weight)
        and weight.device == raw.device
        and weight.dtype in (torch.bfloat16, torch.float32)
        and weight.shape == (dim,)
        and weight.is_contiguous()
    )


def _slots_supported(slots, raw):
    return (
        _raw(slots)
        and slots.device == raw.device
        and slots.dtype == torch.int64
        and slots.shape == (raw.shape[0],)
        and slots.is_contiguous()
    )


def _disjoint_writes(outputs, inputs):
    # Reject shared storage even for disjoint views: ownership must be explicit.
    storage = [value.untyped_storage().data_ptr() for value in outputs]
    return len(set(storage)) == len(storage) and not any(
        value is not None and value.untyped_storage().data_ptr() in storage
        for value in inputs
    )


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
        _pack_e2m1_payload,
    )
    from rtp_llm.models_py.modules.dsv41.ppu.fused_ops import _complex_multiply

    complex_multiply = _complex_multiply()

    @triton.jit(do_not_specialize=["row_stride", "block_stride", "pool_blocks", "freq_rows", "eps"])
    def norm_rope_store(
        raw,
        weight,
        eps,
        frequencies,
        positions,
        status,
        slots,
        pool,
        generations,
        row_stride,
        block_stride,
        pool_blocks,
        freq_rows,
        ENTRIES: tl.constexpr,
        INDEXER: tl.constexpr,
        ROW_PACK: tl.constexpr,
        JOURNAL: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        dim: tl.constexpr = 128 if INDEXER else 512
        if ROW_PACK:
            base = pool + row * block_stride
            offset = 0
            writable = True
            if JOURNAL:
                # Keep the original raw generation; -1 payloads are private too.
                tl.store(generations + row, tl.load(slots + row))
        else:
            slot = tl.load(slots + row).to(tl.int64)
            block = slot // ENTRIES
            offset = slot % ENTRIES
            base = pool + block * block_stride
            writable = (slot >= 0) & (block > 0) & (block < pool_blocks)
        freq_row = row
        if positions is not None:
            freq_row = tl.load(positions + row)
            writable = writable & (freq_row >= 0) & (freq_row < freq_rows) & (tl.load(status) == 0)
        if writable:
            columns = tl.arange(0, dim)
            values = tl.load(raw + row * row_stride + columns).to(tl.float32)
            factor = tl.rsqrt(tl.sum(values * values, 0) / dim + eps)
            values = values * factor
            if weight is not None:
                values = values * tl.load(weight + columns).to(tl.float32)
            values = values.to(tl.bfloat16)

            pair = tl.arange(0, 32)
            real = tl.gather(values, dim - 64 + 2 * pair, 0).to(tl.float32)
            imag = tl.gather(values, dim - 63 + 2 * pair, 0).to(tl.float32)
            cosine = tl.load(frequencies + freq_row * 64 + 2 * pair)
            sine = tl.load(frequencies + freq_row * 64 + 2 * pair + 1)
            real, imag = complex_multiply(real, imag, cosine, sine)
            tail = tl.interleave(real, imag).to(tl.bfloat16)
            values = tl.where(
                columns < dim - 64,
                values,
                tl.gather(tail, tl.maximum(columns - (dim - 64), 0), 0),
            ).to(tl.float32)

            # These are the graph_swa_write codecs, including their NaN, Inf,
            # exponent-floor, reciprocal and BF16 rounding conventions.
            groups = values.reshape((dim // 32, 32))
            has_nan = tl.sum((groups != groups).to(tl.int32), 1) > 0
            maximum = tl.max(tl.abs(groups), 1)
            if INDEXER:
                maximum = tl.maximum(maximum, 6.0 * (2.0**-126))
                maximum = tl.where(has_nan, float("nan"), maximum)
                bits = (maximum * (1.0 / 6.0)).to(tl.uint32, bitcast=True)
                exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)
                scale = (exponent << 23).to(tl.float32, bitcast=True)
                scale = tl.where(exponent >= 255, float("inf"), scale)
                encoded = exponent.to(tl.uint8)
                normalized = tl.div_rn(groups, scale[:, None])
                payload = _pack_e2m1_payload(normalized, dim)
                if ROW_PACK:
                    payload_base = base
                    scale_base = base + 64
                else:
                    payload_base = base + offset * 64
                    scale_base = base + ENTRIES * 64 + offset * 4
                tl.store(payload_base + tl.arange(0, 64), payload)
                tl.store(scale_base + tl.arange(0, 4), encoded)
            else:
                maximum = tl.maximum(maximum, 1.0e-4)
                maximum = tl.where(has_nan, float("nan"), maximum)
                bits = tl.div_rn(maximum, 448.0).to(tl.uint32, bitcast=True)
                encoded = (
                    ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)
                ).to(tl.uint8)
                scale = (encoded.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
                scale = tl.where(encoded == 0, 2.0**-127, scale)
                scale = tl.where(encoded == 255, float("nan"), scale)
                scaled = tl.div_rn(groups, scale[:, None])
                clipped = tl.clamp(scaled, -448.0, 448.0)
                clipped = tl.where(scaled != scaled, scaled, clipped)
                payload = clipped.to(tl.float8e4nv, fp_downcast_rounding="rtne")
                payload = payload.to(tl.uint8, bitcast=True).reshape((512,))
                payload_base = base + offset * 528
                tl.store(payload_base + tl.arange(0, 512), payload)
                tl.store(payload_base + 512 + tl.arange(0, 16), encoded)

    return norm_rope_store


def _launch(
    raw, weight, eps, freqs, slots, pool, generations, entries, indexer, row_pack, *, positions=None, status=None,
):
    if raw.shape[0]:
        _kernel()[(raw.shape[0],)](
            raw,
            weight,
            float(eps),
            torch.view_as_real(freqs),
            positions,
            status,
            slots,
            pool,
            generations,
            raw.stride(0),
            pool.stride(0),
            pool.shape[0],
            freqs.shape[0],
            entries,
            indexer,
            row_pack,
            generations is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )


def _try_pack(raw, weight, eps, freqs, *, indexer):
    dim = INDEXER_HEAD_DIM if indexer else KV_DIM
    width = INDEXER_K_ENTRY_BYTES if indexer else SWA_KV_ENTRY_BYTES
    if not _inputs_supported(raw, weight, eps, freqs, dim):
        return None
    packed = torch.empty((raw.shape[0], width), dtype=torch.uint8, device=raw.device)
    _launch(raw, weight, eps, freqs, None, packed, None, 1, indexer, True)
    return packed


def try_norm_rope_pack_swa(raw_kv, norm_weight, eps, freqs):
    """Return interleaved [T,528] bytes, or None before unsupported work.

    Useful when Prefill owns cache publication/bookkeeping: feed the result
    to its existing packed-row write instead of bypassing that lifecycle.
    """
    return _try_pack(raw_kv, norm_weight, eps, freqs, indexer=False)


def try_norm_rope_pack_indexer(raw_index, norm_weight, eps, freqs):
    """Return [T,68] FP4/UE8M0 rows, or None before unsupported work."""
    return _try_pack(raw_index, norm_weight, eps, freqs, indexer=True)


def _try_store(raw, weight, eps, freqs, cache, layer_id, slots, *, indexer, journal, positions=None, status=None):
    dim = INDEXER_HEAD_DIM if indexer else KV_DIM
    width = INDEXER_K_ENTRY_BYTES if indexer else SWA_KV_ENTRY_BYTES
    tag = "indexer_kv" if indexer else "swa_kv"
    if not (
        type(journal) is bool
        and type(layer_id) is int
        and _inputs_supported(raw, weight, eps, freqs, dim, indexed_freqs=positions is not None)
        and _slots_supported(slots, raw)
    ):
        return False
    if positions is not None and not (
        _slots_supported(positions, raw)
        and _raw(status) and status.device == raw.device and status.dtype == torch.int32
        and status.shape in ((), (1,)) and status.is_contiguous()
    ):
        return False
    from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
    from rtp_llm.models_py.modules.dsv41.ppu.graph_swa_write import (
        _journal_pool,
        _native_pool,
    )

    if journal:
        key = (layer_id, tag)
        payloads = getattr(cache, "payloads", None)
        generations_by_key = getattr(cache, "generations", None)
        if not (
            isinstance(payloads, dict)
            and isinstance(generations_by_key, dict)
            and key in payloads
            and key in generations_by_key
            and _raw(payloads[key])
            and _raw(generations_by_key[key])
        ):
            return False
        target = _journal_pool(cache, layer_id, tag, slots, width)
        if target is None:
            return False
        pool, generations = target
        entries = 1
    else:
        # SharedNativeCache owns extra publication/invalidation state. Its
        # callers use the pack API above and retain their normal write path.
        if not isinstance(cache, NativeGraphCache):
            return False
        if positions is not None and freqs.shape[0] < cache.max_seq_len:
            return False
        if (layer_id, tag) not in cache._pools:
            return False
        target = _native_pool(cache, layer_id, tag, raw.device, width)
        if target is None or not _raw(target[0]):
            return False
        pool, entries = target
        generations = None
    outputs = (pool,) if generations is None else (pool, generations)
    if not _disjoint_writes(outputs, (raw, weight, freqs, slots, positions, status)):
        return False
    _launch(
        raw, weight, eps, freqs, slots, pool, generations, entries, indexer, journal,
        positions=positions, status=status,
    )
    return True


def try_norm_rope_store_swa(
    raw_kv, norm_weight, eps, freqs, cache, layer_id, slots, *, journal=False, positions=None, status=None,
):
    """Fuse raw main K norm/RoPE/FP8 encoding and native or journal storage.

    Native writes can use ``positions`` to index the RoPE table and ``status``
    to suppress publication when the device receipt is nonzero. Out-of-range
    positions also suppress writes.

    Returns False before dispatch if unsupported. Once dispatched, JIT and
    provider errors propagate; the caller must not retry another writer.
    """
    if (positions is None) != (status is None) or (journal is True and positions is not None):
        return False
    return _try_store(
        raw_kv, norm_weight, eps, freqs, cache, layer_id, slots,
        indexer=False, journal=journal, positions=positions, status=status,
    )


def try_norm_rope_store_indexer(
    raw_index, norm_weight, eps, freqs, cache, layer_id, slots, *, journal=False
):
    """Fuse raw Indexer K norm/RoPE/FP4 encoding and native or journal storage.

    Native storage is planar: [entries*64 payload | entries*4 scales].
    Journal storage is row-interleaved [64 payload | 4 scales], matching the
    existing commit writer. The projection feeding this must be pre-norm.
    """
    return _try_store(
        raw_index, norm_weight, eps, freqs, cache, layer_id, slots,
        indexer=True, journal=journal,
    )
