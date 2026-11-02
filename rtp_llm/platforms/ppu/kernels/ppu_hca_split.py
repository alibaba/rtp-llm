"""Small-batch HCA decode compression with the original quantized KV bytes."""
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    _trap_invalid_kv_access,
)


@triton.jit
def _hca_compress_columns(state, reqs, positions, table, compressed,
                          raw_kv, raw_score, ape, starts, cu,
                          STATE_STRIDE: tl.constexpr, ROW_STRIDE: tl.constexpr,
                          TABLE_STRIDE: tl.constexpr, STATE_BLOCK: tl.constexpr,
                          RING: tl.constexpr, NUM_STATE_BLOCKS: tl.constexpr,
                          TRAP: tl.constexpr, RAW: tl.constexpr,
                          RAW_KV_STRIDE: tl.constexpr, RAW_SCORE_STRIDE: tl.constexpr,
                          APE_STRIDE: tl.constexpr):
    token = tl.program_id(0)
    position = tl.load(positions + token).to(tl.int64)
    if (position + 1) % 128 != 0:
        return
    req = tl.load(reqs + token).to(tl.int64)
    pos = position - 127 + tl.arange(0, 128)
    use_raw = tl.full((128,), False, tl.int1)
    if RAW:
        start = tl.load(starts + req).to(tl.int64)
        lo = tl.load(cu + req).to(tl.int64)
        hi = tl.load(cu + req + 1).to(tl.int64)
        offset = pos - start
        use_raw = (pos >= 0) & (offset >= 0) & (offset < hi - lo)
        raw_row = tl.where(use_raw, lo + offset, 0)
    use_cache = (pos >= 0) & ~use_raw
    block = tl.load(table + req * TABLE_STRIDE + (pos // STATE_BLOCK) % TABLE_STRIDE,
                    mask=use_cache, other=0).to(tl.int64)
    if tl.max((block >= NUM_STATE_BLOCKS).to(tl.int32), axis=0) != 0:
        _trap_invalid_kv_access(TRAP)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    valid = use_cache & (block > 0)
    block = tl.where(valid, block, 0)
    ptr = state + block[:, None] * STATE_STRIDE + (pos % RING)[:, None] * ROW_STRIDE + cols[None, :]
    if RAW:
        kv_ptr = tl.where(use_raw[:, None], raw_kv + raw_row[:, None] * RAW_KV_STRIDE + cols[None, :], ptr)
        score_ptr = tl.where(use_raw[:, None], raw_score + raw_row[:, None] * RAW_SCORE_STRIDE + cols[None, :], ptr + 512)
        valid_source = valid | use_raw
        kv = tl.load(kv_ptr, mask=valid_source[:, None], other=0.0)
        score = tl.load(score_ptr, mask=valid_source[:, None], other=float('-inf'))
        ape_values = tl.load(ape + (pos % 128)[:, None] * APE_STRIDE + cols[None, :],
                             mask=use_raw[:, None], other=0.0)
        score += ape_values
    else:
        kv = tl.load(ptr, mask=valid[:, None], other=0)
        score = tl.load(ptr + 512, mask=valid[:, None], other=float('-inf'))
    value = tl.sum(kv * tl.softmax(score, dim=0), axis=0)
    tl.store(compressed + token * 512 + cols, value)


@triton.jit
def _hca_normalize_store(compressed, positions, weight, cache_cs, slots, cache,
                         EPS: tl.constexpr, CS_STRIDE: tl.constexpr,
                         KV_BLOCK: tl.constexpr, KV_STRIDE: tl.constexpr,
                         NUM_KV_BLOCKS: tl.constexpr, TRAP: tl.constexpr):
    token = tl.program_id(0)
    position = tl.load(positions + token).to(tl.int64)
    if (position + 1) % 128 != 0:
        return
    slot = tl.load(slots + token)
    if slot < 0:
        return
    block = slot // KV_BLOCK
    if block >= NUM_KV_BLOCKS:
        _trap_invalid_kv_access(TRAP)
    cols = tl.arange(0, 512)
    value = tl.load(compressed + token * 512 + cols)
    w = tl.load(weight + cols)
    variance = tl.sum(value * value, axis=0) / 512
    norm = value * (1.0 / tl.sqrt(variance + EPS)) * w
    offset = slot % KV_BLOCK
    fp8_ptr = cache + block.to(tl.int64) * KV_STRIDE + offset * 576
    scale_ptr = cache + block.to(tl.int64) * KV_STRIDE + KV_BLOCK * 576 + offset * 8
    q = norm.to(tl.bfloat16).to(tl.float32).reshape(8, 64)
    maximum = tl.maximum(tl.max(tl.abs(q), axis=1), 1e-4)
    exponent = tl.ceil(tl.log2(maximum * (1.0 / 448.0)))
    payload = tl.clamp(q * tl.exp2(-exponent)[:, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(fp8_ptr + cols, payload.to(tl.uint8, bitcast=True).reshape(512), mask=cols < 448)
    groups = tl.arange(0, 8)
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0).to(tl.uint8)
    tl.store(scale_ptr + groups, encoded, mask=groups < 7)
    tl.store(scale_ptr + 7, tl.full((), 0, tl.uint8))
    even, odd = tl.split(norm.reshape(256, 2))
    pairs = tl.arange(0, 256)
    rope_pair = pairs - 224
    cs = cache_cs + (position // 128) * 128 * CS_STRIDE
    cosine = tl.load(cs + tl.maximum(rope_pair, 0), mask=rope_pair >= 0, other=1.0)
    sine = tl.load(cs + 32 + tl.maximum(rope_pair, 0), mask=rope_pair >= 0, other=0.0)
    result = tl.interleave(even * cosine - odd * sine, odd * cosine + even * sine)
    bf16 = (fp8_ptr + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(bf16 + cols - 448, result.to(tl.bfloat16), mask=cols >= 448)


def hca_decode_split(state, reqs, positions, table, weight, eps, cs, slots, cache,
                     state_tokens_per_block, trap, *, raw_kv=None, raw_score=None,
                     ape=None, starts=None, cu=None):
    use_raw = raw_kv is not None
    if not use_raw:
        raw_kv = raw_score = ape = state
        starts = cu = positions
    scratch = torch.empty((positions.numel(), 512), device=state.device, dtype=torch.float32)
    _hca_compress_columns[(positions.numel(), 4)](
        state, reqs, positions, table, scratch, raw_kv, raw_score, ape, starts, cu,
        state.stride(0), state.stride(1),
        table.stride(0), state_tokens_per_block, state.shape[1], state.shape[0], trap,
        use_raw, raw_kv.stride(0), raw_score.stride(0), ape.stride(0),
        num_warps=4)
    _hca_normalize_store[(positions.numel(),)](
        scratch, positions, weight, cs, slots, cache, eps, cs.stride(0),
        cache.shape[1], cache.stride(0), cache.shape[0], trap, num_warps=4)
