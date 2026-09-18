"""Fuse V4.1 decode compression, normalization, RoPE and typed cache stores.

The two-stage organization follows vLLM's V4.1 fused compressor/indexer
stores, with RTP's 584-byte main cache and continuous index FP8 scales.
Stage A reads the previous speculative state before any state writes.
Stage B runs after the unchanged BF16 index projection, and commits state.
This kernel boundary prevents a token CTA from overwriting another CTA's
history when the state ring wraps. All metadata stays on the device.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from rtp_llm.models_py.modules.dsv4.attn_type import CSA_STATE, INDEXER_KV
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import (
    invalid_kv_access_validation_enabled,
    trap_invalid_kv_access_enabled,
)


@triton.jit
def _trap():
    tl.inline_asm_elementwise(
        "trap; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _rope_bf16(x, frequencies, position, D: tl.constexpr):
    """GPT-J pairs in the last 64 columns, with both BF16 boundaries."""
    columns = tl.arange(0, D)
    values = x.to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(values, columns ^ 1, axis=0)
    rotary = columns >= D - 64
    angle = tl.maximum((columns - (D - 64)) // 2, 0)
    cosine = tl.load(frequencies + position * 64 + angle * 2, rotary, other=1)
    sine = tl.load(frequencies + position * 64 + angle * 2 + 1, rotary, other=0)
    product = partner * sine
    # Match the CUDA complex multiply used by apply_rotary_emb: its last
    # multiply-add is fused even though normalization retains separate ops.
    rotated = tl.fma(values, cosine, tl.where(columns % 2 == 0, -product, product))
    return tl.where(rotary, rotated, values).to(tl.bfloat16)


@triton.jit
def _cache_address(
    table,
    position,
    request,
    COLS: tl.constexpr,
    STRIDE: tl.constexpr,
    TPB: tl.constexpr,
    RATIO: tl.constexpr,
):
    column = position // TPB
    valid = (position >= 0) & (column < COLS) & ((position + 1) % RATIO == 0)
    block = tl.load(table + request * STRIDE + column, valid, other=0).to(tl.int64)
    return block, (position % TPB) // RATIO, valid & (block > 0)


@triton.jit
def _compress_norm_main_store_kernel(
    values,
    scores,
    norm_weight,
    positions,
    requests,
    starts,
    frequencies,
    state,
    state_table,
    main_cache,
    main_table,
    latent_out,
    S: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    STATE_EB: tl.constexpr,
    STATE_TPB: tl.constexpr,
    STATE_COLS: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    MAIN_EB: tl.constexpr,
    MAIN_TPB: tl.constexpr,
    MAIN_COLS: tl.constexpr,
    MAIN_TABLE_STRIDE: tl.constexpr,
    MAIN_CACHE_STRIDE: tl.constexpr,
    MAIN_BLOCKS: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    TRAP: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, 512)
    position = tl.load(positions + token).to(tl.int64)
    request = tl.load(requests + token).to(tl.int64)
    current = tl.load(values + token * 512 + columns)
    if RATIO == 2:
        current_score = tl.load(scores + token * 512 + columns)
        if token % S == 0:
            previous_position = tl.maximum(
                tl.load(starts + request).to(tl.int64) - 1, 0
            )
            block = tl.load(
                state_table
                + request * STATE_STRIDE
                + (previous_position // STATE_TPB) % STATE_COLS
            ).to(tl.int64)
            if TRAP:
                if block >= STATE_BLOCKS:
                    _trap()
            row = block * STATE_EB + previous_position % STATE_EB
            previous = tl.load(state + row * 1024 + columns, block > 0, other=0)
            previous_score = tl.load(
                state + row * 1024 + 512 + columns, block > 0, other=0
            )
        else:
            previous = tl.load(values + (token - 1) * 512 + columns)
            previous_score = tl.load(scores + (token - 1) * 512 + columns)
        maximum = tl.maximum(previous_score, current_score)
        # Match torch.softmax's expf and rounded division. Approximate exp2
        # and reciprocal can flip BF16 latent values exactly at a midpoint.
        previous_exp = libdevice.exp(previous_score - maximum)
        current_exp = libdevice.exp(current_score - maximum)
        denominator = previous_exp + current_exp
        pooled = (previous * tl.div_rn(previous_exp, denominator)) + (
            current * tl.div_rn(current_exp, denominator)
        )
    else:
        pooled = current
    inverse_rms = tl.rsqrt(tl.sum(pooled * pooled, 0) / 512 + EPS)
    weight = tl.load(norm_weight + columns).to(tl.float32)
    latent = (pooled * inverse_rms * weight).to(tl.bfloat16)
    # wk consumes normalized pre-RoPE BF16 latent, never the main-cache key.
    tl.store(latent_out + token * 512 + columns, latent)

    block, offset, writable = _cache_address(
        main_table,
        position,
        request,
        MAIN_COLS,
        MAIN_TABLE_STRIDE,
        MAIN_TPB,
        RATIO,
    )
    if writable:
        if TRAP:
            if block >= MAIN_BLOCKS:
                _trap()
        rotated = _rope_bf16(latent, frequencies, (position // RATIO) * RATIO, 512)
        # Match quantize_and_insert_k_cache's BF16 input and 64-column UE8M0
        # groups. The eighth tile is RoPE and is not written as FP8.
        grouped = tl.reshape(rotated, (8, 64))
        maxima = tl.maximum(tl.max(tl.abs(grouped), 1), 1e-4)
        exponent = tl.ceil(tl.log2(maxima / 448.0))
        scale = tl.exp2(exponent)
        quantized = tl.clamp(grouped / scale[:, None], -448.0, 448.0).to(tl.float8e4nv)
        base = main_cache + block * MAIN_CACHE_STRIDE
        tl.store(
            base + offset * 576 + columns,
            tl.reshape(quantized, (512,)).to(tl.uint8, bitcast=True),
            columns < 448,
        )
        rope_columns = tl.arange(0, 64)
        rope_values = tl.gather(rotated, rope_columns + 448, axis=0)
        tl.store(
            (base + offset * 576 + 448).to(tl.pointer_type(tl.bfloat16)) + rope_columns,
            rope_values,
        )
        groups = tl.arange(0, 8)
        encoded = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
        tl.store(
            base + MAIN_EB * 576 + offset * 8 + groups,
            tl.where(groups < 7, encoded, 0),
        )


@triton.jit
def _index_norm_store_state_kernel(
    projected,
    norm_weight,
    positions,
    requests,
    starts,
    frequencies,
    index_cache,
    index_table,
    values,
    scores,
    state,
    state_table,
    S: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    INDEX_EB: tl.constexpr,
    INDEX_TPB: tl.constexpr,
    INDEX_COLS: tl.constexpr,
    INDEX_TABLE_STRIDE: tl.constexpr,
    INDEX_CACHE_STRIDE: tl.constexpr,
    STATE_EB: tl.constexpr,
    STATE_TPB: tl.constexpr,
    STATE_COLS: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    INDEX_BLOCKS: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    TRAP: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    position = tl.load(positions + token).to(tl.int64)
    request = tl.load(requests + token).to(tl.int64)
    block, offset, writable = _cache_address(
        index_table,
        position,
        request,
        INDEX_COLS,
        INDEX_TABLE_STRIDE,
        INDEX_TPB,
        RATIO,
    )
    if writable:
        if TRAP:
            if block >= INDEX_BLOCKS:
                _trap()
        columns = tl.arange(0, 128)
        key = tl.load(projected + token * 128 + columns).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(key * key, 0) / 128 + EPS)
        weight = tl.load(norm_weight + columns).to(tl.float32)
        normalized = (key * inverse_rms * weight).to(tl.bfloat16)
        rotated = _rope_bf16(
            normalized, frequencies, (position // RATIO) * RATIO, 128
        ).to(tl.float32)
        # RTP uses continuous FP32 scales here, unlike vLLM's power-of-two
        # index-store option. Scales follow ALL physical key rows in a page.
        scale = tl.maximum(tl.max(tl.abs(rotated), 0) / 448.0, 1e-12)
        quantized = (rotated / scale).to(tl.float8e4nv)
        base = index_cache + block * INDEX_CACHE_STRIDE
        tl.store(base + offset * 128 + columns, quantized.to(tl.uint8, bitcast=True))
        tl.store(
            (base + INDEX_EB * 128 + offset * 4).to(tl.pointer_type(tl.float32)), scale
        )

    if RATIO == 2:
        # Every stage-A CTA has finished reading the old ring. Preserve raw
        # FP32 value/score rows for both accepted and speculative positions.
        # The original suffix mask prevents stale aliases within each block.
        block_in_sequence = position // STATE_TPB
        block = tl.load(
            state_table + request * STATE_STRIDE + block_in_sequence % STATE_COLS
        ).to(tl.int64)
        end = tl.load(starts + request).to(tl.int64) + S
        effective_end = tl.minimum((block_in_sequence + 1) * STATE_TPB, end)
        if (block > 0) & (position + STATE_EB >= effective_end):
            if TRAP:
                if block >= STATE_BLOCKS:
                    _trap()
            row = block * STATE_EB + position % STATE_EB
            columns = tl.arange(0, 512)
            tl.store(
                state + row * 1024 + columns, tl.load(values + token * 512 + columns)
            )
            tl.store(
                state + row * 1024 + 512 + columns,
                tl.load(scores + token * 512 + columns),
            )


def is_supported(attn, x, positions, req_ids, starts) -> bool:
    """Static decode gate; unsupported layouts retain the unmodified path."""
    if os.environ.get("DSV41_FUSED_DECODE_GLOBAL", "1") == "0":
        return False
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 3
        or x.shape[-1] != 5120
        or not x.is_contiguous()
        or not 0 < x.shape[0] * x.shape[1] <= 64
        or attn.compress_ratio not in (1, 2)
        or (attn.head_dim, attn.index_head_dim, attn.rope_head_dim) != (512, 128, 64)
        or attn._block_tables_by_type is None
        or attn._kv_cache is None
        or invalid_kv_access_validation_enabled()
    ):
        return False
    cp = attn._cp_ctx
    if cp is not None and cp.cp_size > 1:
        return False
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        return False
    for metadata, size in (
        (positions, x.shape[0] * x.shape[1]),
        (req_ids, x.shape[0] * x.shape[1]),
        (starts, x.shape[0]),
    ):
        if (
            metadata.device != x.device
            or metadata.dtype not in (torch.int32, torch.int64)
            or metadata.shape != (size,)
            or not metadata.is_contiguous()
        ):
            return False
    weights = [
        (attn.global_wkv, (512, 5120), (torch.bfloat16,)),
        (attn.global_norm, (512,), (torch.float32, torch.bfloat16)),
        (attn.index_wk, (128, 512), (torch.bfloat16,)),
        (attn.index_k_norm, (128,), (torch.float32, torch.bfloat16)),
    ]
    if attn.compress_ratio == 2:
        weights.append((attn.global_wgate, (512, 5120), (torch.bfloat16,)))
    for weight, shape, dtypes in weights:
        if (
            weight.device != x.device
            or weight.dtype not in dtypes
            or weight.shape != shape
            or not weight.is_contiguous()
            or (torch.is_grad_enabled() and weight.requires_grad)
        ):
            return False
    if torch.is_grad_enabled() and x.requires_grad:
        return False
    frequencies = attn.freqs_cis
    if (
        frequencies.device != x.device
        or frequencies.dtype != torch.complex64
        or frequencies.ndim != 2
        or frequencies.shape[1] != 32
        or not frequencies.is_contiguous()
    ):
        return False
    regions = [attn._global_region(), INDEXER_KV]
    if attn.compress_ratio == 2:
        regions.append(CSA_STATE)
    for region in regions:
        pool = attn._source_pool(region)
        table = attn._block_tables_by_type[region]
        if (
            pool is None
            or pool.device != x.device
            or table.device != x.device
            or table.dtype not in (torch.int32, torch.int64)
            or table.ndim != 2
            or table.shape[0] < x.shape[0]
            or table.shape[1] == 0
            or table.stride(1) != 1
        ):
            return False
        tpb = require_pool_tokens_per_block(attn._kv_cache, region=region)
        if region == CSA_STATE:
            eb = attn._source_entries(region, pool)
            if (
                pool.dtype != torch.float32
                or pool.ndim != 2
                or pool.shape[1] != 1024
                or not pool.is_contiguous()
                or eb <= 0
                or pool.shape[0] % eb
                or x.shape[1] > eb
                or tpb <= 0
            ):
                return False
        elif (
            pool.dtype != torch.uint8
            or pool.ndim != 3
            or (region == INDEXER_KV and not pool.is_contiguous())
            or pool.shape[2] != (132 if region == INDEXER_KV else 584)
            or pool.stride(2) != 1
            or pool.stride(1) != pool.shape[2]
            or pool.stride(0) < pool.shape[1] * pool.shape[2]
            or tpb <= 0
            or tpb % attn.compress_ratio
            or tpb // attn.compress_ratio > pool.shape[1]
        ):
            return False
    return True


def try_produce_global(attn, x, positions, req_ids, starts) -> bool:
    """Write main/index/state pools, or return False without mutations.

    The caller retains ownership of the shared paged-index descriptor. GEMM
    inputs, output dtypes and accumulation are unchanged; only their epilogues
    and metadata arithmetic are fused. Runtime/JIT failures propagate.
    """
    if not is_supported(attn, x, positions, req_ids, starts):
        return False
    from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32

    batch, span, _ = x.shape
    tokens = batch * span
    values = _linear_bf16_bf16_fp32(x.view(tokens, -1), attn.global_wkv)
    scores = (
        _linear_bf16_bf16_fp32(x.view(tokens, -1), attn.global_wgate)
        if attn.compress_ratio == 2
        else values
    )
    latent = torch.empty((tokens, 512), device=x.device, dtype=torch.bfloat16)
    main_region = attn._global_region()
    main = attn._source_pool(main_region)
    main_table = attn._block_tables_by_type[main_region]
    index = attn._source_pool(INDEXER_KV)
    index_table = attn._block_tables_by_type[INDEXER_KV]
    if attn.compress_ratio == 2:
        state = attn._source_pool(CSA_STATE)
        state_table = attn._block_tables_by_type[CSA_STATE]
        state_eb = attn._source_entries(CSA_STATE, state)
        state_tpb = require_pool_tokens_per_block(attn._kv_cache, region=CSA_STATE)
    else:
        state, state_table, state_eb, state_tpb = values, main_table, 1, 1
    state_geometry = dict(
        STATE_EB=state_eb,
        STATE_TPB=state_tpb,
        STATE_COLS=state_table.shape[1],
        STATE_STRIDE=state_table.stride(0),
        STATE_BLOCKS=state.shape[0] // state_eb,
        TRAP=trap_invalid_kv_access_enabled(),
    )
    _compress_norm_main_store_kernel[(tokens,)](
        values,
        scores,
        attn.global_norm,
        positions,
        req_ids,
        starts,
        attn.freqs_cis.view(torch.float32),
        state,
        state_table,
        main,
        main_table,
        latent,
        S=span,
        RATIO=attn.compress_ratio,
        EPS=attn.eps,
        MAIN_EB=main.shape[1],
        MAIN_TPB=require_pool_tokens_per_block(attn._kv_cache, region=main_region),
        MAIN_COLS=main_table.shape[1],
        MAIN_TABLE_STRIDE=main_table.stride(0),
        MAIN_CACHE_STRIDE=main.stride(0),
        MAIN_BLOCKS=main.shape[0],
        **state_geometry,
        num_warps=4,
        enable_fp_fusion=False,
    )
    projected = F.linear(latent, attn.index_wk)
    _index_norm_store_state_kernel[(tokens,)](
        projected,
        attn.index_k_norm,
        positions,
        req_ids,
        starts,
        attn.freqs_cis.view(torch.float32),
        index,
        index_table,
        values,
        scores,
        state,
        state_table,
        S=span,
        RATIO=attn.compress_ratio,
        EPS=attn.eps,
        INDEX_EB=index.shape[1],
        INDEX_TPB=require_pool_tokens_per_block(attn._kv_cache, region=INDEXER_KV),
        INDEX_COLS=index_table.shape[1],
        INDEX_TABLE_STRIDE=index_table.stride(0),
        INDEX_CACHE_STRIDE=index.stride(0),
        INDEX_BLOCKS=index.shape[0],
        **state_geometry,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return True
