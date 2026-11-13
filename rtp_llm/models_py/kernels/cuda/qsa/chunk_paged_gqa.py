# SPDX-License-Identifier: Apache-2.0
"""Head-grouped QSA for long chunks in RTP's paged HND cache.

The grouping and slot-resolution strategy follows SGLang's
``qsa/sparse_attn.py`` at 297e54fd105e974cdf401e5b4d20253a22c16f90.
Cache addressing, FP8 scales, packed counts, BF16 probabilities
and the fused output gate retain the Qwen4 RTP model contract.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from packaging.version import Version


@lru_cache(maxsize=1)
def _require_supported_triton(version):
    if Version(version) < Version("3.6"):
        raise RuntimeError(
            f"QSA chunk_paged_gqa requires Triton >= 3.6; detected Triton {version}. "
            "Use the CUDA 13 production image or upgrade Triton."
        )


def _require_fp8_toolchain():
    """The caller selects ptxas; library code never changes compiler settings."""
    _require_supported_triton(triton.__version__)
    from triton.backends.nvidia.compiler import get_ptxas

    version = get_ptxas(120).version
    if tuple(map(int, version.split("."))) < (13, 2):
        raise RuntimeError(
            "QSA FP8 chunk prefill requires CUDA 13.2+ ptxas; "
            f"selected ptxas {version}. Set TRITON_PTXAS_BLACKWELL_PATH "
            "to the CUDA 13.2+ ptxas executable."
        )


@triton.jit
def _fp8_to_bf16(x):
    # Four FP8 inputs occupy one register; return two packed BF16 pairs.
    return tl.inline_asm_elementwise(
        """{
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.rn.bf16x2.e4m3x2 $0, lo;
            cvt.rn.bf16x2.e4m3x2 $1, hi;
        }""",
        constraints="=r,=r,r",
        args=[x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _resolve_chunk_slots(
    SELECTED,
    TABLE,
    LENGTHS,
    SLOTS,
    Q: tl.constexpr,
    WIDTH: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    count = (
        tl.minimum(tl.maximum(tl.load(SELECTED + row * ROW_WIDTH + WIDTH), 0), WIDTH)
        if HAS_COUNT
        else WIDTH
    )
    token = tl.load(SELECTED + row * ROW_WIDTH + cols, cols < count, other=-1)
    length = tl.load(LENGTHS + row)
    logical = token // PAGE
    candidate = (
        (cols < count) & (token >= 0) & (token < length) & (logical < TABLE_WIDTH)
    )
    physical = tl.load(TABLE + (row // Q) * TABLE_WIDTH + logical, candidate, other=0)
    valid = candidate & (physical > 0) & (physical < PAGES)
    slot = physical * PAGE + token % PAGE
    tl.store(SLOTS + row * WIDTH + cols, tl.where(valid, slot, -1), cols < WIDTH)


@triton.jit
def _chunk_paged_gqa(
    QPTR,
    CACHE,
    SCALE,
    SLOTS,
    SELECTED,
    OUT,
    GATE,
    QS: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    CP: tl.constexpr,
    CKV: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    SP: tl.constexpr,
    SKV: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    HG: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FP8: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    WITH_GATE: tl.constexpr,
    UNIT_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    batch, query = row // QS, row % QS
    heads = group * HG + tl.arange(0, BM)
    dim = tl.arange(0, D)
    q = tl.load(
        QPTR + batch * Q0 + heads[:, None] * Q1 + query * Q2 + dim[None, :],
        heads[:, None] < (group + 1) * HG,
        other=0.0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    count = (
        tl.minimum(tl.maximum(tl.load(SELECTED + row * ROW_WIDTH + WIDTH), 0), WIDTH)
        if HAS_COUNT
        else WIDTH
    )
    cols = tl.arange(0, BN)
    for start in range(tl.cdiv(count, BN)):
        offset = start * BN + cols
        slot = tl.load(SLOTS + row * WIDTH + offset, offset < count, other=-1)
        valid = (offset < count) & (slot >= 0)
        physical = (slot // PAGE).to(tl.int64)
        token = slot % PAGE
        address = (
            CACHE
            + physical[:, None] * CP
            + group * CH
            + token[:, None] * CT
            + dim[None, :]
        )
        key = tl.load(address, valid[:, None], other=0.0)
        if FP8:
            key = _fp8_to_bf16(key)
        if FP8 and not UNIT_SCALE:
            scale_ptr = SCALE + physical * SP + group * SH + token * ST
            key_scale = tl.load(scale_ptr, valid, other=1.0)
        score = tl.dot(q, tl.trans(key)) * (D**-0.5 * 1.4426950408889634)
        if FP8 and not UNIT_SCALE:
            # A token scale is shared by D. Apply it after QK, preserving the
            # FP32 dequant contract without rounding scaled K back to BF16.
            score *= key_scale[None, :]
        score = tl.where(valid[None, :], score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.where(normalizer > 0, tl.exp2(maximum - safe_max), 0.0)
        prob = tl.where(valid[None, :], tl.exp2(score - safe_max[:, None]), 0.0)
        value = tl.load(address + CKV, valid[:, None], other=0.0)
        if FP8:
            value = _fp8_to_bf16(value)
        weighted_prob = prob
        if FP8 and not UNIT_SCALE:
            value_scale = tl.load(scale_ptr + SKV, valid, other=1.0)
            weighted_prob = prob * value_scale[None, :]
        acc = tl.dot(weighted_prob.to(tl.bfloat16), value, acc * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(prob, 1)
        maximum = next_max
    output = tl.where(normalizer[:, None] > 0, acc / normalizer[:, None], 0.0).to(
        tl.bfloat16
    )
    out_offset = (row * HQ + heads[:, None]) * D + dim[None, :]
    if WITH_GATE:
        gate = tl.load(
            GATE + out_offset, heads[:, None] < (group + 1) * HG, other=0.0
        ).to(tl.float32)
        output = output.to(tl.float32) * tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(
        OUT + out_offset, output.to(tl.bfloat16), heads[:, None] < (group + 1) * HG
    )


def chunk_paged_gqa(
    q,
    cache,
    table,
    lengths,
    selected,
    *,
    scale=None,
    selected_has_count=False,
    output_gate=None,
    block_n=64,
    num_warps=None,
    num_stages=None,
    unit_scale=False,
):
    """Consume validated contiguous paged metadata with Triton >= 3.6; no host sync.

    The public sparse-paged wrapper owns value validation. Slot resolution
    independently masks missing/out-of-range pages and invisible tokens.
    Use a single BF16 PV product with FP32 accumulation.
    Long TP2 FP8 chunks use two warps and one pipeline stage. Explicit warp/stage
    overrides retain the original default for the other launch option.
    ``unit_scale`` promises a direct-cast cache, including all reused prefixes.
    This promise comes from the producer, never from tensor shape or dtype.
    FP8 caches require SM120 and CUDA 13.2+ ptxas for direct BF16 conversion.
    Returns logical [B,H,Q,D] backed by token-major storage.
    """
    _require_supported_triton(triton.__version__)
    b, heads, queries, d = q.shape
    kv_heads, page = cache.shape[2:4]
    width = selected.shape[-1] - int(selected_has_count)
    fp8 = cache.dtype == torch.float8_e4m3fn
    if fp8:
        if torch.version.hip is not None or torch.cuda.get_device_capability(
            q.device
        ) != (12, 0):
            raise RuntimeError("QSA FP8 chunk prefill requires NVIDIA SM120")
        _require_fp8_toolchain()
    if (
        num_warps is None
        and num_stages is None
        and cache.dtype == torch.float8_e4m3fn
        and heads == 12
        and kv_heads == 1
        and queries >= 1024
        and width == 2051
        and block_n == 64
    ):
        # The 2K/3K-query workload benefits from four two-warp CTAs per SM.
        # Short chunks do not have enough CTAs to amortize the slower block.
        num_warps, num_stages = 2, 1
    else:
        num_warps = 4 if num_warps is None else num_warps
        num_stages = 2 if num_stages is None else num_stages
    assert q.dtype == torch.bfloat16 and q.is_cuda and q.stride(-1) == 1
    assert heads % kv_heads == 0 and d == 256 and heads // kv_heads <= 16
    assert (
        table.is_contiguous() and lengths.is_contiguous() and selected.is_contiguous()
    )
    assert cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert scale is not None or cache.dtype == torch.bfloat16 or unit_scale
    slots = torch.empty((b, queries, width), dtype=torch.int32, device=q.device)
    output = torch.empty((b, queries, heads, d), dtype=q.dtype, device=q.device)
    if not b or not queries or not width:
        output.zero_()
        return output.transpose(1, 2)
    _resolve_chunk_slots[(b * queries,)](
        selected,
        table,
        lengths,
        slots,
        Q=queries,
        WIDTH=width,
        ROW_WIDTH=selected.shape[-1],
        TABLE_WIDTH=table.shape[1],
        PAGE=page,
        PAGES=cache.shape[0],
        HAS_COUNT=selected_has_count,
        BLOCK=triton.next_power_of_2(width),
        num_warps=4,
    )
    scale_strides = scale.stride() if scale is not None else (0, 0, 0, 0)
    _chunk_paged_gqa[(b * queries, kv_heads)](
        q,
        cache,
        scale if scale is not None else cache,
        slots,
        selected,
        output,
        output_gate if output_gate is not None else output,
        QS=queries,
        HQ=heads,
        HKV=kv_heads,
        D=d,
        Q0=q.stride(0),
        Q1=q.stride(1),
        Q2=q.stride(2),
        CP=cache.stride(0),
        CKV=cache.stride(1),
        CH=cache.stride(2),
        CT=cache.stride(3),
        SP=scale_strides[0],
        SKV=scale_strides[1],
        SH=scale_strides[2],
        ST=scale_strides[3],
        PAGE=page,
        WIDTH=width,
        ROW_WIDTH=selected.shape[-1],
        HG=heads // kv_heads,
        BM=max(16, triton.next_power_of_2(heads // kv_heads)),
        BN=block_n,
        FP8=fp8,
        HAS_COUNT=selected_has_count,
        WITH_GATE=output_gate is not None,
        UNIT_SCALE=unit_scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output.transpose(1, 2)
