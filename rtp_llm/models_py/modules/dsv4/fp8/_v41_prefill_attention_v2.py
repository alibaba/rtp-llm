"""V2 restructures of the SM120 BF16 sparse prefill kernel (R6-1).

Flag ``DSV41_SPARSE_PREFILL_V2`` (default "0" = legacy kernel untouched):
  "s2"   : legacy kernel launched with num_stages=2 — gathers are pipelined
           (smem 80->49 KB, 2 CTA/SM), arithmetic order unchanged, output
           BITWISE identical to the legacy launch (measured).
  "qt"   : scores_T = dot(kv, qT) formulation with q transposed once per
           CTA — removes the per-tile 32 KB kvT smem copy; QK dot
           accumulation order changes (1-ULP BF16 output differences,
           measured maxabs 4.9e-4, cos 1.0 over 67M elements).
  "qt64" : qt with 64-key tiles (10 serial softmax steps instead of 20) —
           fastest measured (2.338 ms vs 2.957 ms legacy at rows=2048,
           heads=64, width=640, count=16384); 1-ULP BF16 differences on
           ~19% of elements (rescale history changes), maxabs 4.9e-4.

Microbench evidence: r6_mb_sparse_prefill{,2}.py (node artifacts,
mb_sparse_prefill_results.json). num_warps=8 measured slower for every
variant (M=16 dots split poorly); num_stages=3 == num_stages=2.
"""

import os

import torch
import triton
import triton.language as tl

SPARSE_PREFILL_V2_FLAG = "DSV41_SPARSE_PREFILL_V2"
_MODES = ("0", "s2", "qt", "qt64", "h32")


def sparse_prefill_v2_mode() -> str:
    mode = os.environ.get(SPARSE_PREFILL_V2_FLAG, "0")
    return mode if mode in _MODES else "0"


@triton.jit
def _sparse_prefill_qt(
    Q,
    KV,
    IDX,
    LENS,
    SINK,
    OUT,
    COUNT,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    HB: tl.constexpr,
    DIRECT_SINK_DIV_RN: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HB + tl.arange(0, HB)
    columns = tl.arange(0, 512)
    address = row * HEADS * 512 + heads[:, None] * 512 + columns[None, :]
    q = tl.load(Q + address, heads[:, None] < HEADS, other=0)
    qT = tl.trans(q)  # [512,HB] — one copy per CTA, reused every tile
    limit = tl.load(LENS + row)
    acc = tl.full((HB, 512), 0, tl.float32)
    maximum = tl.full((HB,), -float("inf"), tl.float32)
    total = tl.full((HB,), 0, tl.float32)
    for tile in tl.range(0, tl.cdiv(WIDTH, TILE)):
        offset = tile * TILE + tl.arange(0, TILE)
        slot = tl.load(IDX + row * WIDTH + offset, offset < WIDTH, other=-1).to(
            tl.int64
        )
        live = (offset < WIDTH) & (offset < limit)
        tl.device_assert(
            ~live | ((slot >= -1) & (slot < COUNT)), "V4.1 invalid BF16 KV index"
        )
        valid = live & (slot >= 0) & (slot < COUNT)
        kv = tl.load(
            KV + slot[:, None] * 512 + columns[None, :], valid[:, None], other=0
        )
        scores_T = tl.dot(kv, qT) * (512**-0.5)  # [TILE,16]
        scores_T = tl.where(valid[:, None], scores_T, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores_T, 0))
        finite_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - finite_max)
        probabilities_T = tl.exp(scores_T - finite_max[None, :])
        acc = tl.dot(
            tl.trans(probabilities_T).to(tl.bfloat16), kv, acc * alpha[:, None]
        )
        total = total * alpha + tl.sum(probabilities_T, 0)
        maximum = next_max
    sink = tl.load(SINK + heads, heads < HEADS, other=0)
    if DIRECT_SINK_DIV_RN:
        # Test-only reference epilogue (mirrors the legacy kernel).
        has_keys = total > 0
        finite_max = tl.where(has_keys, maximum, 0.0)
        denominator = total + tl.exp(sink - finite_max)
        result = tl.where(has_keys[:, None], tl.div_rn(acc, denominator[:, None]), 0.0)
    else:
        next_max = tl.maximum(maximum, sink)
        alpha = tl.exp(maximum - next_max)
        denominator = total * alpha + tl.exp(sink - next_max)
        result = acc * (alpha / denominator)[:, None]
    tl.store(OUT + address, result, heads[:, None] < HEADS)


def sparse_prefill_attention_v2(query, kv, indices, lengths, sinks, mode):
    """Drop-in replacement for sparse_prefill_attention under the V2 flag."""
    from ._v41_prefill_attention_sm120 import _sparse_prefill

    if not (
        query.is_cuda
        and torch.cuda.get_device_capability(query.device) == (12, 0)
        and query.ndim == 3
        and query.shape[-1] == 512
        and 1 <= query.shape[1] <= 64
        and query.dtype == torch.bfloat16
        and query.is_contiguous()
        and kv.ndim == 3
        and kv.shape[1:] == (1, 512)
        and kv.dtype == torch.bfloat16
        and kv.is_contiguous()
        and indices.ndim == 3
        and indices.shape[:2] == (query.shape[0], 1)
        and 0 < indices.shape[-1] <= 640
        and indices.dtype in (torch.int32, torch.int64)
        and indices.is_contiguous()
        and lengths.shape == (query.shape[0],)
        and lengths.dtype in (torch.int32, torch.int64)
        and lengths.is_contiguous()
        and sinks.shape == (query.shape[1],)
        and sinks.dtype == torch.float32
        and sinks.is_contiguous()
        and all(t.device == query.device for t in (kv, indices, lengths, sinks))
    ):
        raise ValueError("Unsupported SM120 V4.1 sparse BF16 prefill layout")
    output = torch.empty_like(query)
    if not query.shape[0]:
        return output
    if mode == "s2":
        grid = (query.shape[0], triton.cdiv(query.shape[1], 16))
        # Legacy kernel, pipelined gathers. Bitwise-identical arithmetic.
        _sparse_prefill[grid](
            query,
            kv,
            indices,
            lengths,
            sinks,
            output,
            kv.shape[0],
            HEADS=query.shape[1],
            WIDTH=indices.shape[-1],
            num_warps=4,
            num_stages=2,
        )
        return output
    if mode == "h32":
        tile, hb, warps = 32, 32, 8
    elif mode == "qt64":
        tile, hb, warps = 64, 16, 4
    else:
        tile, hb, warps = 32, 16, 4
    grid = (query.shape[0], triton.cdiv(query.shape[1], hb))
    _sparse_prefill_qt[grid](
        query,
        kv,
        indices,
        lengths,
        sinks,
        output,
        kv.shape[0],
        HEADS=query.shape[1],
        WIDTH=indices.shape[-1],
        TILE=tile,
        HB=hb,
        num_warps=warps,
        num_stages=2,
    )
    return output
