"""V2 restructures of the SM120 BF16 sparse prefill kernel (R6-1).

Flag ``DSV41_SPARSE_PREFILL_V2`` (DEFAULT "qt64" since the R24 adoption —
user decision 2026-10-30 14:02 UTC+8; set the env to "s2" for the R21
byte-exact pipelined launch or "0" for the legacy single-stage launch —
unrecognized values fail safe to "0").
R23 evidence for qt64 (A/B pool): median −52.7 ms, 3/3 pairs vs the s2
default; op-level rel_l2 4.4-5.3e-4 vs s2, max_abs 1.56e-2, differ-element
fraction 1.1-3.3% (1-ULP BF16 class); tokens 64/64 + 4/4 warmup, every
argmax matched; ON sha deterministic across sessions.
R21 evidence for s2 (A/B pool): byte-exact vs legacy (bitwise identical
launch, parity sha preserved), median −6.5 ms, 3/3 pairs.
Modes:
  "s2"   : legacy kernel launched with num_stages=2 — gathers are pipelined
           (smem 80->49 KB, 2 CTA/SM), arithmetic order unchanged, output
           BITWISE identical to the legacy launch (measured).
  "qt"   : scores_T = dot(kv, qT) formulation with q transposed once per
           CTA — removes the per-tile 32 KB kvT smem copy; QK dot
           accumulation order changes (1-ULP BF16 output differences,
           measured maxabs 4.9e-4, cos 1.0 over 67M elements).
  "qt64" : qt with 64-key tiles (10 serial softmax steps instead of 20) —
           fastest measured (2.338 ms vs 2.957 ms legacy at rows=2048,
           heads=64, width=640, count=16384); 1-ULP BF16 differences
           (rescale history changes).

Microbench evidence: r6_mb_sparse_prefill{,2}.py (node artifacts,
mb_sparse_prefill_results.json). num_warps=8 measured slower for every
variant (M=16 dots split poorly); num_stages=3 == num_stages=2.
"""

import os

import torch
import triton
import triton.language as tl

SPARSE_PREFILL_V2_FLAG = "DSV41_SPARSE_PREFILL_V2"
SPARSE_PREFILL_V2_VERIFY_FLAG = "DSV41_SPARSE_PREFILL_V2_VERIFY"
_MODES = ("0", "s2", "qt", "qt64", "h32")
_SPV2_ANNOUNCED = False
# R23: op-level numerics for the flip-class modes (qt/qt64/h32) — cumulative
# comparison of the mode output against the s2 launch (which is bitwise the
# legacy output). Verify-only (DSV41_SPARSE_PREFILL_V2_VERIFY=1); timed arms
# run without it.
_SPV2_VERIFY_STATE = {
    "calls": 0,
    "sq_err": 0.0,
    "sq_ref": 0.0,
    "max_abs": 0.0,
    "n_diff": 0,
    "n_elems": 0,
}


def sparse_prefill_v2_mode() -> str:
    # R24 adoption: absent env selects "qt64" (1-ULP flip class, median
    # −52.7 ms 3/3 in R23); "s2" (R21 byte-exact) and "0" (legacy) are the
    # escape hatches; unrecognized values fail safe to "0".
    mode = os.environ.get(SPARSE_PREFILL_V2_FLAG, "qt64")
    return mode if mode in _MODES else "0"


def _spv2_verify_enabled() -> bool:
    return os.environ.get(SPARSE_PREFILL_V2_VERIFY_FLAG, "0") == "1"


def _spv2_announce(mode: str) -> None:
    """R21/A: one-shot engagement proof per process (stderr -> native-rank*.log)."""
    global _SPV2_ANNOUNCED
    if not _SPV2_ANNOUNCED:
        _SPV2_ANNOUNCED = True
        import sys

        print(
            f"[SPV2] mode={mode} engaged ({SPARSE_PREFILL_V2_FLAG})",
            file=sys.stderr,
            flush=True,
        )


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

    _spv2_announce(mode)

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
    if _spv2_verify_enabled() and query.shape[0]:
        _spv2_verify_against_s2(query, kv, indices, lengths, sinks, output, mode)
    return output


def _spv2_verify_against_s2(query, kv, indices, lengths, sinks, output, mode):
    """R23/C: recompute with the s2 launch (bitwise == legacy/default) and
    accumulate op-level numerics: cumulative rel_l2, running max-abs and the
    differ-element fraction (the KDA '~19% of elements' metric)."""
    from ._v41_prefill_attention_sm120 import _sparse_prefill

    ref = torch.empty_like(query)
    _sparse_prefill[(query.shape[0], triton.cdiv(query.shape[1], 16))](
        query,
        kv,
        indices,
        lengths,
        sinks,
        ref,
        kv.shape[0],
        HEADS=query.shape[1],
        WIDTH=indices.shape[-1],
        num_warps=4,
        num_stages=2,
    )
    a = output.float()
    b = ref.float()
    d = a - b
    st = _SPV2_VERIFY_STATE
    st["calls"] += 1
    st["sq_err"] += float((d * d).sum())
    st["sq_ref"] += float((b * b).sum())
    st["max_abs"] = max(st["max_abs"], float(d.abs().max()))
    neq = d != 0
    st["n_diff"] += int(neq.sum())
    st["n_elems"] += int(d.numel())
    if st["calls"] == 1 or st["calls"] % 21 == 0:
        import sys

        rel = (st["sq_err"] ** 0.5) / max(st["sq_ref"] ** 0.5, 1e-30)
        frac = st["n_diff"] / max(st["n_elems"], 1)
        print(
            "[SPV2VERIFY] mode=%s calls=%d op_rel_l2=%.6f op_max_abs=%.6e "
            "diff_elem_frac=%.4f" % (mode, st["calls"], rel, st["max_abs"], frac),
            file=sys.stderr,
            flush=True,
        )
