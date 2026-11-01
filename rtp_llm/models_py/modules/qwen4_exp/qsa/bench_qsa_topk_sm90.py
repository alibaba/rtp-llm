"""SM90 gate for the fused QSA decode topk (step 2).

Bench: fused _packed_qsa_tokens vs the native path (torch.topk + expand) at
decode shapes. Parity: outputs must match int32-exactly.
"""

import statistics
import struct
import sys

import torch
import triton

from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
    _expand_tokens,
    try_select_qsa_tokens,
)

FAILURES = []


def native_select(scores, lengths, finite_only=True):
    """Replicate the pre-fusion fallback path (c)."""
    rows, columns = scores.shape
    values, blocks = torch.topk(scores, min(512, columns), dim=-1)
    output = torch.empty((rows, 2051), device=scores.device, dtype=torch.int32)
    _expand_tokens[(rows,)](
        values,
        blocks,
        lengths,
        output,
        K=values.shape[1],
        LENGTH_STRIDE=lengths.stride(0),
        FINITE_ONLY=finite_only,
        num_warps=4,
    )
    return output


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(start, end))
    return statistics.median(times)


def make_scores(rows, columns, seed, edge=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(rows, columns, device="cuda", generator=g)
    if edge:
        scores[0, : columns // 2] = 1.0  # heavy ties
        if rows > 1:
            scores[1, 0] = float("nan")
            scores[1, 1] = float("inf")
            scores[1, 2] = float("-inf")
            scores[1, 3] = 0.0
            scores[1, 4] = -0.0
            # Sign-bit NaN: torch.topk placement is an implementation detail
            # that must be matched bit-exactly, not assumed.
            scores[1, 5] = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]
    return scores


def main():
    assert torch.cuda.is_available()
    print(f"device cap: {torch.cuda.get_device_capability()}")
    rows_grid = (1, 2, 4, 8, 16, 32)
    cols_grid = (1024, 1536, 2048, 3072, 4096, 6144, 8192)
    print(f"{'rows':>4} {'cols':>5} {'native us':>9} {'fused us':>9} {'speedup':>7}  parity")
    for columns in cols_grid:
        for rows in rows_grid:
            seed = 9000 + rows * 100 + columns
            scores = make_scores(rows, columns, seed)
            lengths = torch.randint(
                1, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
            )
            # Mask like the paged scorer: -inf past each completed-block limit.
            complete = lengths // 4
            col_ids = torch.arange(columns, device="cuda")
            masked = torch.where(
                col_ids[None, :] < complete[:, None],
                scores,
                torch.full_like(scores, float("-inf")),
            ).contiguous()
            fused = try_select_qsa_tokens(
                masked,
                lengths,
                compress_ratio=4,
                token_budget=2048,
                scores_are_masked=True,
                finite_only=True,
            )
            native = native_select(masked, lengths)
            if fused is None:
                print(f"{rows:>4} {columns:>5}  fused path not taken (gate)")
                FAILURES.append(f"gate rows={rows} cols={columns}")
                continue
            parity = torch.equal(fused, native)
            if not parity:
                FAILURES.append(f"parity rows={rows} cols={columns}")
            t_native = bench(lambda: native_select(masked, lengths))
            t_fused = bench(lambda: try_select_qsa_tokens(
                masked, lengths, compress_ratio=4, token_budget=2048,
                scores_are_masked=True, finite_only=True))
            print(
                f"{rows:>4} {columns:>5} {t_native:>9.1f} {t_fused:>9.1f} "
                f"{t_native / t_fused:>6.2f}x  {'OK' if parity else 'MISMATCH'}"
            )
    # Edge-case parity sweep.
    for columns in (1024, 2048, 6144):
        for rows in (1, 4, 16):
            scores = make_scores(rows, columns, 777 + columns, edge=True)
            lengths = torch.randint(
                1, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
            )
            lengths[0] = columns * 4
            if rows > 1:
                # Keep the injected inf/NaN columns inside the visible range.
                lengths[1] = columns * 4
            complete = lengths // 4
            col_ids = torch.arange(columns, device="cuda")
            masked = torch.where(
                col_ids[None, :] < complete[:, None],
                scores,
                torch.full_like(scores, float("-inf")),
            ).contiguous()
            fused = try_select_qsa_tokens(
                masked, lengths, compress_ratio=4, token_budget=2048,
                scores_are_masked=True, finite_only=True,
            )
            native = native_select(masked, lengths)
            ok = fused is not None and torch.equal(fused, native)
            print(f"edge rows={rows} cols={columns}: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                FAILURES.append(f"edge rows={rows} cols={columns}")

    # Production-signature parity: the indexer calls with raw (unmasked)
    # relu-style scores and finite_only=False; the kernel masks internally and
    # must match the masked native reference exactly.
    for columns in (1024, 1536, 2048, 3072, 4096, 6144, 8192):
        for rows in (1, 2, 8):
            g = torch.Generator(device="cuda").manual_seed(5000 + rows * 100 + columns)
            raw = torch.relu(torch.randn(rows, columns, device="cuda", generator=g))
            if rows >= 2:
                raw[1, 0] = float("inf")
                raw[1, 1] = float("nan")
            if rows >= 8:
                raw[7, :] = 0.0
                lengths7 = 3
            lengths = torch.randint(
                1, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
            )
            if rows >= 8:
                lengths[7] = lengths7
            complete = lengths // 4
            col_ids = torch.arange(columns, device="cuda")
            masked = torch.where(
                col_ids[None, :] < complete[:, None],
                raw,
                torch.full_like(raw, float("-inf")),
            ).contiguous()
            fused = try_select_qsa_tokens(
                raw, lengths, compress_ratio=4, token_budget=2048, finite_only=False,
            )
            native = native_select(masked, lengths, finite_only=False)
            ok = fused is not None and torch.equal(fused, native)
            print(f"prod rows={rows} cols={columns}: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                FAILURES.append(f"prod rows={rows} cols={columns}")
    print("ALL-PASS" if not FAILURES else "FAILURES: " + ", ".join(FAILURES))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
