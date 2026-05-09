"""UT for ``rtp_llm_ops.dsv4_top_k_per_row_prefill``.

Vendored from vLLM (csrc/sampler.cu::top_k_per_row_prefill).  Hybrid kernel:
the first 12288 rows use insertion-sort blocks, the remainder use
radix-select blocks.

Op contract:
  logits      : [num_rows, max_T] float32, only [row_starts[r], row_ends[r])
                 read along dim 1
  row_starts  : [num_rows] int32 — inclusive begin per row
  row_ends    : [num_rows] int32 — exclusive end per row
  indices_out : [num_rows, K] int32 — written; positions past the row's valid
                 count are -1 padded.  Indices are RELATIVE TO row_starts[r]
                 (i.e. in 0..rowEnd-rowStart-1).
  num_rows    : == logits.size(0)
  stride0     : logits.stride(0)
  stride1     : logits.stride(1)
  top_k       : K

Order across the valid prefix is unspecified — compare as sets.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_dsv4_top_k_per_row_prefill.py
"""

from __future__ import annotations

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops

_HAS_OP = hasattr(rtp_llm_ops, "dsv4_top_k_per_row_prefill")
_HAS_INDEXED_OP = hasattr(rtp_llm_ops, "dsv4_top_k_per_row_prefill_indexed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make(num_rows: int, max_T: int, lens, *, seed: int = 0):
    """Build (logits[num_rows, max_T], row_starts, row_ends).

    ``lens`` is an iterable of per-row valid lengths (rowEnd - rowStart).
    Each row's valid window starts at column 0 (so row_starts == 0).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_rows, max_T, device="cuda", generator=g)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.tensor(list(lens), dtype=torch.int32, device="cuda")
    return logits, row_starts, row_ends


def _run(logits, row_starts, row_ends, K):
    N = logits.size(0)
    out = torch.full((N, K), -1, dtype=torch.int32, device=logits.device)
    rtp_llm_ops.dsv4_top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        out,
        N,
        logits.stride(0),
        logits.stride(1),
        K,
    )
    return out


def _run_indexed(logits, row_starts, row_ends, row_indices, K):
    N = logits.size(0)
    out = torch.full((N, K), -1, dtype=torch.int32, device=logits.device)
    rtp_llm_ops.dsv4_top_k_per_row_prefill_indexed(
        logits,
        row_starts,
        row_ends,
        row_indices,
        out,
        N,
        logits.stride(0),
        logits.stride(1),
        K,
    )
    return out


def _assert_equiv(out, logits, row_starts, row_ends, K, *, tag: str):
    N = logits.size(0)
    out_h = out.cpu()
    rs_h = row_starts.cpu().tolist()
    re_h = row_ends.cpu().tolist()
    for r in range(N):
        s, e = rs_h[r], re_h[r]
        L = max(0, e - s)
        keep = min(K, L)
        # Padding past keep must be -1.
        pad = out_h[r, keep:]
        assert (pad == -1).all(), f"{tag}: row {r} pad not -1: {pad.tolist()[:8]}..."
        valid = out_h[r, :keep].tolist()
        assert -1 not in valid, f"{tag}: row {r} valid prefix has -1: {valid[:8]}..."
        assert len(set(valid)) == len(
            valid
        ), f"{tag}: row {r} duplicates in valid prefix"
        if keep == 0:
            continue
        # Reference: torch.topk over the row's valid window.
        ref_vals, ref_idx = logits[r, s:e].topk(keep, dim=-1)
        ref_set = set(ref_idx.tolist())
        valid_set = set(valid)
        # Threshold comparison: values at the kernel's selected indices vs the
        # K-th torch.topk value.  For ties, the set may differ but the values
        # at the boundary must match.
        if valid_set != ref_set:
            kernel_vals = logits[r, s:e].cpu()[valid].sort(descending=True).values
            ref_vals_sorted = ref_vals.cpu().sort(descending=True).values
            torch.testing.assert_close(
                kernel_vals,
                ref_vals_sorted,
                atol=0.0,
                rtol=0.0,
                msg=lambda m: (
                    f"{tag}: row {r} top-{keep} mismatch.\n"
                    f"  kernel - ref: {valid_set - ref_set}\n"
                    f"  ref - kernel: {ref_set - valid_set}\n"
                    f"  {m}"
                ),
            )
    print(f"  [{tag}] N={N} K={K} OK")


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def test_small_batch_uniform():
    """N small, all rows full-length."""
    logits, rs, re = _make(8, 4096, [4096] * 8, seed=0)
    out = _run(logits, rs, re, K=128)
    _assert_equiv(out, logits, rs, re, K=128, tag="uniform N=8 T=4096 K=128")


def test_variable_lengths_short_long_mix():
    """Cu-seqlen-style: short rows + long rows mixed."""
    lens = [128, 4096, 256, 8192, 64, 2048, 1024, 16]
    logits, rs, re = _make(len(lens), 8192, lens, seed=1)
    out = _run(logits, rs, re, K=64)
    _assert_equiv(out, logits, rs, re, K=64, tag="varlen mix")


def test_short_rows_padding():
    """Some rows have valid_len < K — exercises -1 padding."""
    lens = [16, 32, 8, 100, 1, 200]  # K=64 so first 3 + idx 4 trigger padding
    logits, rs, re = _make(len(lens), 256, lens, seed=2)
    out = _run(logits, rs, re, K=64)
    _assert_equiv(out, logits, rs, re, K=64, tag="short rows")


def test_long_T_radix_inside_block():
    """Single-block path with a long row — exercises histogram radix steps."""
    logits, rs, re = _make(2, 65536, [65536, 32768], seed=3)
    out = _run(logits, rs, re, K=512)
    _assert_equiv(out, logits, rs, re, K=512, tag="single-row long T")


def test_radix_branch_above_threshold():
    """num_rows > 12288 forces the radix-block launch for the tail rows."""
    N = 12288 + 64
    lens = [512] * N
    logits, rs, re = _make(N, 512, lens, seed=4)
    out = _run(logits, rs, re, K=64)
    _assert_equiv(out, logits, rs, re, K=64, tag="N>12288 radix tail")


def test_indexed_row_schedule_writes_original_rows():
    """Explicit row schedule changes launch order only, not output rows."""
    if not _HAS_INDEXED_OP:
        print("  [indexed schedule] SKIP: op not built")
        return
    lens = [128, 4096, 256, 8192, 64, 2048, 1024, 16]
    logits, rs, re = _make(len(lens), 8192, lens, seed=41)
    row_indices = torch.tensor(
        [7, 0, 6, 1, 5, 2, 4, 3], dtype=torch.int32, device="cuda"
    )
    out = _run_indexed(logits, rs, re, row_indices, K=64)
    _assert_equiv(out, logits, rs, re, K=64, tag="indexed schedule")


def test_zero_length_row():
    """row_starts == row_ends → all -1."""
    logits, rs, re = _make(4, 1024, [1024, 0, 512, 0], seed=5)
    out = _run(logits, rs, re, K=128)
    _assert_equiv(out, logits, rs, re, K=128, tag="zero-len rows")


# ---------------------------------------------------------------------------
# Bench vs torch.topk
# ---------------------------------------------------------------------------
def _bench(fn, *args, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters  # ms


def bench_prefill_sweep():
    print("\n  prefill sweep — kernel vs torch.topk")
    print(
        "    {:>5}  {:>6}  {:>5}  {:>10}  {:>10}  {:>10}".format(
            "N", "T", "K", "torch", "kernel", "speedup"
        )
    )
    cases = [
        (8, 2048, 64),
        (32, 2048, 128),
        (128, 2048, 64),
        (512, 2048, 64),
        (4096, 2048, 64),
        (16384, 2048, 64),  # crosses kSortingAlgorithmThreshold
    ]
    for N, T, K in cases:
        logits, rs, re = _make(N, T, [T] * N, seed=100)
        out = torch.full((N, K), -1, dtype=torch.int32, device="cuda")

        def run_kernel():
            rtp_llm_ops.dsv4_top_k_per_row_prefill(
                logits,
                rs,
                re,
                out,
                N,
                logits.stride(0),
                logits.stride(1),
                K,
            )

        def run_torch():
            return logits.topk(K, dim=-1)[1]

        t_t = _bench(run_torch)
        t_k = _bench(run_kernel)
        speedup = t_t / t_k if t_k > 0 else float("inf")
        print(
            f"    {N:5d}  {T:6d}  {K:5d}  {t_t*1e3:8.2f}us  {t_k*1e3:8.2f}us  "
            f"{speedup:8.2f}x"
        )


if __name__ == "__main__":
    if not _HAS_OP:
        print(
            "SKIP: rtp_llm_ops.dsv4_top_k_per_row_prefill not built — "
            "rebuild //rtp_llm:rtp_compute_ops"
        )
        raise SystemExit(0)
    print("== Correctness ==")
    test_small_batch_uniform()
    test_variable_lengths_short_long_mix()
    test_short_rows_padding()
    test_long_T_radix_inside_block()
    test_radix_branch_above_threshold()
    test_indexed_row_schedule_writes_original_rows()
    test_zero_length_row()
    print("\n== Benchmark ==")
    bench_prefill_sweep()
    print("\nOK")
