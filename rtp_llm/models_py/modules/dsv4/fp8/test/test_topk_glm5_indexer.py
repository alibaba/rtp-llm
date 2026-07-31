"""UT for ``rtp_llm_ops.topk_glm5_indexer``.

Adapted from the exact SGLang radix-select TopK in
``e7b190c165b2edaa92bbc`` for the GLM5 indexer decode hot path.

Op contract (mirrors vLLM):
  logits   : [N, T]  float32 row-contiguous; stride(0) may exceed T
  lengths  : [N]     int32     — per-row valid count; positions past
                                  ``lengths[r]`` are written as -1 in output
  output   : [N, K]  int32     — written; ordering of valid indices is NOT
                                  guaranteed (compare as sets, not lists)
  workspace: contiguous CUDA uint8 tensor, retained for old-op ABI compatibility
  K        : 512, 1024, or 2048 (compile-time dispatch in the kernel)
  max_seq_len: max possible T across rows; controls cooperative launch path

Equivalence semantics:
  topk_set(logits[r, : lengths[r]]) == set(output[r, :min(K, lengths[r])])
  output[r, k] == -1 for k >= min(K, lengths[r])
  Order across the valid prefix is unspecified.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/fp8/test/test_topk_glm5_indexer.py
"""

from __future__ import annotations

from typing import Tuple

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops

# When the .so doesn't have the binding yet (pre-rebuild), exit cleanly so
# CI / local runs report SKIP rather than ImportError.
_HAS_OP = hasattr(rtp_llm_ops, "topk_glm5_indexer")

WORKSPACE_BYTES = 1024 * 1024  # matches RADIX_TOPK_WORKSPACE_SIZE


# ---------------------------------------------------------------------------
# Reference: torch.topk over each row's valid prefix; -1 padding past it.
# Returns the SET of valid indices per row, since kernel order is unspecified.
# ---------------------------------------------------------------------------
def ref_topk_sets(logits: torch.Tensor, lengths: torch.Tensor, k: int):
    N, T = logits.shape
    sets = []
    for r in range(N):
        L = int(lengths[r].item())
        if L == 0:
            sets.append(set())
            continue
        keep = min(k, L)
        idxs = logits[r, :L].topk(keep, dim=-1)[1].tolist()
        sets.append(set(idxs))
    return sets


def _run(logits, lengths, k, max_seq_len) -> torch.Tensor:
    N, T = logits.shape
    out = torch.full((N, k), -1, dtype=torch.int32, device=logits.device)
    ws = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=logits.device)
    rtp_llm_ops.topk_glm5_indexer(logits, lengths, out, ws, k, max_seq_len)
    return out


def _assert_equiv(
    out: torch.Tensor, logits: torch.Tensor, lengths: torch.Tensor, k: int, *, tag: str
):
    N, _ = logits.shape
    ref = ref_topk_sets(logits, lengths, k)
    out_h = out.cpu()
    for r in range(N):
        L = int(lengths[r].item())
        keep = min(k, L)
        # Padding contract: positions past `keep` must be -1.
        pad = out_h[r, keep:]
        assert (pad == -1).all(), f"{tag}: row {r} pad not -1: {pad.tolist()[:8]}..."
        # Valid prefix: indices form a set equal to torch.topk's set.
        valid = out_h[r, :keep].tolist()
        assert -1 not in valid, f"{tag}: row {r} valid prefix contains -1"
        assert len(set(valid)) == len(
            valid
        ), f"{tag}: row {r} duplicates in valid prefix"
        assert set(valid) == ref[r], (
            f"{tag}: row {r} top-{keep} mismatch.\n"
            f"  kernel set diff (kernel - ref): {set(valid) - ref[r]}\n"
            f"  kernel set diff (ref - kernel): {ref[r] - set(valid)}"
        )
    print(f"  [{tag}] N={N} k={k} L_max={int(lengths.max())} OK")


def _assert_value_equiv(
    out: torch.Tensor,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    *,
    tag: str,
):
    """Check top-k equivalence without requiring identical indices for ties."""
    N, _ = logits.shape
    out_h = out.cpu()
    for r in range(N):
        L = int(lengths[r].item())
        keep = min(k, L)
        valid = out_h[r, :keep].long()
        pad = out_h[r, keep:]

        assert (pad == -1).all(), f"{tag}: row {r} pad not -1: {pad.tolist()[:8]}..."
        assert ((valid >= 0) & (valid < L)).all(), f"{tag}: row {r} has an invalid index"
        assert torch.unique(valid).numel() == keep, f"{tag}: row {r} has duplicate indices"

        actual = logits[r, valid.to(logits.device)].sort().values
        expected = logits[r, :L].topk(keep, sorted=False).values.sort().values
        assert torch.equal(actual, expected), (
            f"{tag}: row {r} top-{keep} value multiset mismatch.\n"
            f"  actual tail: {actual[-8:].tolist()}\n"
            f"  expected tail: {expected[-8:].tolist()}"
        )
    print(f"  [{tag}] N={N} k={k} L_max={int(lengths.max())} OK")


def _assert_replay_value_equiv(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    max_seq_len: int,
    *,
    replays: int,
    tag: str,
):
    """Replay one input while reusing output/workspace and check exact values."""
    N, _ = logits.shape
    host_lengths = lengths.cpu().tolist()
    expected = [
        logits[r, : int(length)]
        .topk(min(k, int(length)), sorted=False)
        .values.sort()
        .values
        for r, length in enumerate(host_lengths)
    ]
    sentinel = torch.iinfo(torch.int32).min
    out = torch.full((N, k), sentinel, dtype=torch.int32, device=logits.device)
    ws = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=logits.device)

    for replay in range(replays):
        out.fill_(sentinel)
        rtp_llm_ops.topk_glm5_indexer(logits, lengths, out, ws, k, max_seq_len)
        torch.cuda.synchronize()

        out_h = out.cpu()
        for r, length in enumerate(host_lengths):
            length = int(length)
            keep = min(k, length)
            valid = out_h[r, :keep].long()
            pad = out_h[r, keep:]
            assert (pad == -1).all(), (
                f"{tag}: replay {replay}, row {r} pad is not -1"
            )
            assert ((valid >= 0) & (valid < length)).all(), (
                f"{tag}: replay {replay}, row {r} has an invalid index"
            )
            assert torch.unique(valid).numel() == keep, (
                f"{tag}: replay {replay}, row {r} has duplicate indices"
            )
            actual = logits[r, valid.to(logits.device)].sort().values
            assert torch.equal(actual, expected[r]), (
                f"{tag}: replay {replay}, row {r} selected wrong values"
            )

    print(f"  [{tag}] {replays}/{replays} replays OK")


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def _make(
    N: int, T: int, *, seed: int = 0, lengths_mode: str = "full"
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(N, T, device="cuda", generator=g)
    if lengths_mode == "full":
        lengths = torch.full((N,), T, dtype=torch.int32, device="cuda")
    elif lengths_mode == "half":
        lengths = torch.full((N,), T // 2, dtype=torch.int32, device="cuda")
    elif lengths_mode == "varied":
        lengths = torch.randint(
            1, T + 1, (N,), dtype=torch.int32, device="cuda", generator=g
        )
    elif lengths_mode == "small":
        # Some rows have lengths < K, forcing -1 padding.
        lengths = torch.randint(
            1, T // 8, (N,), dtype=torch.int32, device="cuda", generator=g
        )
    elif lengths_mode == "all_zero":
        lengths = torch.zeros(N, dtype=torch.int32, device="cuda")
    else:
        raise ValueError(lengths_mode)
    return logits, lengths


def _make_single_coarse_bin(
    N: int, T: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build unique, increasing FP32 values in one FP16 coarse bin.

    Consecutive FP32 bit patterns starting at 1.0 remain strictly ordered.
    For the lengths below they also share one ``decode_bin`` and one
    ``convert_to_uint8`` bin, deterministically overflowing the corresponding
    threshold-candidate buffer.
    """
    bits = torch.arange(
        0x3F800000,
        0x3F800000 + T,
        dtype=torch.int32,
        device="cuda",
    )
    logits = bits.view(torch.float32).unsqueeze(0).expand(N, -1).contiguous()
    lengths = torch.full((N,), T, dtype=torch.int32, device="cuda")
    return logits, lengths


def _make_negative_single_coarse_bin(
    N: int, T: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build unique, increasing negative FP32 values in one FP16 coarse bin."""
    offsets = torch.arange(T, dtype=torch.int64, device="cuda")
    bits = (0xBFD00000 - offsets).to(torch.int32)  # -1.625 toward zero
    logits = bits.view(torch.float32).unsqueeze(0).expand(N, -1).contiguous()
    lengths = torch.full((N,), T, dtype=torch.int32, device="cuda")
    return logits, lengths


def _make_negative_midpoint_overflow(
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Force count_gt >= K and count_eq > kMaxNumTie on a negative bin edge."""
    assert T >= 6000

    # High edge of the negative coarse bin containing -1.0. FP16
    # round-to-nearest-even maps this exact midpoint back to -1.0 and therefore
    # into the threshold histogram bin. The FP32 collect predicate classifies
    # equality with v_hi as strictly above.
    v_hi = torch.tensor(
        (-1.0 + -0.99951171875) * 0.5, dtype=torch.float32
    ).item()
    logits = torch.full((1, T), -2.0, dtype=torch.float32, device="cuda")
    logits[0, :3000] = v_hi

    # More than kMaxNumTie distinct in-bin values force the exact fallback.
    offsets = torch.arange(3000, dtype=torch.int64, device="cuda")
    bits = (0xBF814000 + offsets).to(torch.int32)
    logits[0, 3000:6000] = bits.view(torch.float32)
    lengths = torch.tensor([T], dtype=torch.int32, device="cuda")
    return logits, lengths


def test_decode_b1_k512_full():
    logits, lengths = _make(1, 2048, seed=0, lengths_mode="full")
    out = _run(logits, lengths, k=512, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=512, tag="decode B=1 K=512 full")


def test_decode_b1_k512_varied():
    """Decode with the row's valid prefix < K — exercises -1 padding."""
    logits, lengths = _make(1, 2048, seed=1, lengths_mode="small")
    out = _run(logits, lengths, k=512, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=512, tag="decode B=1 K=512 small-len")


def test_decode_b1_k1024_full():
    logits, lengths = _make(1, 2048, seed=2, lengths_mode="full")
    out = _run(logits, lengths, k=1024, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=1024, tag="decode B=1 K=1024 full")


def test_decode_b1_k2048_full():
    logits, lengths = _make(1, 2048, seed=3, lengths_mode="full")
    out = _run(logits, lengths, k=2048, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=2048, tag="decode B=1 K=2048 full")


def test_batched_decode_b4_varied():
    logits, lengths = _make(4, 2048, seed=4, lengths_mode="varied")
    out = _run(logits, lengths, k=512, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=512, tag="decode B=4 varied")


def test_batched_decode_b16_half():
    logits, lengths = _make(16, 2048, seed=5, lengths_mode="half")
    out = _run(logits, lengths, k=512, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=512, tag="decode B=16 half")


def test_padded_row_stride():
    """Rows can be a padded view, so the kernel must use stride(0)."""
    N, T, PAD, K = 3, 2048, 17, 512
    g = torch.Generator(device="cuda").manual_seed(56)
    base = torch.randn(N, T + PAD, device="cuda", generator=g)
    logits = base[:, :T]
    assert logits.stride(0) > logits.size(1)
    lengths = torch.full((N,), T, dtype=torch.int32, device="cuda")
    out = _run(logits, lengths, k=K, max_seq_len=T)
    _assert_equiv(out, logits, lengths, k=K, tag="padded row stride")


def test_mtp_batched_decode_flattened_bs_rows():
    """MTP passes score as [B, S, T]; the op contract is [B*S, T]."""
    B, S, T, K = 4, 3, 2048, 512
    g = torch.Generator(device="cuda").manual_seed(55)
    score = torch.randn(B, S, T, device="cuda", generator=g)
    lengths = torch.tensor(
        [
            [1, 2, 17],
            [511, 512, 513],
            [1024, 1536, 2048],
            [7, 129, 777],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    out_3d = torch.full((B, S, K), -1, dtype=torch.int32, device="cuda")
    ws = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    rtp_llm_ops.topk_glm5_indexer(
        score.view(B * S, T),
        lengths.view(B * S),
        out_3d.view(B * S, K),
        ws,
        K,
        T,
    )

    _assert_equiv(
        out_3d.view(B * S, K),
        score.view(B * S, T),
        lengths.view(B * S),
        k=K,
        tag="MTP B*S flattened",
    )


def test_batched_streaming_path_b64():
    """A larger row batch exercises the batched streaming dispatcher."""
    logits, lengths = _make(64, 2048, seed=6, lengths_mode="varied")
    out = _run(logits, lengths, k=512, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=512, tag="streaming B=64")


def test_long_seq_radix_path():
    """T > 32768 (RADIX_THRESHOLD) routes through the cooperative radix path."""
    logits, lengths = _make(2, 65536, seed=7, lengths_mode="full")
    out = _run(logits, lengths, k=2048, max_seq_len=65536)
    _assert_equiv(out, logits, lengths, k=2048, tag="radix L=64K")


def test_cluster_repeated_pivot_replay():
    """Stress cross-CTA collect/publication and the exact overflow fallback.

    B=8 and max_seq_len=262144 route to direct Cluster8. Each row repeats a
    different 64-value page, so its exact pivot occurs 4096 times. This forces
    DSMEM histogram reduction, cross-rank candidate aggregation, and a full-row
    exact boundary scan on every replay.
    """
    rows, stride, valid, k = 8, 262592, 262144, 2048
    generator = torch.Generator(device="cuda").manual_seed(20260731)
    page = torch.randn(
        (rows, 64), device="cuda", dtype=torch.float32, generator=generator
    )
    logits = page.repeat(1, (stride + 63) // 64)[:, :stride].contiguous()
    logits[:, valid:] = -torch.inf
    lengths = torch.full((rows,), valid, device="cuda", dtype=torch.int32)

    _assert_replay_value_equiv(
        logits,
        lengths,
        k,
        valid,
        replays=100,
        tag="Cluster8 repeated pivot",
    )


def test_cluster_ragged_long_short_replay():
    """Mix Cluster8 rows with the seq_len <= K trivial branch.

    Dispatch is based on the launch-wide max_seq_len, while each cluster reads
    its own row length. Direct dispatch assigns a separate hardware cluster to
    each row; this verifies the ragged early-return protocol but does not rely
    on persistent cross-row state.
    """
    rows, stride, valid, k = 8, 262592, 262144, 2048
    generator = torch.Generator(device="cuda").manual_seed(2026073104)
    page = torch.randn(
        (rows, 64), device="cuda", dtype=torch.float32, generator=generator
    )
    logits = page.repeat(1, (stride + 63) // 64)[:, :stride].contiguous()
    logits[:, valid:] = -torch.inf
    lengths = torch.tensor(
        [valid, 1024, valid, 1, valid, k, valid, 17],
        device="cuda",
        dtype=torch.int32,
    )

    _assert_replay_value_equiv(
        logits,
        lengths,
        k,
        valid,
        replays=100,
        tag="Cluster8 ragged long/short",
    )


def test_exact_boundary_negative_midpoint_output_bounds():
    """Exact fallback must not write past output when collect already filled K.

    Before the fix, the first K selected values looked correct, but the exact
    fallback underflowed ``K - count_gt`` and overwrote 3000 int32 values in
    the two guard rows backing this contiguous one-row output view.
    """
    T, K = 8193, 2048
    sentinel = -123456789
    logits, lengths = _make_negative_midpoint_overflow(T)
    storage = torch.full(
        (3, K), sentinel, dtype=torch.int32, device=logits.device
    )
    out = storage[:1]
    ws = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=logits.device)

    rtp_llm_ops.topk_glm5_indexer(logits, lengths, out, ws, K, T)
    torch.cuda.synchronize()

    indices = out[0].long()
    assert ((indices >= 0) & (indices < T)).all()
    assert torch.unique(indices).numel() == K
    actual = logits[0, indices].sort().values
    expected = logits[0].topk(K, sorted=False).values.sort().values
    assert torch.equal(actual, expected)
    assert (storage[1:] == sentinel).all(), (
        "exact_boundary_scan_topk wrote beyond output[:, topk]"
    )
    print("  [exact boundary negative midpoint output bounds] OK")


def test_cluster_exact_boundary_negative_midpoint():
    """Cluster8 exact fallback must not write beyond its shared output array."""
    T, K = 262144, 2048
    logits, lengths = _make_negative_midpoint_overflow(T)
    out = _run(logits, lengths, k=K, max_seq_len=T)
    torch.cuda.synchronize()
    _assert_value_equiv(
        out,
        logits,
        lengths,
        k=K,
        tag="Cluster8 exact boundary negative midpoint",
    )


def test_histogram_2048_candidate_overflow_exact():
    """DBUF=3708 overflow must rescan the full row instead of truncating."""
    N, T = 1, 4095
    logits, lengths = _make_single_coarse_bin(N, T)
    for K in (512, 1024, 2048):
        out = _run(logits, lengths, k=K, max_seq_len=T)
        _assert_equiv(out, logits, lengths, k=K, tag="histogram_2048 overflow")


def test_histogram_256_candidate_overflow_exact():
    """The medium path starts above 8192, already beyond its 4096 capacity."""
    N, T = 1, 8193
    logits, lengths = _make_single_coarse_bin(N, T)
    for K in (512, 1024, 2048):
        out = _run(logits, lengths, k=K, max_seq_len=T)
        _assert_equiv(out, logits, lengths, k=K, tag="histogram_256 overflow")


def test_histogram_256_overflow_with_fp32_pivot_ties_exact():
    """Overflow fallback must select the right value multiset across pivot ties.

    The old implementation refines only the first 4096 collected candidates
    and therefore misses the globally largest values at the end of this row.
    """
    N, T, K = 1, 8193, 512
    logits, lengths = _make_single_coarse_bin(N, T)

    # Keep 256 values strictly above the pivot and make 512 values equal to
    # it. Any 256 of the tied indices are valid members of the final top-512.
    pivot = logits[0, T - K].clone()
    logits[:, T - 768 : T - 256] = pivot

    out = _run(logits, lengths, k=K, max_seq_len=T)
    _assert_value_equiv(
        out,
        logits,
        lengths,
        k=K,
        tag="histogram_256 overflow with FP32 pivot ties",
    )


def test_histogram_256_negative_candidate_overflow_exact():
    """Overflow fallback must preserve ordered-FP32 semantics for negatives."""
    N, T, K = 1, 8193, 512
    logits, lengths = _make_negative_single_coarse_bin(N, T)
    out = _run(logits, lengths, k=K, max_seq_len=T)
    _assert_equiv(
        out,
        logits,
        lengths,
        k=K,
        tag="histogram_256 negative overflow",
    )


def test_batched_candidate_overflow_exact():
    """The batched dispatcher stays exact when the candidate buffer overflows."""
    N, T = 33, 32768
    logits, lengths = _make_single_coarse_bin(N, T)
    for K in (512, 1024, 2048):
        out = _run(logits, lengths, k=K, max_seq_len=T)
        _assert_equiv(out, logits, lengths, k=K, tag="batched candidate overflow")


def test_single_coarse_bin_large_radix_exact():
    """The existing large full-radix path stays exact on adversarial input."""
    N, T, K = 1, 65536, 512
    logits, lengths = _make_single_coarse_bin(N, T)
    out = _run(logits, lengths, k=K, max_seq_len=T)
    _assert_equiv(out, logits, lengths, k=K, tag="large radix single-bin")


def test_zero_length_row():
    """lengths[r] == 0 must yield an all-(-1) row."""
    logits, lengths = _make(2, 1024, seed=8, lengths_mode="full")
    lengths[1] = 0
    out = _run(logits, lengths, k=512, max_seq_len=1024)
    _assert_equiv(out, logits, lengths, k=512, tag="row-len 0")


def test_lengths_2d_accepted():
    """Op accepts lengths as 1D or 2D (decode passes [B, 1] in some flows)."""
    logits, lengths_1d = _make(4, 1024, seed=9, lengths_mode="varied")
    lengths_2d = lengths_1d.view(4, 1)
    out = _run(logits, lengths_2d.view(-1), k=512, max_seq_len=1024)
    _assert_equiv(out, logits, lengths_1d, k=512, tag="lengths 2D ok via view")


# ---------------------------------------------------------------------------
# Bench — compare against the framework reference implementation.
# ---------------------------------------------------------------------------
def _bench(fn, *args, warmup: int = 50, iters: int = 500) -> float:
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


def bench_decode_sweep():
    print("\n  decode sweep — kernel vs torch.topk")
    print(
        "    {:>4}  {:>5}  {:>5}  {:>10}  {:>10}  {:>10}".format(
            "B", "T", "K", "torch", "kernel", "speedup"
        )
    )
    fail = []
    cases = [
        (1, 512, 512, True),
        (1, 1024, 512, True),
        (1, 2048, 512, True),
        (1, 2048, 1024, True),
        (1, 2048, 2048, True),
        (4, 2048, 512, True),
        (16, 2048, 512, True),
        (64, 2048, 512, True),
    ]
    ws = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    for B, T, K, strict in cases:
        logits, lengths = _make(B, T, seed=100, lengths_mode="full")
        out = torch.full((B, K), -1, dtype=torch.int32, device="cuda")

        def run_kernel():
            rtp_llm_ops.topk_glm5_indexer(logits, lengths, out, ws, K, T)

        def run_torch():
            return logits.topk(K, dim=-1)[1]

        t_t = _bench(run_torch)
        t_k = _bench(run_kernel)
        marker = "" if t_k < t_t else (" (REGRESS!)" if strict else " (info)")
        print(
            f"    {B:4d}  {T:5d}  {K:5d}  {t_t*1e3:8.2f}us  {t_k*1e3:8.2f}us  "
            f"{t_t/t_k:8.2f}x{marker}"
        )
        if strict and not (t_k < t_t):
            fail.append((B, T, K))
    assert not fail, f"topk_glm5_indexer slower than torch.topk at: {fail}"


if __name__ == "__main__":
    if not _HAS_OP:
        print(
            "SKIP: rtp_llm_ops.topk_glm5_indexer not built — "
            "rebuild //rtp_llm:rtp_compute_ops"
        )
        raise SystemExit(0)
    print("== Correctness ==")
    test_decode_b1_k512_full()
    test_decode_b1_k512_varied()
    test_decode_b1_k1024_full()
    test_decode_b1_k2048_full()
    test_batched_decode_b4_varied()
    test_batched_decode_b16_half()
    test_padded_row_stride()
    test_mtp_batched_decode_flattened_bs_rows()
    test_batched_streaming_path_b64()
    test_long_seq_radix_path()
    test_cluster_repeated_pivot_replay()
    test_cluster_ragged_long_short_replay()
    test_exact_boundary_negative_midpoint_output_bounds()
    test_cluster_exact_boundary_negative_midpoint()
    test_histogram_2048_candidate_overflow_exact()
    test_histogram_256_candidate_overflow_exact()
    test_histogram_256_overflow_with_fp32_pivot_ties_exact()
    test_histogram_256_negative_candidate_overflow_exact()
    test_batched_candidate_overflow_exact()
    test_single_coarse_bin_large_radix_exact()
    test_zero_length_row()
    test_lengths_2d_accepted()
    print("\n== Benchmark ==")
    bench_decode_sweep()
    print("\nOK")
