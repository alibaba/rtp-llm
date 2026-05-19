"""UT for ``rtp_llm_ops.dsv4_persistent_topk``.

Vendored from vLLM (csrc/persistent_topk.cuh + csrc/topk.cu @ b55d830).
Replaces ``torch.topk`` on the DSv4 indexer decode hot path.

Op contract (mirrors vLLM):
  logits   : [N, T]  float32 row-contiguous; stride(0) may exceed T
  lengths  : [N]     int32     — per-row valid count; positions past
                                  ``lengths[r]`` are written as -1 in output
  output   : [N, K]  int32     — written; ordering of valid indices is NOT
                                  guaranteed (compare as sets, not lists)
  workspace: uint8 ≥ 1 MB, CUDA tensor (any shape, ``.size(0) >= 1MB``)
  K        : 512, 1024, or 2048 (compile-time dispatch in the kernel)
  max_seq_len: max possible T across rows; controls cooperative launch path

Equivalence semantics:
  topk_set(logits[r, : lengths[r]]) == set(output[r, :min(K, lengths[r])])
  output[r, k] == -1 for k >= min(K, lengths[r])
  Order across the valid prefix is unspecified.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_dsv4_persistent_topk.py
"""

from __future__ import annotations

from typing import Tuple

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops

# When the .so doesn't have the binding yet (pre-rebuild), exit cleanly so
# CI / local runs report SKIP rather than ImportError.
_HAS_OP = hasattr(rtp_llm_ops, "dsv4_persistent_topk")

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
    rtp_llm_ops.dsv4_persistent_topk(logits, lengths, out, ws, k, max_seq_len)
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
    rtp_llm_ops.dsv4_persistent_topk(
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


def test_filtered_path_b64():
    """num_rows > 32 with ≥128KB smem/block triggers FilteredTopK path."""
    logits, lengths = _make(64, 2048, seed=6, lengths_mode="varied")
    out = _run(logits, lengths, k=512, max_seq_len=2048)
    _assert_equiv(out, logits, lengths, k=512, tag="filtered B=64")


def test_long_seq_radix_path():
    """T > 32768 (RADIX_THRESHOLD) routes through the cooperative radix path."""
    logits, lengths = _make(2, 65536, seed=7, lengths_mode="full")
    out = _run(logits, lengths, k=2048, max_seq_len=65536)
    _assert_equiv(out, logits, lengths, k=2048, tag="radix L=64K")


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
# Bench — compare against torch.topk (current production path).
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
            rtp_llm_ops.dsv4_persistent_topk(logits, lengths, out, ws, K, T)

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
    assert not fail, f"dsv4_persistent_topk slower than torch.topk at: {fail}"


if __name__ == "__main__":
    if not _HAS_OP:
        print(
            "SKIP: rtp_llm_ops.dsv4_persistent_topk not built — "
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
    test_filtered_path_b64()
    test_long_seq_radix_path()
    test_zero_length_row()
    test_lengths_2d_accepted()
    print("\n== Benchmark ==")
    bench_decode_sweep()
    print("\nOK")
