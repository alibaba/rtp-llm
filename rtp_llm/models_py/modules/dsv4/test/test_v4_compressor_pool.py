"""UT for ``_compressor_triton.v4_compressor_pool``.

After the P0 cleanup in ``compressor.py`` this kernel is the **only** path
producing the pooled compressor output; the torch
``(kv * score.softmax(dim=2)).sum(dim=2)`` REF branch was removed.

Two layer types from ``dsv4_attention_shapes.md``:
  * CSA — overlap=True,  ratio=4    → G = 2*ratio = 8,   D = head_dim = 512
  * HCA — overlap=False, ratio=128  → G = ratio    = 128, D = head_dim = 512
Indexer's own compressor uses head_dim = 128 (CSA-only, G=8).

Shapes covered:
  * Decode CSA   — ``[B=1, NB=1,   G=8,   D=512]``
  * Decode HCA   — ``[B=1, NB=1,   G=128, D=512]``
  * Prefill CSA  — ``[B=1, NB=16,  G=8,   D=512]``
  * Prefill HCA  — ``[B=1, NB=4,   G=128, D=512]``
  * Indexer CSA  — ``[B=1, NB=1,   G=8,   D=128]``

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_v4_compressor_pool.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4._compressor_triton import v4_compressor_pool


# ---------------------------------------------------------------------------
# Reference — exactly mirrors the deleted else-branch in compressor.py
# ---------------------------------------------------------------------------
def ref_compressor_pool(kv: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """``(kv * score.softmax(dim=2)).sum(dim=2)`` over [B, NB, G, D]."""
    return (kv * score.softmax(dim=2)).sum(dim=2)


def _make_inputs(B: int, NB: int, G: int, D: int, *, seed: int = 0):
    torch.manual_seed(seed)
    kv = torch.randn(B, NB, G, D, dtype=torch.float32, device="cuda")
    score = torch.randn(B, NB, G, D, dtype=torch.float32, device="cuda")
    return kv.contiguous(), score.contiguous()


def _bench(fn, *args, warmup: int = 25, iters: int = 200) -> float:
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


def _assert_close(
    cand: torch.Tensor,
    ref: torch.Tensor,
    *,
    tag: str,
    rtol: float = 1e-3,
    atol: float = 1e-4,
):
    diff = (cand - ref).abs()
    rel = diff / (ref.abs() + 1e-6)
    print(
        f"  [{tag}] max_abs={diff.max():.4e} mean_abs={diff.mean():.4e} "
        f"max_rel={rel.max():.4e}"
    )
    assert torch.allclose(
        cand, ref, rtol=rtol, atol=atol
    ), f"{tag}: max_abs={diff.max():.4e} max_rel={rel.max():.4e} above tol"


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def test_csa_decode():
    kv, score = _make_inputs(B=1, NB=1, G=8, D=512, seed=0)
    cand = v4_compressor_pool(kv, score)
    ref = ref_compressor_pool(kv, score)
    _assert_close(cand, ref, tag="CSA decode")


def test_hca_decode():
    kv, score = _make_inputs(B=1, NB=1, G=128, D=512, seed=1)
    cand = v4_compressor_pool(kv, score)
    ref = ref_compressor_pool(kv, score)
    _assert_close(cand, ref, tag="HCA decode")


def test_csa_prefill_chunk():
    kv, score = _make_inputs(B=1, NB=16, G=8, D=512, seed=2)
    cand = v4_compressor_pool(kv, score)
    ref = ref_compressor_pool(kv, score)
    _assert_close(cand, ref, tag="CSA prefill")


def test_hca_prefill_chunk():
    kv, score = _make_inputs(B=1, NB=4, G=128, D=512, seed=3)
    cand = v4_compressor_pool(kv, score)
    ref = ref_compressor_pool(kv, score)
    _assert_close(cand, ref, tag="HCA prefill")


def test_indexer_csa_d128():
    """Indexer's nested compressor: head_dim = 128, ratio = 4 → G=8."""
    kv, score = _make_inputs(B=1, NB=1, G=8, D=128, seed=4)
    cand = v4_compressor_pool(kv, score)
    ref = ref_compressor_pool(kv, score)
    _assert_close(cand, ref, tag="indexer CSA D=128")


def test_batched_b4():
    kv, score = _make_inputs(B=4, NB=8, G=8, D=512, seed=5)
    cand = v4_compressor_pool(kv, score)
    ref = ref_compressor_pool(kv, score)
    _assert_close(cand, ref, tag="B=4")


def test_minus_inf_in_score():
    """Score may contain -inf for padded G slots (overlap_transform fills
    invalid positions with -inf)."""
    torch.manual_seed(6)
    kv = torch.randn(1, 2, 8, 512, dtype=torch.float32, device="cuda")
    score = torch.randn(1, 2, 8, 512, dtype=torch.float32, device="cuda")
    score[:, 0, :4, :] = float("-inf")
    cand = v4_compressor_pool(kv, score)
    ref = ref_compressor_pool(kv, score)
    _assert_close(cand, ref, tag="-inf padding")


def test_empty_NB():
    kv = torch.empty(1, 0, 8, 512, dtype=torch.float32, device="cuda")
    score = torch.empty(1, 0, 8, 512, dtype=torch.float32, device="cuda")
    out = v4_compressor_pool(kv, score)
    assert out.shape == (1, 0, 512)


# ---------------------------------------------------------------------------
# P2 — overlap-fold path (CSA decode raw-state input)
# ---------------------------------------------------------------------------
def ref_overlap_then_pool(
    kv_raw: torch.Tensor, sc_raw: torch.Tensor, ratio: int, d: int
) -> torch.Tensor:
    """REF for overlap=True: do the torch.cat first, then standard pool.
    kv_raw / sc_raw: [B, NB, 2*ratio, 2*d]; output [B, NB, d]."""
    B, NB, twoR, twoD = kv_raw.shape
    assert twoR == 2 * ratio and twoD == 2 * d
    kv_view = torch.cat(
        [kv_raw[:, :, :ratio, :d], kv_raw[:, :, ratio:, d:]], dim=2
    )  # [B, NB, 2r, d]
    sc_view = torch.cat([sc_raw[:, :, :ratio, :d], sc_raw[:, :, ratio:, d:]], dim=2)
    return ref_compressor_pool(kv_view.contiguous(), sc_view.contiguous())


def _make_raw_state(B: int, NB: int, ratio: int, d: int, *, seed: int):
    """[B, NB, 2*ratio, 2*d] raw CSA state — like compressor.kv_state /
    score_state in memory."""
    torch.manual_seed(seed)
    kv = torch.randn(B, NB, 2 * ratio, 2 * d, dtype=torch.float32, device="cuda")
    sc = torch.randn(B, NB, 2 * ratio, 2 * d, dtype=torch.float32, device="cuda")
    return kv.contiguous(), sc.contiguous()


def test_overlap_csa_decode_d512():
    """CSA decode (ratio=4, d=512) — production shape."""
    ratio, d = 4, 512
    kv_raw, sc_raw = _make_raw_state(B=1, NB=1, ratio=ratio, d=d, seed=200)
    cand = v4_compressor_pool(kv_raw, sc_raw, overlap=True, out_d=d)
    ref = ref_overlap_then_pool(kv_raw, sc_raw, ratio, d)
    _assert_close(cand, ref, tag="overlap CSA d=512")


def test_overlap_indexer_csa_d128():
    """Indexer's nested compressor: ratio=4, d=128."""
    ratio, d = 4, 128
    kv_raw, sc_raw = _make_raw_state(B=1, NB=1, ratio=ratio, d=d, seed=201)
    cand = v4_compressor_pool(kv_raw, sc_raw, overlap=True, out_d=d)
    ref = ref_overlap_then_pool(kv_raw, sc_raw, ratio, d)
    _assert_close(cand, ref, tag="overlap indexer d=128")


def test_overlap_batched_b4():
    ratio, d = 4, 512
    kv_raw, sc_raw = _make_raw_state(B=4, NB=1, ratio=ratio, d=d, seed=202)
    cand = v4_compressor_pool(kv_raw, sc_raw, overlap=True, out_d=d)
    ref = ref_overlap_then_pool(kv_raw, sc_raw, ratio, d)
    _assert_close(cand, ref, tag="overlap B=4")


def test_overlap_with_minus_inf_padding():
    """``score_state`` may legitimately contain -inf in the upper-half slots
    that haven't been filled yet (overlap_transform fill value).  The kernel's
    softmax denom must stay finite for the lower-half slots."""
    ratio, d = 4, 512
    kv_raw, sc_raw = _make_raw_state(B=1, NB=1, ratio=ratio, d=d, seed=203)
    # Mark the "second-window upper half" as -inf — kernel reads it as
    # the upper half (g >= ratio reads d:2d) so this lands in the post-cat view.
    sc_raw[:, :, ratio:, d:] = float("-inf")
    cand = v4_compressor_pool(kv_raw, sc_raw, overlap=True, out_d=d)
    ref = ref_overlap_then_pool(kv_raw, sc_raw, ratio, d)
    _assert_close(cand, ref, tag="overlap -inf upper")


def test_overlap_strides_non_default_NB():
    """NB > 1 raw state — exercises NB stride correctness when overlap=True."""
    ratio, d = 4, 512
    kv_raw, sc_raw = _make_raw_state(B=1, NB=3, ratio=ratio, d=d, seed=204)
    cand = v4_compressor_pool(kv_raw, sc_raw, overlap=True, out_d=d)
    ref = ref_overlap_then_pool(kv_raw, sc_raw, ratio, d)
    _assert_close(cand, ref, tag="overlap NB=3")


def bench_overlap_decode():
    """Compare overlap-fold kernel against the prior torch.cat + kernel path."""
    cases = [
        ("overlap CSA d=512", 1, 4, 512),
        ("overlap indexer d=128", 1, 4, 128),
    ]
    print("\n  P2 overlap decode — kernel-only vs torch.cat+kernel")
    print(
        "    {:<28}  {:>10}  {:>10}  {:>10}".format(
            "case", "cat+pool", "fused", "speedup"
        )
    )
    fail = []
    for name, B, ratio, d in cases:
        kv_raw, sc_raw = _make_raw_state(B=B, NB=1, ratio=ratio, d=d, seed=300)

        def run_old():
            kv_view = torch.cat(
                [kv_raw[:, :, :ratio, :d], kv_raw[:, :, ratio:, d:]], dim=2
            ).contiguous()
            sc_view = torch.cat(
                [sc_raw[:, :, :ratio, :d], sc_raw[:, :, ratio:, d:]], dim=2
            ).contiguous()
            return v4_compressor_pool(kv_view, sc_view)

        def run_new():
            return v4_compressor_pool(kv_raw, sc_raw, overlap=True, out_d=d)

        t_o = _bench(run_old)
        t_n = _bench(run_new)
        marker = "" if t_n < t_o else " (REGRESS)"
        print(
            f"    {name:<28}  {t_o*1e3:8.2f}us  {t_n*1e3:8.2f}us  "
            f"{t_o/t_n:8.2f}x{marker}"
        )
        if not (t_n < t_o):
            fail.append(name)
    assert not fail, f"overlap-fold not faster than cat+pool at: {fail}"


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
def bench_decode():
    # ``strict=True`` cases must beat eager — these are the perf-critical
    # production hot paths (HCA layer count dominates V4-Flash).  Small
    # CSA (G=8) shapes are launch-bound and printed for visibility only.
    cases = [
        ("CSA decode (G=8 D=512)", 1, 1, 8, 512, False),
        ("HCA decode (G=128 D=512)", 1, 1, 128, 512, True),
        ("indexer CSA (G=8 D=128)", 1, 1, 8, 128, False),
    ]
    print("\n  decode shapes")
    print(
        "    {:<28}  {:>10}  {:>10}  {:>10}".format("case", "eager", "fused", "speedup")
    )
    fail = []
    for name, B, NB, G, D, strict in cases:
        kv, score = _make_inputs(B=B, NB=NB, G=G, D=D, seed=100)
        t_e = _bench(ref_compressor_pool, kv, score)
        t_c = _bench(v4_compressor_pool, kv, score)
        marker = (
            ""
            if t_c < t_e
            else (" (REGRESS!)" if strict else " (launch-bound, info only)")
        )
        print(
            f"    {name:<28}  {t_e*1e3:8.2f}us  {t_c*1e3:8.2f}us  {t_e/t_c:8.2f}x{marker}"
        )
        if strict and not (t_c < t_e):
            fail.append(name)
    assert not fail, f"v4_compressor_pool slower than eager at: {fail}"


def bench_prefill():
    # HCA strictly required to beat eager (G=128 makes eager softmax expensive).
    # CSA (G=8) is launch-bound on both sides; kernel is steady ~16us, eager
    # fluctuates 15-26us depending on cuDNN/launch cache state.  Print only.
    cases = [
        ("CSA prefill NB=16", 1, 16, 8, 512, False),
        ("CSA prefill NB=64", 1, 64, 8, 512, False),
        ("HCA prefill NB=4", 1, 4, 128, 512, True),
        ("HCA prefill NB=16", 1, 16, 128, 512, True),
    ]
    print("\n  prefill shapes")
    print(
        "    {:<28}  {:>10}  {:>10}  {:>10}".format("case", "eager", "fused", "speedup")
    )
    fail = []
    for name, B, NB, G, D, strict in cases:
        kv, score = _make_inputs(B=B, NB=NB, G=G, D=D, seed=101)
        t_e = _bench(ref_compressor_pool, kv, score)
        t_c = _bench(v4_compressor_pool, kv, score)
        marker = (
            ""
            if t_c < t_e
            else (" (REGRESS!)" if strict else " (launch-bound, info only)")
        )
        print(
            f"    {name:<28}  {t_e*1e3:8.2f}us  {t_c*1e3:8.2f}us  {t_e/t_c:8.2f}x{marker}"
        )
        if strict and not (t_c < t_e):
            fail.append(name)
    assert not fail, f"v4_compressor_pool slower than eager at: {fail}"


if __name__ == "__main__":
    print("== Correctness (overlap=False) ==")
    test_csa_decode()
    test_hca_decode()
    test_csa_prefill_chunk()
    test_hca_prefill_chunk()
    test_indexer_csa_d128()
    test_batched_b4()
    test_minus_inf_in_score()
    test_empty_NB()
    print("\n== Correctness (overlap=True, P2) ==")
    test_overlap_csa_decode_d512()
    test_overlap_indexer_csa_d128()
    test_overlap_batched_b4()
    test_overlap_with_minus_inf_padding()
    test_overlap_strides_non_default_NB()
    print("\n== Benchmark ==")
    bench_decode()
    bench_prefill()
    bench_overlap_decode()
    print("\nOK")
