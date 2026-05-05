"""UT for ``_indexer_score_triton.v4_indexer_score``.

After the P0 cleanup in ``indexer.py`` this kernel is the **only** path
that produces the indexer score (the torch ``einsum + relu + wsum``
reference branch was removed).  This UT locks down its math against an
inline reference and asserts a wall-clock win on the canonical V4-Flash
shapes.

Shapes covered (from ``dsv4_attention_shapes.md``):
  * Decode    — ``[B=1, S=1, H=64, D=128]`` × ``T=2048`` (T_max for 8K ctx)
  * Prefill fresh    — ``[B=1, S=64, H=64, D=128]`` × ``T=16`` (causal)
  * Prefill continue — ``[B=1, S=8,  H=64, D=128]`` × ``T=512`` (no mask)

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_v4_indexer_score.py
"""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score


# ---------------------------------------------------------------------------
# Reference — exactly mirrors the deleted else-branch in indexer.py
# ---------------------------------------------------------------------------
def ref_indexer_score(
    q: torch.Tensor,  # [B, S, H, D] bf16
    kv: torch.Tensor,  # [B, T, D]    bf16
    weights: torch.Tensor,  # [B, S, H]    bf16/fp32
    q_pos: Optional[torch.Tensor],  # [B, S] int32 — when not None apply causal mask
    compress_ratio: int,
) -> torch.Tensor:
    q_f = q.float()
    kv_f = kv.float()
    w_f = weights.float()
    # einsum: [B,S,H,T]
    score = torch.einsum("bshd,btd->bsht", q_f, kv_f)
    score = (score.relu_() * w_f.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
    if q_pos is not None:
        B, S, T = score.shape
        kv_cols = torch.arange(T, device=score.device).view(1, 1, T)
        # Kernel masks t >= (q_pos + 1) // compress_ratio  (causal-on-compressed)
        thresh = ((q_pos.to(torch.int64) + 1) // compress_ratio).view(B, S, 1)
        score = torch.where(
            kv_cols < thresh, score, torch.full_like(score, float("-inf"))
        )
    return score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_inputs(B: int, S: int, T: int, H: int = 64, D: int = 128, *, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    kv = torch.randn(B, T, D, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    return q.contiguous(), kv.contiguous(), w.contiguous()


def _bench(fn, *args, warmup: int = 25, iters: int = 100) -> float:
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
    cand: torch.Tensor, ref: torch.Tensor, *, tag: str, rtol: float, atol: float
):
    # Compare in fp32 (kernel returns fp32, ref also fp32). Allow generous tol
    # on RELU+sum across H=64 in bf16 — ULP noise accumulates linearly.
    finite = torch.isfinite(ref)
    diff = (cand[finite] - ref[finite]).abs()
    rel = diff / (ref[finite].abs() + 1e-6)
    print(
        f"  [{tag}] max_abs={diff.max():.4e} mean_abs={diff.mean():.4e} "
        f"max_rel={rel.max():.4e}"
    )
    assert torch.allclose(
        cand[finite], ref[finite], rtol=rtol, atol=atol
    ), f"{tag}: max_abs={diff.max():.4e} max_rel={rel.max():.4e} above tol"
    # Mask positions must agree exactly.
    assert torch.equal(
        torch.isinf(cand) & (cand < 0), torch.isinf(ref) & (ref < 0)
    ), f"{tag}: -inf mask positions differ"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_decode_no_mask():
    """Decode-style: [B=1, S=1, T=2048], no causal mask (q_pos=None)."""
    q, kv, w = _make_inputs(B=1, S=1, T=2048, seed=0)
    cand = v4_indexer_score(q, kv, w, q_pos=None, compress_ratio=4)
    ref = ref_indexer_score(q, kv, w, q_pos=None, compress_ratio=4)
    _assert_close(cand, ref, tag="decode", rtol=1e-2, atol=5e-2)


def test_decode_with_q_pos_no_mask_path():
    """Decode also exercises q_pos=tensor on a single token; with the kernel's
    mask convention (t < (q_pos+1)//ratio) the typical decode q_pos covers
    all live T, so no -inf appears.  Match anyway."""
    q, kv, w = _make_inputs(B=2, S=1, T=512, seed=11)
    # q_pos large enough that (q_pos+1)//4 == T → all T columns valid
    q_pos = torch.full((2, 1), 4 * 512 - 1, dtype=torch.int32, device="cuda")
    cand = v4_indexer_score(q, kv, w, q_pos=q_pos, compress_ratio=4)
    ref = ref_indexer_score(q, kv, w, q_pos=q_pos, compress_ratio=4)
    _assert_close(cand, ref, tag="decode_qpos", rtol=1e-2, atol=5e-2)


def test_prefill_fresh_causal_mask():
    """Fresh prefill: q_pos = arange(S); kernel applies causal mask on compressed."""
    B, S, T = 1, 64, 16  # S tokens, T = S/ratio for ratio=4
    q, kv, w = _make_inputs(B=B, S=S, T=T, seed=2)
    q_pos = (
        torch.arange(S, dtype=torch.int32, device="cuda")
        .view(1, S)
        .expand(B, S)
        .contiguous()
    )
    cand = v4_indexer_score(q, kv, w, q_pos=q_pos, compress_ratio=4)
    ref = ref_indexer_score(q, kv, w, q_pos=q_pos, compress_ratio=4)
    _assert_close(cand, ref, tag="prefill_fresh", rtol=1e-2, atol=5e-2)


def test_prefill_continuation_no_mask():
    """Continuation prefill (sp > 0): pass q_pos=None, no causal mask in kernel."""
    B, S, T = 1, 8, 512
    q, kv, w = _make_inputs(B=B, S=S, T=T, seed=3)
    cand = v4_indexer_score(q, kv, w, q_pos=None, compress_ratio=4)
    ref = ref_indexer_score(q, kv, w, q_pos=None, compress_ratio=4)
    _assert_close(cand, ref, tag="prefill_cont", rtol=1e-2, atol=5e-2)


def test_batched_decode():
    """Batched decode B=4 — confirms per-row q_pos is applied independently."""
    B, S, T = 4, 1, 1024
    q, kv, w = _make_inputs(B=B, S=S, T=T, seed=4)
    # Different live-T per row.
    q_pos = torch.tensor([100, 500, 1023, 50], dtype=torch.int32, device="cuda").view(
        B, 1
    )
    # Convert to "global pos" — kernel divides by ratio.
    q_pos = (q_pos * 4 + 3).to(torch.int32)
    cand = v4_indexer_score(q, kv, w, q_pos=q_pos, compress_ratio=4)
    ref = ref_indexer_score(q, kv, w, q_pos=q_pos, compress_ratio=4)
    _assert_close(cand, ref, tag="batched", rtol=1e-2, atol=5e-2)


def test_empty_S_or_T():
    """Boundary: S=0 or T=0 must return an empty tensor without launching."""
    q = torch.empty(1, 0, 64, 128, dtype=torch.bfloat16, device="cuda")
    kv = torch.empty(1, 0, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(1, 0, 64, dtype=torch.bfloat16, device="cuda")
    out = v4_indexer_score(q, kv, w, q_pos=None, compress_ratio=4)
    assert out.shape == (1, 0, 0)


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
def bench_decode_sweep():
    # Production decode: T_max ≤ 2048 for 8K ctx (V4-Flash default).  T=4096
    # exceeds the kernel's tuning point and is reported info-only.
    print("\n  decode: B=1, S=1, H=64, D=128 — T sweep")
    print("    {:>6}  {:>10}  {:>10}  {:>10}".format("T", "eager", "fused", "speedup"))
    fail = []
    for T, strict in (
        (256, True),
        (512, True),
        (1024, True),
        (2048, True),
        (4096, False),
    ):
        q, kv, w = _make_inputs(B=1, S=1, T=T, seed=10)
        t_e = _bench(ref_indexer_score, q, kv, w, None, 4)
        t_c = _bench(v4_indexer_score, q, kv, w, None, 4)
        marker = (
            "" if t_c < t_e else (" (REGRESS!)" if strict else " (out-of-range, info)")
        )
        print(
            f"    {T:6d}  {t_e*1e3:8.2f}us  {t_c*1e3:8.2f}us  {t_e/t_c:8.2f}x{marker}"
        )
        if strict and not (t_c < t_e):
            fail.append(T)
    assert not fail, f"v4_indexer_score slower than eager at T={fail}"


def bench_prefill_sweep():
    # Kernel tuned for large S with BF16 tensor-core mma (V4-Flash 64K + CP=4
    # → S=T=16384).  Win grows with S; tiny S is launch-bound.
    print("\n  prefill fresh: B=1, S=T*4, H=64, D=128 — S sweep")
    print("    {:>6}  {:>10}  {:>10}  {:>10}".format("S", "eager", "fused", "speedup"))
    fail = []
    for S, strict in ((64, False), (256, True), (1024, True), (4096, True)):
        T = S // 4
        q, kv, w = _make_inputs(B=1, S=S, T=T, seed=20)
        q_pos = (
            torch.arange(S, dtype=torch.int32, device="cuda")
            .view(1, S)
            .expand(1, S)
            .contiguous()
        )
        t_e = _bench(ref_indexer_score, q, kv, w, q_pos, 4)
        t_c = _bench(v4_indexer_score, q, kv, w, q_pos, 4)
        marker = (
            "" if t_c < t_e else (" (REGRESS!)" if strict else " (launch-bound, info)")
        )
        print(
            f"    {S:6d}  {t_e*1e3:8.2f}us  {t_c*1e3:8.2f}us  {t_e/t_c:8.2f}x{marker}"
        )
        if strict and not (t_c < t_e):
            fail.append(S)
    assert not fail, f"v4_indexer_score slower than eager at S={fail}"


if __name__ == "__main__":
    print("== Correctness ==")
    test_decode_no_mask()
    test_decode_with_q_pos_no_mask_path()
    test_prefill_fresh_causal_mask()
    test_prefill_continuation_no_mask()
    test_batched_decode()
    test_empty_S_or_T()
    print("\n== Benchmark ==")
    bench_decode_sweep()
    bench_prefill_sweep()
    print("\nOK")
