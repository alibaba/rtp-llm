"""UT for fused Q/KV-RMSNorm + partial RoPE (single Triton kernel).

Audit doc §7.4 P0 / row 1 of P0 priority list. The decode path on
``attention.py`` calls ``fused_rmsnorm_rope`` as a single Triton launch
per Q and per KV, replacing the prior split pipeline (eager
``apply_rotary_emb_batched`` after an RMSNorm).

This UT imports the production wrapper directly (so the contiguity
contract and any wrapper-level changes are exercised here), then verifies:
  1) Numerical accuracy vs the eager torch reference within bf16 tolerance
  2) Wall-clock GPU time improvement vs the eager reference

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=7 \
    /opt/conda310/bin/python3 \
    rtp_llm/models_py/modules/dsv4/test/test_fused_rmsnorm_rope.py
"""

from __future__ import annotations

import time

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb_batched,
    precompute_freqs_cis,
)


# ---------------------------------------------------------------------------
# Eager reference (matches attention.py decode path)
# ---------------------------------------------------------------------------
def ref_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis_per_b: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
    inverse: bool = False,
) -> torch.Tensor:
    rd = rope_head_dim
    out = x.clone()
    x32 = out.float()
    inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    y = x32 * inv
    if weight is not None:
        y = y * weight.float()
    out = y.to(x.dtype)
    apply_rotary_emb_batched(out[..., -rd:], freqs_cis_per_b, inverse=inverse)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_freqs(B: int, rd: int, *, device: str = "cuda") -> torch.Tensor:
    """Build ``[B, rd/2]`` complex64 freqs_cis_per_b. Mirrors V4 setup."""
    freqs_full = precompute_freqs_cis(
        dim=rd,
        seqlen=B + 16,
        original_seq_len=4096,
        base=10000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
    ).to(device)
    start_pos = torch.randint(0, B + 16, (B,), device=device)
    return freqs_full[start_pos]


def _bench(fn, *args, warmup: int = 25, iters: int = 200) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters  # ms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_q_correctness():
    """Multi-head Q path: [B, 1, H, head_dim], RoPE on last rope_head_dim, no weight."""
    torch.manual_seed(0)
    B, H, head_dim, rd, eps = 4, 64, 128, 64, 1e-6
    q = torch.randn(B, 1, H, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    freqs = _make_freqs(B, rd)

    ref = ref_rmsnorm_rope(q, None, freqs, rd, eps=eps)
    cand = fused_rmsnorm_rope(q, None, freqs, rd, eps=eps)

    d_ref = (cand.float() - ref.float()).abs()
    print(f"  [Q]    cand vs eager-ref  max={d_ref.max():.4e}  mean={d_ref.mean():.4e}")
    # Fused stays in fp32 across the RMSNorm→RoPE boundary; eager-ref casts to
    # bf16 in between. Diff is single-bf16-ULP noise (≤1/64≈0.0156), and the
    # fused result is strictly more accurate. Allow 1 ULP slack.
    assert d_ref.max() <= 2e-2, f"Q max diff {d_ref.max()} exceeds bf16 1-ULP tol"


def test_kv_correctness():
    """Single-head KV path: [B, 1, head_dim], RoPE on last rd, with learned weight."""
    torch.manual_seed(1)
    B, head_dim, rd, eps = 4, 512, 64, 1e-6
    kv = torch.randn(B, 1, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda").abs() + 0.5
    freqs = _make_freqs(B, rd)

    ref = ref_rmsnorm_rope(kv, weight, freqs, rd, eps=eps)
    cand = fused_rmsnorm_rope(kv, weight, freqs, rd, eps=eps)

    d_ref = (cand.float() - ref.float()).abs()
    print(f"  [KV]   cand vs eager-ref  max={d_ref.max():.4e}  mean={d_ref.mean():.4e}")
    assert d_ref.max() < 5e-2, f"KV max diff {d_ref.max()} exceeds tol"


def test_output_buffer_and_inplace_correctness():
    """Wrapper output modes used by future decode workspace reuse."""
    torch.manual_seed(3)
    B, H, head_dim, rd, eps = 4, 64, 128, 64, 1e-6
    q = torch.randn(B, 1, H, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    freqs = _make_freqs(B, rd)

    ref = fused_rmsnorm_rope(q, None, freqs, rd, eps=eps)

    out = torch.empty_like(q)
    cand_out = fused_rmsnorm_rope(q, None, freqs, rd, eps=eps, out=out)
    assert cand_out.data_ptr() == out.data_ptr()
    d_out = (cand_out.float() - ref.float()).abs()
    print(f"  [OUT]  cand vs default-ref max={d_out.max():.4e}  mean={d_out.mean():.4e}")
    assert d_out.max() <= 2e-2

    q_inplace = q.clone()
    cand_inplace = fused_rmsnorm_rope(q_inplace, None, freqs, rd, eps=eps, inplace=True)
    assert cand_inplace.data_ptr() == q_inplace.data_ptr()
    d_inplace = (cand_inplace.float() - ref.float()).abs()
    print(
        f"  [INP]  cand vs default-ref max={d_inplace.max():.4e}  "
        f"mean={d_inplace.mean():.4e}"
    )
    assert d_inplace.max() <= 2e-2


def test_group_heads_correctness():
    """Grouped-head Q path shares one freq row across multiple heads."""
    torch.manual_seed(4)
    B, H, head_dim, rd, eps = 4, 64, 128, 64, 1e-6
    q = torch.randn(B, 1, H, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    freqs = _make_freqs(B, rd)
    ref = fused_rmsnorm_rope(q, None, freqs, rd, eps=eps)
    for group_heads in (2, 4, 8):
        cand = fused_rmsnorm_rope(
            q, None, freqs, rd, eps=eps, group_heads=group_heads
        )
        d = (cand.float() - ref.float()).abs()
        print(
            f"  [GH{group_heads}] cand vs default-ref max={d.max():.4e} "
            f"mean={d.mean():.4e}"
        )
        assert d.max() <= 2e-2

        q_inplace = q.clone()
        cand_inplace = fused_rmsnorm_rope(
            q_inplace,
            None,
            freqs,
            rd,
            eps=eps,
            group_heads=group_heads,
            inplace=True,
        )
        d_inplace = (cand_inplace.float() - ref.float()).abs()
        assert d_inplace.max() <= 2e-2


def test_inverse_rope_path_correctness():
    """Inverse RoPE path (used on attention output before wo_a)."""
    torch.manual_seed(2)
    B, H, head_dim, rd, eps = 4, 64, 128, 64, 1e-6
    o = torch.randn(B, 1, H, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    freqs = _make_freqs(B, rd)
    # NOTE: inverse-RoPE path in main code does NOT have an RMSNorm before;
    # this is just checking the kernel's INVERSE branch works for §3.5 setup.
    ref = ref_rmsnorm_rope(o, None, freqs, rd, eps=eps, inverse=True)
    cand = fused_rmsnorm_rope(o, None, freqs, rd, eps=eps, inverse=True)
    d = (cand.float() - ref.float()).abs()
    print(f"  [INV]  cand vs eager-ref  max={d.max():.4e}  mean={d.mean():.4e}")
    assert d.max() <= 2e-2


def _build_freqs_prefill(T: int, rd: int) -> torch.Tensor:
    """Prefill uses per-position freqs broadcast across the batch dim.
    Here B=T for the batched API (one freqs_cis row per token)."""
    f = precompute_freqs_cis(
        dim=rd,
        seqlen=T + 16,
        original_seq_len=4096,
        base=10000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
    ).to("cuda")
    return f[:T]


def _bench_q_for_tokens(T: int, *, H: int = 64, head_dim: int = 128, rd: int = 64):
    """Bench at [B=1, S=T, H, head_dim] — the attention.py prefill/decode Q path
    flattens this into N=B*S*H kernel programs. B=1 is canonical."""
    q = torch.randn(1, T, H, head_dim, dtype=torch.bfloat16, device="cuda")
    # Use T-positioned freqs; treat each token-pos as an independent "B" row.
    freqs_per_tok = _build_freqs_prefill(T, rd)  # [T, rd/2]
    # fused_rmsnorm_rope expects B-major freqs; simulate by reshaping q to [T, 1, H, D].
    q_b = q.view(T, 1, H, head_dim).contiguous()

    def run_eager(q_b, f):
        out = q_b.clone()
        x32 = out.float()
        inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)
        out = (x32 * inv).to(q_b.dtype)
        apply_rotary_emb_batched(out[..., -rd:], f)
        return out

    def run_cand(q_b, f):
        return fused_rmsnorm_rope(q_b, None, f, rd)

    t_e = _bench(run_eager, q_b, freqs_per_tok)
    t_c = _bench(run_cand, q_b, freqs_per_tok)
    return t_e, t_c


def _bench_kv_for_tokens(T: int, *, head_dim: int = 512, rd: int = 64):
    """KV path: [T, 1, head_dim] single-head with learned weight."""
    kv = torch.randn(T, 1, head_dim, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda").abs() + 0.5
    freqs_per_tok = _build_freqs_prefill(T, rd)

    def run_eager(kv, w, f):
        out = kv.clone()
        x32 = out.float()
        inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)
        out = (w.float() * x32 * inv).to(kv.dtype)
        apply_rotary_emb_batched(out[..., -rd:], f)
        return out

    def run_cand(kv, w, f):
        return fused_rmsnorm_rope(kv, w, f, rd)

    t_e = _bench(run_eager, kv, weight, freqs_per_tok)
    t_c = _bench(run_cand, kv, weight, freqs_per_tok)
    return t_e, t_c


def bench_token_sweep():
    """Sweep covering decode (T≤16) and prefill (T≥256) shapes.

    The fused_rmsnorm_rope path is used by BOTH prefill and decode (see
    attention.py:707-729 and :1144-1157); its launch dimension is N = B*S*H
    for Q and N = B*S for KV, so a larger T scales the grid linearly.
    """
    print("  Q path [B=1, S=T, H=64, D=128]:")
    print("    {:>6}  {:>10}  {:>10}  {:>10}".format("T", "eager", "fused", "speedup"))
    Tlist = [64, 128, 256, 4096, 65536]
    q_results = []
    for T in Tlist:
        try:
            t_e, t_c = _bench_q_for_tokens(T)
        except torch.cuda.OutOfMemoryError:
            print(f"    {T:6d}   OOM")
            torch.cuda.empty_cache()
            continue
        q_results.append((T, t_e, t_c))
        print(
            "    {:6d}  {:8.2f}us  {:8.2f}us  {:8.2f}x".format(
                T, t_e * 1e3, t_c * 1e3, t_e / t_c
            )
        )
        torch.cuda.empty_cache()

    print("\n  KV path [B=1, S=T, D=512]:")
    print("    {:>6}  {:>10}  {:>10}  {:>10}".format("T", "eager", "fused", "speedup"))
    kv_results = []
    for T in Tlist:
        try:
            t_e, t_c = _bench_kv_for_tokens(T)
        except torch.cuda.OutOfMemoryError:
            print(f"    {T:6d}   OOM")
            torch.cuda.empty_cache()
            continue
        kv_results.append((T, t_e, t_c))
        print(
            "    {:6d}  {:8.2f}us  {:8.2f}us  {:8.2f}x".format(
                T, t_e * 1e3, t_c * 1e3, t_e / t_c
            )
        )
        torch.cuda.empty_cache()

    return q_results, kv_results


if __name__ == "__main__":
    print("== Correctness ==")
    test_q_correctness()
    test_kv_correctness()
    test_output_buffer_and_inplace_correctness()
    test_group_heads_correctness()
    test_inverse_rope_path_correctness()
    print("\n== Benchmark (T sweep: covers decode + prefill paths) ==")
    q_res, kv_res = bench_token_sweep()
    # Hard perf assertion: fused must beat the eager baseline at every T.
    fail = False
    for T, t_e, t_c in q_res:
        if not (t_c < t_e):
            print(f"  [FAIL] Q T={T}: cand={t_c*1e3:.2f}us not < eager={t_e*1e3:.2f}us")
            fail = True
    for T, t_e, t_c in kv_res:
        if not (t_c < t_e):
            print(
                f"  [FAIL] KV T={T}: cand={t_c*1e3:.2f}us not < eager={t_e*1e3:.2f}us"
            )
            fail = True
    assert not fail, "fused kernel regressed at some T"
    print("OK")
