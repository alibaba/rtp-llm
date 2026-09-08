"""UT for replacing ``block._RMSNorm.forward`` (torch 6-launch chain) with
the framework C++ ``rtp_llm_ops.rmsnorm`` single-launch op.

Audit doc §7.3.4 / row #4, #22, #31, #33: dsv4's ``_RMSNorm`` (block.py) is
used at attn_norm / ffn_norm / MTP enorm / hnorm / norm / transformer final
norm.  Pre-integration each forward was the classic 6-launch
``x.float().square().mean().rsqrt()...`` pattern — ~1032 launches / step
across 43 layers per the trace.  Live ``_RMSNorm.forward`` now delegates to
``rtp_llm_ops.rmsnorm`` (matching vLLM's bf16-weight convention).

This UT verifies:
  1) Numerical agreement of the live C++ path vs the pinned pre-integration
     torch baseline, across decode + prefill shapes.
  2) Wall-clock improvement vs the torch baseline.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=7 \
    /opt/conda310/bin/python3 \
    rtp_llm/models_py/modules/dsv4/test/test_rmsnorm_replace.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.block import _RMSNorm as LiveRMSNorm


class _TorchFallbackRMSNorm(nn.Module):
    """The original 6-launch torch RMSNorm — pinned here as the UT baseline.
    Used to measure the speedup vs the pre-integration path.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x32).to(dtype)


def _build_pair(dim: int, seed: int = 0):
    """Build a torch-baseline + live C++ ``_RMSNorm`` pair with matched weights
    (bf16 for the live path, fp32 (upcast of the same bf16 bytes) for the
    baseline so the RMS math is numerically aligned)."""
    torch.manual_seed(seed)
    live = LiveRMSNorm(dim).cuda()
    baseline = _TorchFallbackRMSNorm(dim).cuda()
    w_bf = (torch.randn(dim, device="cuda") * 0.1 + 1.0).abs().to(torch.bfloat16)
    with torch.no_grad():
        live.weight.copy_(w_bf)
        baseline.weight.copy_(w_bf.float())
    return baseline, live


def _bench(fn, *args, warmup: int = 25, iters: int = 200) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def test_live_matches_pinned_baseline():
    """``block._RMSNorm`` now owns a bf16 weight and calls C++ ``rtp_llm_ops.rmsnorm``.
    Compare against the pinned torch baseline, **both fed identical bf16
    weight** — verifies the algorithm matches, independent of the fp32→bf16
    weight downgrade. vLLM's DeepSeek V4 uses the same bf16-weight convention.
    """
    from rtp_llm.models_py.modules.dsv4.block import _RMSNorm as LiveRMSNorm

    dim = 4096
    torch.manual_seed(42)
    live = LiveRMSNorm(dim).cuda()
    baseline = _TorchFallbackRMSNorm(dim).cuda()
    with torch.no_grad():
        # Build weight in bf16 so both paths operate on the same numeric input.
        w_bf = (torch.randn(dim, device="cuda") * 0.1 + 1.0).abs().to(torch.bfloat16)
        live.weight.copy_(w_bf)
        # Baseline parameter is fp32 but we copy bf16 values (exact up-cast).
        baseline.weight.copy_(w_bf.float())
    x = torch.randn(1, 128, dim, dtype=torch.bfloat16, device="cuda") * 0.3
    y_live = live(x)
    y_ref = baseline(x)
    d = (y_live.float() - y_ref.float()).abs()
    print(f"  [live vs pinned baseline]  max={d.max():.4e}  mean={d.mean():.4e}")
    assert (
        d.max() <= 2e-2
    ), f"live _RMSNorm vs pinned baseline diff > 1 bf16 ULP: {d.max()}"


def test_correctness():
    """Match across typical attn_norm / ffn_norm / final norm shapes."""
    dim = 4096  # dsv4 hidden_size
    baseline, live = _build_pair(dim)

    for B, T in [(1, 1), (1, 8), (1, 64), (1, 128), (1, 256), (1, 4096), (4, 1024)]:
        x = torch.randn(B, T, dim, dtype=torch.bfloat16, device="cuda") * 0.3
        with torch.no_grad():
            y_live = live(x)
            y_ref = baseline(x)
        d = (y_live.float() - y_ref.float()).abs()
        print(f"  [dim={dim} B={B} T={T}]  max diff={d.max():.4e}  mean={d.mean():.4e}")
        # Both paths do fp32 intermediates; diff is at most 1 bf16 ULP from
        # variance-accumulation order differences.
        assert d.max() <= 2e-2, (
            f"_RMSNorm replacement diff exceeds ~1 bf16 ULP tol @ "
            f"B={B} T={T}: {d.max()}"
        )


def bench_token_sweep():
    dim = 4096
    baseline, live = _build_pair(dim, seed=1)

    print(f"  [dim={dim}]")
    print(
        "    {:>6}  {:>10}  {:>10}  {:>10}".format(
            "T", "torch (6-launch)", "C++ rmsnorm", "speedup"
        )
    )
    Tlist = [1, 8, 16, 64, 128, 256, 4096, 65536]
    results = []
    for T in Tlist:
        x = torch.randn(1, T, dim, dtype=torch.bfloat16, device="cuda")

        def run_baseline(x):
            return baseline(x)

        def run_live(x):
            return live(x)

        try:
            t_base = _bench(run_baseline, x)
            t_live = _bench(run_live, x)
        except torch.cuda.OutOfMemoryError:
            print(f"    {T:6d}   OOM")
            torch.cuda.empty_cache()
            continue
        results.append((T, t_base, t_live))
        print(
            "    {:6d}  {:8.2f}us  {:8.2f}us  {:8.2f}x".format(
                T, t_base * 1e3, t_live * 1e3, t_base / t_live
            )
        )
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    print("== Correctness ==")
    test_live_matches_pinned_baseline()
    test_correctness()
    print("\n== Benchmark (T sweep) ==")
    results = bench_token_sweep()
    fail = False
    for T, t_base, t_live in results:
        if not (t_live < t_base):
            print(
                f"  [FAIL] T={T}: live C++={t_live*1e3:.2f}us "
                f"not < torch={t_base*1e3:.2f}us"
            )
            fail = True
    assert not fail
    print("OK")
