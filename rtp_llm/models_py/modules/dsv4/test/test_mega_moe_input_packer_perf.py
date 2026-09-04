"""Performance guard for the common MegaMoE input packer."""

from __future__ import annotations

import unittest
from statistics import median
from types import SimpleNamespace

import torch

from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
    fused_pack_mega_moe_inputs_legacy,
    fused_pack_mega_moe_inputs_optimized,
)


def _make_buf(tokens: int, dim: int, topk: int, device: str):
    return SimpleNamespace(
        x=torch.empty((tokens, dim), dtype=torch.float8_e4m3fn, device=device),
        x_sf=torch.empty((tokens, dim // 128), dtype=torch.int32, device=device),
        topk_idx=torch.empty((tokens, topk), dtype=torch.int64, device=device),
        topk_weights=torch.empty((tokens, topk), dtype=torch.float32, device=device),
    )


def _time_batch(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _bench_pair(
    legacy, optimized, warmup: int = 30, iters: int = 200, rounds: int = 7
) -> tuple[float, float, float]:
    for _ in range(warmup):
        legacy()
        optimized()
    torch.cuda.synchronize()

    legacy_times = []
    optimized_times = []
    ratios = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            legacy_ms = _time_batch(legacy, iters)
            optimized_ms = _time_batch(optimized, iters)
        else:
            optimized_ms = _time_batch(optimized, iters)
            legacy_ms = _time_batch(legacy, iters)
        legacy_times.append(legacy_ms)
        optimized_times.append(optimized_ms)
        ratios.append(optimized_ms / legacy_ms)
    return median(legacy_times), median(optimized_times), median(ratios)


class MegaMoEInputPackerPerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA is required by this dedicated SM100 Bazel target"
            )

    def _case(self, tokens: int, dim: int = 4096, topk: int = 6):
        torch.manual_seed(tokens)
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16) * 0.3
        weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        )
        ref = _make_buf(tokens, dim, topk, "cuda")
        got = _make_buf(tokens, dim, topk, "cuda")

        def run_legacy():
            fused_pack_mega_moe_inputs_legacy(
                x, weights, indices, ref.x, ref.x_sf, ref.topk_idx, ref.topk_weights
            )

        def run_optimized():
            fused_pack_mega_moe_inputs_optimized(
                x, weights, indices, got.x, got.x_sf, got.topk_idx, got.topk_weights
            )

        run_legacy()
        run_optimized()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(ref.x.view(torch.uint8).cpu(), got.x.view(torch.uint8).cpu())
        )
        self.assertTrue(torch.equal(ref.x_sf.cpu(), got.x_sf.cpu()))
        self.assertTrue(torch.equal(ref.topk_idx.cpu(), got.topk_idx.cpu()))
        self.assertTrue(torch.equal(ref.topk_weights.cpu(), got.topk_weights.cpu()))

        legacy_ms, optimized_ms, paired_ratio = _bench_pair(run_legacy, run_optimized)
        print(
            f"[MegaMoE pack] T={tokens:5d} D={dim} topk={topk}: "
            f"legacy={legacy_ms * 1000:.2f}us optimized={optimized_ms * 1000:.2f}us "
            f"paired_speedup={1.0 / paired_ratio:.2f}x"
        )
        return legacy_ms, optimized_ms, paired_ratio

    def test_perf_token_sweep(self):
        gated = {}
        for tokens in (1, 4, 16, 64, 256, 1024, 8192):
            legacy_ms, optimized_ms, paired_ratio = self._case(tokens)
            if tokens in (16, 64, 256, 1024):
                gated[tokens] = (legacy_ms, optimized_ms, paired_ratio)

        regressions = []
        for tokens, (legacy_ms, optimized_ms, paired_ratio) in gated.items():
            if paired_ratio > 0.84:
                regressions.append(
                    f"T={tokens}: optimized={optimized_ms * 1000:.2f}us, "
                    f"legacy={legacy_ms * 1000:.2f}us, ratio={paired_ratio:.3f}"
                )
        self.assertFalse(
            regressions,
            "MegaMoE packer missed the per-bucket speedup floor: "
            + "; ".join(regressions),
        )

        # Pairing and alternating call order reduces clock and launch-order bias.
        # Require at least 17% representative latency reduction while allowing
        # one percentage point of per-bucket measurement variation.
        representative_ratio = median(value[2] for value in gated.values())
        self.assertLessEqual(
            representative_ratio,
            0.85,
            "MegaMoE packer representative perf gate failed: "
            f"median paired ratio={representative_ratio:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
