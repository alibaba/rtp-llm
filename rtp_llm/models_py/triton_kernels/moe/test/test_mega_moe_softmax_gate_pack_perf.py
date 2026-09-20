"""Softmax gate-pack should stay close to the sqrtsoftplus fused path."""

from __future__ import annotations

import unittest
from statistics import median
from types import SimpleNamespace

import torch

from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
    fused_pack_mega_moe_gate_inputs,
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


def _bench_pair(baseline, candidate, warmup: int = 30, iters: int = 80, rounds: int = 5):
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    baseline_times = []
    candidate_times = []
    ratios = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            baseline_ms = _time_batch(baseline, iters)
            candidate_ms = _time_batch(candidate, iters)
        else:
            candidate_ms = _time_batch(candidate, iters)
            baseline_ms = _time_batch(baseline, iters)
        baseline_times.append(baseline_ms)
        candidate_times.append(candidate_ms)
        ratios.append(candidate_ms / baseline_ms)
    return median(baseline_times), median(candidate_times), median(ratios)


class MegaMoeSoftmaxGatePackPerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated SM100 Bazel target")

    def _case(self, tokens: int, dim: int = 4096, experts: int = 512, topk: int = 10):
        torch.manual_seed(tokens + 7)
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16) * 0.3
        scores = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(experts, device="cuda", dtype=torch.float32) * 0.1
        sqrt_buf = _make_buf(tokens, dim, topk, "cuda")
        softmax_buf = _make_buf(tokens, dim, topk, "cuda")

        def run_sqrtsoftplus():
            fused_pack_mega_moe_gate_inputs(
                x,
                scores,
                sqrt_buf.x,
                sqrt_buf.x_sf,
                sqrt_buf.topk_idx,
                sqrt_buf.topk_weights,
                topk=topk,
                score_func="sqrtsoftplus",
                route_scale=1.0,
                bias=bias,
            )

        def run_softmax():
            fused_pack_mega_moe_gate_inputs(
                x,
                scores,
                softmax_buf.x,
                softmax_buf.x_sf,
                softmax_buf.topk_idx,
                softmax_buf.topk_weights,
                topk=topk,
                score_func="softmax",
                route_scale=1.0,
            )

        run_sqrtsoftplus()
        run_softmax()
        torch.cuda.synchronize()
        self.assertEqual(tuple(softmax_buf.x.shape), (tokens, dim))

        iters = 200 if tokens <= 256 else (80 if tokens <= 4096 else 30)
        sqrt_ms, softmax_ms, ratio = _bench_pair(
            run_sqrtsoftplus, run_softmax, iters=iters
        )
        print(
            f"[MegaMoE softmax gate-pack] T={tokens:5d} D={dim} E={experts} topk={topk}: "
            f"sqrtsoftplus={sqrt_ms * 1000:.2f}us softmax={softmax_ms * 1000:.2f}us "
            f"ratio={ratio:.3f}"
        )
        return sqrt_ms, softmax_ms, ratio

    def test_softmax_matches_sqrtsoftplus_token_sweep(self):
        # Decode through 64k prefill. Softmax should stay within ~20% of
        # the production sqrtsoftplus fused path on the same pack+route grid.
        tokens_list = (1, 8, 64, 256, 1024, 8192, 32768, 65536)
        gated = {}
        for tokens in tokens_list:
            sqrt_ms, softmax_ms, ratio = self._case(tokens)
            gated[tokens] = (sqrt_ms, softmax_ms, ratio)

        regressions = []
        for tokens, (sqrt_ms, softmax_ms, ratio) in gated.items():
            limit = 1.25 if tokens <= 64 else 1.20
            if ratio > limit:
                regressions.append(
                    f"T={tokens}: softmax={softmax_ms * 1000:.2f}us, "
                    f"sqrtsoftplus={sqrt_ms * 1000:.2f}us, ratio={ratio:.3f}"
                )
        self.assertFalse(
            regressions,
            "softmax gate-pack lagged the sqrtsoftplus path: " + "; ".join(regressions),
        )

        representative_ratio = median(value[2] for value in gated.values())
        self.assertLessEqual(
            representative_ratio,
            1.15,
            "softmax gate-pack representative ratio "
            f"{representative_ratio:.3f} exceeded 1.15x of sqrtsoftplus",
        )


if __name__ == "__main__":
    unittest.main()
