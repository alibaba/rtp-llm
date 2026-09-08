"""Performance guard for the DSV4 grouped FP4 routed-expert strategy."""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MoeCfg,
    _has_fp8_fp4_grouped_kernel,
)
from rtp_llm.utils.model_weight import W


def _cfg(E: int, D: int, inter: int, topk: int, tokens: int) -> MoeCfg:
    return MoeCfg(
        layer_id=0,
        dim=D,
        moe_inter_dim=inter,
        n_routed_experts=E,
        n_activated_experts=topk,
        swiglu_limit=10.0,
        ep_size=1,
        ep_rank=0,
        n_local_experts=E,
        local_expert_start=0,
        local_expert_end=E,
        max_tokens_per_rank=tokens,
    )


def _fp4_weight(out_dim: int, in_dim: int) -> torch.Tensor:
    return torch.randint(
        -128,
        127,
        (out_dim, in_dim // 2),
        dtype=torch.int8,
        device="cuda",
    )


def _fp4_scale(out_dim: int, in_dim: int) -> torch.Tensor:
    return torch.full(
        (out_dim, in_dim // 32),
        120,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)


def _make_layer_weights(E: int, D: int, inter: int) -> dict:
    return {
        W.v4_routed_w1_w: _fp4_weight(E * inter, D).view(E, inter, D // 2),
        W.v4_routed_w1_s: _fp4_scale(E * inter, D).view(E, inter, D // 32),
        W.v4_routed_w2_w: _fp4_weight(E * D, inter).view(E, D, inter // 2),
        W.v4_routed_w2_s: _fp4_scale(E * D, inter).view(E, D, inter // 32),
        W.v4_routed_w3_w: _fp4_weight(E * inter, D).view(E, inter, D // 2),
        W.v4_routed_w3_s: _fp4_scale(E * inter, D).view(E, inter, D // 32),
    }


def _clone_weights(layer_weights: dict) -> dict:
    return {name: tensor.clone() for name, tensor in layer_weights.items()}


def _make_inputs(tokens: int, D: int, E: int, topk: int):
    x = torch.randn(tokens, D, dtype=torch.bfloat16, device="cuda") * 0.2
    indices = (
        torch.arange(tokens * topk, dtype=torch.int64, device="cuda")
        .view(tokens, topk)
        .remainder_(E)
    )
    weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x, weights, indices


def _relative_mean_error(ref: torch.Tensor, got: torch.Tensor) -> float:
    diff = (ref.float() - got.float()).abs().mean().item()
    scale = ref.float().abs().mean().item() + 1e-6
    return diff / scale


def _bench(fn, warmup: int = 5, iters: int = 12, repeats: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return sorted(samples)[len(samples) // 2]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GroupedFP4StrategyPerfTest(unittest.TestCase):
    def test_cuda_graph_capture_bs1_matches_eager(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM100 required")
        if not _has_fp8_fp4_grouped_kernel():
            self.skipTest("grouped FP8xFP4 DeepGEMM kernel unavailable")

        torch.manual_seed(20260515)
        E, D, inter, topk, tokens = 8, 512, 256, 6, 1
        cfg = _cfg(E, D, inter, topk, tokens)
        grouped = GroupedFP4Strategy(cfg)
        grouped.setup_weights(_make_layer_weights(E, D, inter))

        x, weights, indices = _make_inputs(tokens, D, E, topk)
        with torch.inference_mode():
            eager_y = grouped(x, weights, indices)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                graph_y = grouped(x, weights, indices)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            graph_y = grouped(x, weights, indices)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(tuple(graph_y.shape), (tokens, D))
        self.assertTrue(torch.isfinite(graph_y).all().item())
        self.assertLess(
            _relative_mean_error(eager_y, graph_y),
            0.05,
        )

    def test_grouped_fp4_beats_local_loop(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM100 required")
        if not _has_fp8_fp4_grouped_kernel():
            self.skipTest("grouped FP8xFP4 DeepGEMM kernel unavailable")

        torch.manual_seed(20260514)
        E, D, inter, topk, tokens = 32, 512, 256, 6, 1024
        cfg = _cfg(E, D, inter, topk, tokens)
        layer_weights = _make_layer_weights(E, D, inter)

        local = LocalLoopStrategy(cfg)
        local.setup_weights(_clone_weights(layer_weights))
        grouped = GroupedFP4Strategy(cfg)
        grouped.setup_weights(_clone_weights(layer_weights))

        x_check, _, indices_check = _make_inputs(256, D, E, topk)
        weights_check = torch.ones(256, topk, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            local_y = local(x_check, weights_check, indices_check)
            grouped_y = grouped(x_check, weights_check, indices_check)
        torch.cuda.synchronize()

        self.assertEqual(tuple(local_y.shape), (256, D))
        self.assertEqual(tuple(grouped_y.shape), (256, D))
        self.assertTrue(torch.isfinite(local_y).all().item())
        self.assertTrue(torch.isfinite(grouped_y).all().item())
        correctness_rel = _relative_mean_error(local_y, grouped_y)
        self.assertLess(
            correctness_rel,
            0.05,
            f"grouped FP4 output diverged from local loop: rel={correctness_rel:.4f}",
        )

        x, weights, indices = _make_inputs(tokens, D, E, topk)
        with torch.inference_mode():
            perf_y = grouped(x, weights, indices)
        torch.cuda.synchronize()
        self.assertEqual(tuple(perf_y.shape), (tokens, D))
        self.assertTrue(torch.isfinite(perf_y).all().item())

        with torch.inference_mode():
            local_ms = _bench(lambda: local(x, weights, indices))
            grouped_ms = _bench(lambda: grouped(x, weights, indices))

        print(
            f"[DSV4 grouped_fp4] tokens={tokens} E={E} D={D} inter={inter} "
            f"topk={topk}: local={local_ms:.3f}ms grouped={grouped_ms:.3f}ms "
            f"speedup={local_ms / grouped_ms:.2f}x correctness_rel={correctness_rel:.4f}"
        )
        self.assertLess(
            grouped_ms,
            local_ms * 0.90,
            f"grouped FP4 path was not faster: local={local_ms:.3f}ms "
            f"grouped={grouped_ms:.3f}ms",
        )


if __name__ == "__main__":
    unittest.main()
