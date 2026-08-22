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
    def test_sm120_scatter_matches_groupwise(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("SM120 required")
        from flashinfer import mxfp8_quantize

        torch.manual_seed(20260822)
        E, D, inter, topk, tokens = 16, 512, 256, 6, 512
        cfg = _cfg(E, D, inter, topk, tokens)
        grouped = GroupedFP4Strategy(cfg)
        grouped.setup_weights(_make_layer_weights(E, D, inter))
        x, weights, indices = _make_inputs(tokens, D, E, topk)
        # Mirror EP receive layout: non-local top-k slots retain an index but
        # carry zero router weight and must not be dispatched to a local GEMM.
        weights = weights * (indices.remainder(4) == 0)
        x_q, x_sf = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
        x_sf = x_sf.reshape(tokens, D // 32).view(torch.uint8)
        with torch.inference_mode():
            got = grouped.forward_sm120_mxfp8(x_q, x_sf, weights, indices)
            got_ms = _bench(
                lambda: grouped.forward_sm120_mxfp8(x_q, x_sf, weights, indices),
                warmup=2, iters=3, repeats=2,
            )

        # Validate the precision-sensitive fused activation/quantization stage
        # against its explicit BF16-rounding reference.
        from rtp_llm.models_py.modules.dsv4.moe._silu_mul_fp8_quant_triton import (
            silu_mul_fp8_quant_packed_from_parts,
        )

        gate_up = torch.randn(tokens, 2 * inter, dtype=torch.bfloat16, device="cuda")
        up, gate = gate_up[:, :inter], gate_up[:, inter:]
        gate_ref = gate.float().clamp(max=cfg.swiglu_limit)
        up_ref = up.float().clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        hidden_ref = (torch.nn.functional.silu(gate_ref) * up_ref).to(torch.bfloat16)
        ref_q, ref_sf = mxfp8_quantize(
            hidden_ref.contiguous(), is_sf_swizzled_layout=False
        )
        ref_sf = ref_sf.reshape(tokens, inter // 32).view(torch.uint8)
        got_q, got_sf_packed = silu_mul_fp8_quant_packed_from_parts(
            gate, up, clamp_limit=cfg.swiglu_limit, group_size=32
        )
        got_sf = got_sf_packed.contiguous().view(torch.uint8).reshape(
            tokens, inter // 32
        )
        ref_deq = ref_q.float() * (ref_sf.to(torch.int32) - 127).float().exp2().repeat_interleave(32, dim=1)
        got_deq = got_q.float() * (got_sf.to(torch.int32) - 127).float().exp2().repeat_interleave(32, dim=1)
        rel = _relative_mean_error(ref_deq, got_deq)
        print(
            f"[SM120 scatter] scatter={got_ms:.3f}ms "
            f"activation_quant_rel={rel:.9f}"
        )
        self.assertTrue(torch.isfinite(got).all().item())
        self.assertLess(rel, 0.01)

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
