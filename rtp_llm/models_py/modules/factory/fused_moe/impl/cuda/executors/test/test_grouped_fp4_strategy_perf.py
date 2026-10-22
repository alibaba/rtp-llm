"""Performance guard for the generic grouped FP4 routed-expert strategy."""

from __future__ import annotations

import unittest
from statistics import median

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4 import (
    GroupedFp4Executor,
    _has_fp8_fp4_grouped_kernel,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.utils.model_weight import W


def _cfg(E: int, D: int, inter: int, topk: int, tokens: int):
    return Fp8Fp4MoeRuntimeConfig(
        layer_id=0,
        hidden_size=D,
        moe_inter_dim=inter,
        expert_num=E,
        moe_k=topk,
        n_shared_experts=1,
        swiglu_limit=10.0,
        ep_size=1,
        ep_rank=0,
        max_tokens_per_rank=tokens,
        moe_strategy="auto",
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
        W.moe_w1: _fp4_weight(E * 2 * inter, D).view(E, 2 * inter, D // 2),
        W.moe_s1: _fp4_scale(E * 2 * inter, D).view(E, 2 * inter, D // 32),
        W.moe_w2: _fp4_weight(E * D, inter).view(E, D, inter // 2),
        W.moe_s2: _fp4_scale(E * D, inter).view(E, D, inter // 32),
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


def _bench(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _bench_pair(
    local,
    grouped,
    *,
    warmup: int = 5,
    iters: int = 12,
    repeats: int = 5,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        local()
        grouped()
    torch.cuda.synchronize()

    local_samples = []
    grouped_samples = []
    paired_ratios = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            local_ms = _bench(local, iters)
            grouped_ms = _bench(grouped, iters)
        else:
            grouped_ms = _bench(grouped, iters)
            local_ms = _bench(local, iters)
        local_samples.append(local_ms)
        grouped_samples.append(grouped_ms)
        paired_ratios.append(grouped_ms / local_ms)
    return median(local_samples), median(grouped_samples), median(paired_ratios)


class GroupedFP4StrategyPerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated SM100 target")
        if torch.cuda.get_device_capability()[0] != 10:
            raise AssertionError("SM100 is required by this dedicated SM100 target")
        if not _has_fp8_fp4_grouped_kernel():
            raise AssertionError(
                "grouped FP8xFP4 DeepGEMM kernel is required by this dedicated "
                "SM100 target"
            )

    def test_cuda_graph_capture_bs1_matches_eager(self):
        torch.manual_seed(20260515)
        E, D, inter, topk, tokens = 8, 512, 256, 6, 1
        cfg = _cfg(E, D, inter, topk, tokens)
        grouped = GroupedFp4Executor(
            cfg, FusedMoEQuantConfig(), _make_layer_weights(E, D, inter)
        )

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

        torch.manual_seed(20260517)
        x.copy_(torch.randn_like(x) * 0.2)
        new_weights = torch.rand_like(weights)
        weights.copy_(new_weights / new_weights.sum(dim=-1, keepdim=True))
        indices.copy_(torch.flip(indices, dims=(1,)))
        with torch.inference_mode():
            replay_eager_y = grouped(x, weights, indices).clone()
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(tuple(graph_y.shape), (tokens, D))
        self.assertTrue(torch.isfinite(graph_y).all().item())
        self.assertFalse(torch.equal(eager_y, replay_eager_y))
        self.assertLess(
            _relative_mean_error(replay_eager_y, graph_y),
            0.05,
        )

    def test_decode_and_cuda_graph_ab_performance(self):
        torch.manual_seed(20260516)
        E, D, inter, topk, tokens = 16, 512, 256, 6, 8
        cfg = _cfg(E, D, inter, topk, tokens)
        layer_weights = _make_layer_weights(E, D, inter)
        local = LocalLoopExecutor(
            cfg, FusedMoEQuantConfig(), _clone_weights(layer_weights)
        )
        grouped = GroupedFp4Executor(
            cfg, FusedMoEQuantConfig(), _clone_weights(layer_weights)
        )
        x, weights, indices = _make_inputs(tokens, D, E, topk)

        with torch.inference_mode():
            local_eager = local(x, weights, indices).clone()
            grouped_eager = grouped(x, weights, indices).clone()
        self.assertLess(_relative_mean_error(local_eager, grouped_eager), 0.05)

        def capture(executor):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    graph_out = executor(x, weights, indices)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                graph_out = executor(x, weights, indices)
            return graph, graph_out

        local_graph, local_graph_out = capture(local)
        grouped_graph, grouped_graph_out = capture(grouped)
        local_graph.replay()
        grouped_graph.replay()
        torch.cuda.synchronize()
        self.assertLess(
            _relative_mean_error(local_eager, local_graph_out),
            0.05,
        )
        self.assertLess(
            _relative_mean_error(grouped_eager, grouped_graph_out),
            0.05,
        )

        local_ms, grouped_ms, paired_ratio = _bench_pair(
            local_graph.replay,
            grouped_graph.replay,
            warmup=5,
            iters=100,
            repeats=5,
        )
        print(
            f"[grouped_fp4 graph] tokens={tokens}: "
            f"local={local_ms:.3f}ms grouped={grouped_ms:.3f}ms"
        )
        self.assertLess(
            paired_ratio,
            1.10,
            f"grouped FP4 graph regressed by more than 10%: "
            f"local={local_ms:.3f}ms grouped={grouped_ms:.3f}ms "
            f"median paired ratio={paired_ratio:.3f}",
        )

    def test_grouped_fp4_beats_local_loop(self):
        torch.manual_seed(20260514)
        E, D, inter, topk, tokens = 32, 512, 256, 6, 1024
        cfg = _cfg(E, D, inter, topk, tokens)
        layer_weights = _make_layer_weights(E, D, inter)

        local = LocalLoopExecutor(
            cfg, FusedMoEQuantConfig(), _clone_weights(layer_weights)
        )
        grouped = GroupedFp4Executor(
            cfg, FusedMoEQuantConfig(), _clone_weights(layer_weights)
        )

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
            local_ms, grouped_ms, paired_ratio = _bench_pair(
                lambda: local(x, weights, indices),
                lambda: grouped(x, weights, indices),
            )

        print(
            f"[grouped_fp4] tokens={tokens} E={E} D={D} inter={inter} "
            f"topk={topk}: local={local_ms:.3f}ms grouped={grouped_ms:.3f}ms "
            f"paired_speedup={1.0 / paired_ratio:.2f}x "
            f"correctness_rel={correctness_rel:.4f}"
        )
        self.assertLess(
            paired_ratio,
            0.90,
            f"grouped FP4 path was not faster: local={local_ms:.3f}ms "
            f"grouped={grouped_ms:.3f}ms median paired ratio={paired_ratio:.3f}",
        )

    def test_tp2_loader_setup_and_forward_keep_full_cuda_layout(self):
        from rtp_llm.device.device_type import DeviceType, get_device_type
        from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
        from rtp_llm.model_loader.load_config import LoadConfig

        if (
            get_device_type() != DeviceType.Cuda
            or torch.cuda.get_device_capability()[0] != 10
        ):
            self.skipTest("requires CUDA SM100")
        self.assertTrue(_has_fp8_fp4_grouped_kernel())
        torch.manual_seed(1396)
        E, D, inter, topk, tokens = 8, 512, 256, 6, 32
        raw = _make_layer_weights(E, D, inter)
        cfg = _cfg(E, D, inter, topk, tokens)
        baseline = GroupedFp4Executor(cfg, FusedMoEQuantConfig(), _clone_weights(raw))
        x, weights, indices = _make_inputs(tokens, D, E, topk)
        with torch.inference_mode():
            expected = baseline(x, weights, indices).clone()
            for rank in (0, 1):
                load = LoadConfig.model_construct(
                    tp_size=2,
                    tp_rank=rank,
                    ep_size=1,
                    ep_rank=0,
                    dp_size=1,
                    dp_rank=0,
                    ffn_tp_size=1,
                    ffn_tp_rank=0,
                    hidden_size=D,
                    head_num=1,
                    head_num_kv=1,
                    size_per_head=D,
                    moe_pure_tp_mode=True,
                    compute_dtype=torch.bfloat16,
                    weight_preparation=None,
                )
                loaded = {}
                for name, tensor in raw.items():
                    loaded.update(MoeAtomicWeight(name, [])._split(tensor, load))
                    self.assertEqual(loaded[name].shape, tensor.shape)
                cfg.tp_size = 2
                strategy = GroupedFp4Executor(cfg, FusedMoEQuantConfig(), loaded)
                actual = strategy(x, weights, indices)
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
