"""CUDA validation coverage for local-loop expert indices."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    normalize_moe_w13_gate_up,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.utils.model_weight import W

_PROBE = textwrap.dedent(
    """
    import sys
    import torch

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
        _validate_topk_indices,
    )

    mode = sys.argv[1]
    invalid = int(sys.argv[2])
    indices = torch.zeros((1, 2), dtype=torch.long, device="cuda")
    if mode == "eager":
        indices[0, 1] = invalid
        _validate_topk_indices(indices, 2)
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _validate_topk_indices(indices, 2)
        indices[0, 1] = invalid
        graph.replay()
    torch.cuda.synchronize()
    """
)


def _cfg(experts: int, dim: int, inter: int, topk: int, tokens: int):
    return Fp8Fp4MoeRuntimeConfig(
        layer_id=0,
        hidden_size=dim,
        moe_inter_dim=inter,
        expert_num=experts,
        moe_k=topk,
        n_shared_experts=1,
        swiglu_limit=10.0,
        ep_size=1,
        ep_rank=0,
        max_tokens_per_rank=tokens,
        moe_strategy="local_loop",
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


def _make_layer_weights(experts: int, dim: int, inter: int) -> dict:
    return {
        W.moe_w1: _fp4_weight(experts * 2 * inter, dim).view(
            experts, 2 * inter, dim // 2
        ),
        W.moe_s1: _fp4_scale(experts * 2 * inter, dim).view(
            experts, 2 * inter, dim // 32
        ),
        W.moe_w2: _fp4_weight(experts * dim, inter).view(experts, dim, inter // 2),
        W.moe_s2: _fp4_scale(experts * dim, inter).view(experts, dim, inter // 32),
    }


def _make_inputs(tokens: int, dim: int, experts: int, topk: int):
    x = torch.randn(tokens, dim, dtype=torch.bfloat16, device="cuda") * 0.2
    indices = (
        torch.arange(tokens * topk, dtype=torch.int64, device="cuda")
        .view(tokens, topk)
        .remainder_(experts)
    )
    weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
    weights.div_(weights.sum(dim=-1, keepdim=True))
    return x, weights, indices


def _relative_mean_error(ref: torch.Tensor, got: torch.Tensor) -> float:
    diff = (ref.float() - got.float()).abs().mean().item()
    scale = ref.float().abs().mean().item() + 1.0e-6
    return diff / scale


class LocalLoopCudaIndexValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated target")
        if torch.cuda.get_device_capability()[0] != 10:
            raise AssertionError("SM100 is required by this dedicated target")

    def _assert_probe_rejects(self, mode: str, invalid: int) -> None:
        env = os.environ.copy()
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, mode, str(invalid)],
            text=True,
            capture_output=True,
            env=env,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("assert", output)

    def test_eager_rejects_negative_and_out_of_range_indices(self):
        for invalid in (-1, 2):
            with self.subTest(invalid=invalid):
                self._assert_probe_rejects("eager", invalid)

    def test_cuda_graph_replay_rejects_negative_and_out_of_range_indices(self):
        for invalid in (-1, 2):
            with self.subTest(invalid=invalid):
                self._assert_probe_rejects("graph", invalid)

    def test_w13_layout_preserves_e8m0_scale_dtype_and_bytes(self):
        for device in ("cpu", "cuda"):
            gate = torch.tensor([0, 127], dtype=torch.uint8, device=device).reshape(
                1, 2, 1
            )
            up = torch.tensor([128, 255], dtype=torch.uint8, device=device).reshape(
                1, 2, 1
            )
            expected = torch.cat((gate, up), dim=-2)
            for layout, parts in (("gate_up", (gate, up)), ("up_gate", (up, gate))):
                with self.subTest(device=device, layout=layout):
                    scales = torch.cat(parts, dim=-2).view(torch.float8_e8m0fnu)
                    weights = torch.ones((1, 4, 1), dtype=torch.int8, device=device)

                    _, normalized = normalize_moe_w13_gate_up(
                        weights, scales, 2, layout
                    )

                    self.assertEqual(normalized.dtype, scales.dtype)
                    torch.testing.assert_close(normalized.view(torch.uint8), expected)

    def _assert_graph_replay_matches_eager(
        self, tokens: int, topk_dispatch_max_n: int
    ) -> None:
        torch.manual_seed(20260906 + tokens + topk_dispatch_max_n)
        experts, dim, inter, topk = 8, 512, 256, 2
        executor = LocalLoopExecutor(
            _cfg(experts, dim, inter, topk, tokens),
            FusedMoEQuantConfig(),
            _make_layer_weights(experts, dim, inter),
        )
        x, weights, indices = _make_inputs(tokens, dim, experts, topk)

        with patch.dict(
            os.environ,
            {"MOE_LOCAL_LOOP_TOPK_MAX_N": str(topk_dispatch_max_n)},
        ), torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    executor(x, weights, indices)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = executor(x, weights, indices)
            initial_output = graph_output.clone()

            x.copy_(torch.randn_like(x) * 0.2)
            new_weights = torch.rand_like(weights)
            weights.copy_(new_weights / new_weights.sum(dim=-1, keepdim=True))
            indices.add_(1).remainder_(experts)
            executor._W1_w.view(torch.uint8)[0, 0, 0].bitwise_xor_(1)

            expected = executor(x, weights, indices).clone()
            # Replay must write the captured allocation, independently of eager.
            graph_output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()

        self.assertFalse(torch.equal(initial_output, expected))
        self.assertTrue(torch.isfinite(graph_output).all().item())
        self.assertLess(_relative_mean_error(expected, graph_output), 0.05)

    def test_cuda_graph_bs1_topk_dispatch_matches_eager_after_mutation(self):
        self._assert_graph_replay_matches_eager(tokens=1, topk_dispatch_max_n=32)

    def test_cuda_graph_batched_topk_dispatch_matches_eager_after_mutation(self):
        self._assert_graph_replay_matches_eager(tokens=3, topk_dispatch_max_n=32)

    def test_cuda_graph_dense_fallback_matches_eager_after_mutation(self):
        self._assert_graph_replay_matches_eager(tokens=3, topk_dispatch_max_n=2)

    def test_multiple_capture_sizes_keep_earlier_graph_output_storage(self):
        torch.manual_seed(2026090601)
        experts, dim, inter, topk = 8, 512, 256, 2
        executor = LocalLoopExecutor(
            _cfg(experts, dim, inter, topk, tokens=3),
            FusedMoEQuantConfig(),
            _make_layer_weights(experts, dim, inter),
        )
        x1, weights1, indices1 = _make_inputs(1, dim, experts, topk)
        x3, weights3, indices3 = _make_inputs(3, dim, experts, topk)

        with patch.dict(
            os.environ, {"MOE_LOCAL_LOOP_TOPK_MAX_N": "32"}
        ), torch.inference_mode():
            for x, weights, indices in (
                (x1, weights1, indices1),
                (x3, weights3, indices3),
            ):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        executor(x, weights, indices)
                torch.cuda.current_stream().wait_stream(stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = executor(x, weights, indices)
                if x.size(0) == 1:
                    first_graph = graph
                    first_graph_output = graph_output
                    first_output_ptr = graph_output.data_ptr()
                else:
                    second_output_ptr = graph_output.data_ptr()

            self.assertNotEqual(first_output_ptr, second_output_ptr)

            x1.copy_(torch.randn_like(x1) * 0.2)
            new_weights = torch.rand_like(weights1)
            weights1.copy_(new_weights / new_weights.sum(dim=-1, keepdim=True))
            indices1.add_(1).remainder_(experts)
            expected = executor(x1, weights1, indices1).clone()

            # A later capture and eager call must not invalidate the first graph.
            first_graph_output.fill_(float("nan"))
            first_graph.replay()
            torch.cuda.synchronize()

        self.assertTrue(torch.isfinite(first_graph_output).all().item())
        self.assertLess(_relative_mean_error(expected, first_graph_output), 0.05)

    def test_multilayer_prefill_does_not_retain_fp32_workspaces(self):
        experts, dim, inter, topk, tokens = 2, 512, 256, 2, 2048
        executors = [
            LocalLoopExecutor(
                _cfg(experts, dim, inter, topk, tokens),
                FusedMoEQuantConfig(),
                _make_layer_weights(experts, dim, inter),
            )
            for _ in range(8)
        ]
        x, weights, indices = _make_inputs(tokens, dim, experts, topk)

        def measure(layers):
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for executor in layers:
                # The strategy consumes the FP32 result and returns BF16.
                output = executor(x, weights, indices).to(x.dtype)
                del output
            torch.cuda.synchronize()
            return (
                torch.cuda.max_memory_allocated() - baseline,
                torch.cuda.memory_allocated() - baseline,
            )

        with torch.inference_mode():
            # Exclude one-time GEMM initialization from workspace measurements.
            executors[0](x, weights, indices)
            one_peak, one_steady = measure(executors[:1])
            many_peak, many_steady = measure(executors)

        workspace_bytes = tokens * dim * 4
        self.assertLessEqual(one_steady, 65536)
        self.assertLessEqual(many_steady, 65536)
        self.assertLessEqual(many_peak, one_peak + workspace_bytes)

    def test_destroyed_graphs_release_their_accumulators(self):
        experts, dim, inter, topk = 2, 512, 256, 2
        executor = LocalLoopExecutor(
            _cfg(experts, dim, inter, topk, tokens=512),
            FusedMoEQuantConfig(),
            _make_layer_weights(experts, dim, inter),
        )
        inputs = [_make_inputs(n, dim, experts, topk) for n in (128, 256, 512)]
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for x, weights, indices in inputs:
                    executor(x, weights, indices)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()

            for x, weights, indices in inputs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = executor(x, weights, indices)
                graph.replay()
                torch.cuda.synchronize()
                del output, graph
                gc.collect()
                torch.cuda.synchronize()

            self.assertLessEqual(torch.cuda.memory_allocated() - baseline, 65536)


if __name__ == "__main__":
    unittest.main()
