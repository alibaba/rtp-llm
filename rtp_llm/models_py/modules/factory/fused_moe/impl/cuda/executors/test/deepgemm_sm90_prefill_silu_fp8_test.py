"""Preserve native BF16 rounding, MN-major scales and routed expert outputs."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    deepgemm_hybrid_executor as owner,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepgemm_sm90_scale_layout_test import (
    clone_payload,
    invoke,
    make_executor,
    make_payload,
)
from rtp_llm.models_py.triton_kernels.moe.sm90_prefill_silu_fp8 import (
    silu_mul_native_fp8,
)
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


def make_prefill_payload(rows, routing="uniform"):
    payload = make_payload(rows, "skewed" if routing == "skewed" else "uniform")
    ids = payload.expert_topk_ids
    if routing == "all_local":
        ids.copy_(
            (
                torch.arange(rows, device="cuda")[:, None] * 10
                + torch.arange(10, device="cuda")[None, :]
            )
            % 256
        )
    elif routing == "sparse":
        ids[:, 1:] = -1
    elif routing == "remote":
        ids.fill_(-1)
    counts = torch.bincount(ids[ids >= 0], minlength=256).to(torch.int32)
    payload.expert_tokens_meta.expert_num_tokens.copy_(counts)
    payload.expert_tokens_meta.expert_num_tokens_cpu = counts.cpu().tolist()
    return payload


def reference(x):
    activated = torch.empty((x.shape[0], 640), device=x.device, dtype=torch.bfloat16)
    owner.silu_and_mul(activated, x)
    q, scale = trt_fp8_quantize_128(activated, True)
    return q, scale.t()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DeepGemmSM90PrefillSiluFp8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 required")
        torch.manual_seed(20261116)
        cls.executor = make_executor(graph=False)
        assert cls.executor._sm90_prefill_silu_fp8

    def assert_exact(self, expected, actual):
        self.assertTrue(
            torch.equal(expected[0].view(torch.uint8), actual[0].view(torch.uint8))
        )
        torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)
        self.assertEqual(actual[1].stride(), (1, actual[0].shape[0]))

    def test_native_rounding_and_scales(self):
        for rows in (128, 256, 32768, 65536, 196608, 327680):
            for magnitude in (0.0, 1e-5, 1.0, 100.0):
                with self.subTest(rows=rows, magnitude=magnitude):
                    x = (torch.randn(rows, 1280, device="cuda") * magnitude).to(
                        torch.bfloat16
                    )
                    self.assert_exact(reference(x), silu_mul_native_fp8(x))

    def test_structured_gate_order_and_zero_groups(self):
        x = torch.empty((128, 1280), device="cuda", dtype=torch.bfloat16)
        x[:, :640] = torch.arange(640, device="cuda") % 11 - 5
        x[:, 640:] = (torch.arange(640, device="cuda") % 17 - 8) / 4
        x[:, :128] = 0
        self.assert_exact(reference(x), silu_mul_native_fp8(x))

    def test_graph_replay_reads_changed_values(self):
        x = torch.randn(1024, 1280, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            silu_mul_native_fp8(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = silu_mul_native_fp8(x)
        for magnitude in (0.0, 0.01, 10.0):
            x.copy_(torch.randn_like(x) * magnitude)
            graph.replay()
            self.assert_exact(reference(x), actual)
        graph.reset()

    def test_complete_contiguous_executor(self):
        executor = self.executor
        executor.enable_cuda_graph = False
        executor._sm90_mn_major_scales = True
        for rows in (8192, 32768):
            for routing in ("uniform", "skewed", "all_local", "sparse", "remote"):
                with self.subTest(rows=rows, routing=routing):
                    payload = make_prefill_payload(rows, routing)
                    executor._sm90_prefill_silu_fp8 = False
                    expected = invoke(executor, clone_payload(payload), False)
                    executor._sm90_prefill_silu_fp8 = True
                    actual = invoke(executor, clone_payload(payload), False)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_unqualified_paths_do_not_launch_fusion(self):
        executor = self.executor
        executor._sm90_prefill_silu_fp8 = True
        with patch.object(
            owner,
            "silu_mul_native_fp8",
            side_effect=AssertionError("unexpected fusion"),
        ):
            for masked, rows_list in (
                (True, (16, 32, 64)),
                (False, (2048, 8191, 8193, 32767, 32769)),
            ):
                executor.enable_cuda_graph = masked
                executor._sm90_mn_major_scales = True
                for rows in rows_list:
                    invoke(executor, make_payload(rows), masked)
            executor.enable_cuda_graph = False
            executor._sm90_mn_major_scales = False
            invoke(executor, make_payload(8192), False)

    def test_complete_prefill_graph_changed_routes(self):
        executor = self.executor
        executor.enable_cuda_graph = True
        executor._sm90_mn_major_scales = True
        for rows in (8192, 32768):
            payload = make_prefill_payload(rows)
            inputs = [clone_payload(payload), clone_payload(payload)]
            graphs, outputs = [], []
            for enabled, static in zip((False, True), inputs):
                executor._sm90_prefill_silu_fp8 = enabled
                for _ in range(2):
                    invoke(executor, static, False)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs.append(invoke(executor, static, False))
                graphs.append(graph)
            for routing in ("all_local", "skewed", "sparse", "remote"):
                fresh = make_prefill_payload(rows, routing)
                for static in inputs:
                    for name in (
                        "expert_x",
                        "expert_x_scale",
                        "expert_topk_ids",
                        "expert_topk_weights",
                    ):
                        getattr(static, name).copy_(getattr(fresh, name))
                payload.expert_tokens_meta.expert_num_tokens.copy_(
                    fresh.expert_tokens_meta.expert_num_tokens
                )
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for graph in graphs:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
