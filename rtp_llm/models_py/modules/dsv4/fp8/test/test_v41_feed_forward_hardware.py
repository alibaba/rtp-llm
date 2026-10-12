"""SM120 FP32 routing, complete FFN numerics and dynamic decode graph replay."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_expert_hardware as shared_fixture,
)
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_routed_hardware as routed_fixture,
)
from rtp_llm.models_py.modules.dsv4.v41_expert import V41SharedExpert
from rtp_llm.models_py.modules.dsv4.v41_feed_forward import V41FeedForward, V41Router
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


class V41FeedForwardHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        previous = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(4145)

    def test_router_full_geometry_matches_fp32_reference_and_rejects_tf32(self):
        matrix = (torch.randn(384, 5120, device="cuda") / 5120**0.5).bfloat16()
        bias = torch.randn(384, device="cuda") * 0.1
        op = V41Router(matrix, bias, gate_temp=0.7)
        x = torch.randn(17, 5120, dtype=torch.bfloat16, device="cuda")
        valid = torch.arange(17, device="cuda") % 3 != 1
        x[~valid] = float("nan")
        scores = F.softplus(
            torch.where(valid.cpu()[:, None], x.cpu(), 0).float()
            @ matrix.cpu().float().T
            / 0.7
        ).sqrt()
        expected_ids = (scores + bias.cpu()).topk(6, dim=-1).indices
        expected = scores.gather(1, expected_ids)
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * 1.5
        weights, ids = op(x, valid)
        torch.testing.assert_close(
            ids.cpu()[valid.cpu()], expected_ids[valid.cpu()].int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            weights.cpu()[valid.cpu()], expected[valid.cpu()], rtol=1e-6, atol=1e-7
        )
        self.assertTrue(bool((weights[~valid] == 0).all()))
        self.assertTrue(bool((ids[~valid] == -1).all()))
        torch.backends.cuda.matmul.allow_tf32 = True
        with self.assertRaisesRegex(RuntimeError, "TF32"):
            op(x, valid)

    def test_complete_ffn_masked_contiguous_and_graph_updates(self):
        shared_weights = shared_fixture.V41ExpertHardwareTest.weights()
        routed_weights = routed_fixture.V41RoutedHardwareTest.weights(8)
        matrix = (torch.randn(8, 5120, device="cuda") / 5120**0.5).bfloat16()
        bias = torch.randn(8, device="cuda") * 0.1
        router = V41Router(matrix, bias, max_rows=8)
        shared = V41SharedExpert(shared_weights)
        local = V41LocalMoE(
            V41RoutedExperts(routed_weights), max_tokens=8, masked_token_limit=4
        )
        op = V41FeedForward(14, router, shared, local)
        cpu_router = V41Router(matrix.cpu(), bias.cpu(), max_rows=8)
        cpu_shared = V41SharedExpert(
            {k: v.cpu() for k, v in shared_weights.items()}, backend="eager"
        )
        cpu_local = V41LocalMoE(
            V41RoutedExperts(
                {k: v.cpu() for k, v in routed_weights.items()}, backend="eager"
            ),
            max_tokens=8,
            masked_token_limit=4,
        )

        def check(actual, x, valid):
            weights, ids = cpu_router(x.cpu(), valid.cpu())
            contributions = cpu_local(x.cpu(), weights, ids, valid_rows=valid.cpu())
            shared_result = cpu_shared(x.cpu(), valid.cpu()).float()
            expected = (contributions + shared_result).bfloat16()
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            # Independent CPU GEMMs can cross intermediate BF16/FP8 boundaries.
            torch.testing.assert_close(actual.cpu(), expected, rtol=3e-2, atol=0.12)
            self.assertTrue(bool((actual[~valid] == 0).all()))

        for rows in (0, 3, 5):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            valid = torch.arange(rows, device="cuda") % 3 != 1
            x[~valid] = float("nan")
            check(op(x, valid), x, valid)
        x = torch.randn(3, 5120, device="cuda", dtype=torch.bfloat16)
        valid = torch.ones(3, device="cuda", dtype=torch.bool)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op(x, valid)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op(x, valid)
        for live in (False, True):
            valid.fill_(live)
            x.normal_()
            if not live:
                x.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(actual, op(x, valid), rtol=0, atol=0)
            check(actual, x, valid)


if __name__ == "__main__":
    unittest.main()
