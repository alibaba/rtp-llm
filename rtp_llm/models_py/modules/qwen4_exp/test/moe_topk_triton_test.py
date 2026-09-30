import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.base.cuda.select_topk import SelectTopk
from rtp_llm.models_py.modules.qwen4_exp.moe_topk_triton import Qwen4ExpMoeTopk


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MoeTopkTritonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(461)
        config = ModelConfig()
        config.expert_num = 512
        config.moe_k = 10
        config.has_moe_norm = True
        self.native = SelectTopk(config)
        self.fused = Qwen4ExpMoeTopk(self.native)

    def compare(self, logits, ids, weights):
        expected_ids = torch.empty_like(ids)
        expected_weights = torch.empty_like(weights)
        self.native(logits, expected_ids, expected_weights)
        torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(weights, expected_weights, atol=2e-7, rtol=2e-6)

    def test_native_parity_for_ties_and_small_and_large_batches(self):
        for rows in (1, 7, 17, 32, 127, 4093):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(rows=rows, dtype=dtype):
                    logits = torch.randn(rows, 512, device="cuda", dtype=dtype)
                    logits[0] = 0
                    ids = torch.empty(rows, 10, device="cuda", dtype=torch.int32)
                    weights = torch.empty(rows, 10, device="cuda")
                    self.fused(logits, ids, weights)
                    self.compare(logits, ids, weights)

    def test_graph_replay_refreshes_strided_logits(self):
        logits = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)[:, :512]
        ids = torch.empty(17, 10, device="cuda", dtype=torch.int32)
        weights = torch.empty(17, 10, device="cuda")
        self.fused(logits, ids, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.fused(logits, ids, weights)
        for _ in range(3):
            logits.copy_(torch.randn_like(logits))
            logits[0] = 0
            graph.replay()
            self.compare(logits, ids, weights)

    def test_nonfinite_logits_preserve_native_routing_and_bounded_ids(self):
        logits = torch.randn(7, 512, device="cuda")
        logits[0] = float("nan")
        logits[1, 0] = float("nan")
        logits[2, 0] = float("inf")
        logits[3] = float("-inf")
        ids = torch.empty(7, 10, device="cuda", dtype=torch.int32)
        weights = torch.empty(7, 10, device="cuda")
        self.fused(logits, ids, weights)
        expected_ids = torch.empty_like(ids)
        expected_weights = torch.empty_like(weights)
        self.native(logits, expected_ids, expected_weights)
        torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(
            weights, expected_weights, atol=2e-7, rtol=2e-6, equal_nan=True
        )
        self.assertTrue(((ids >= 0) & (ids < 512)).all())

    def test_unsupported_geometry_delegates_without_touching_outputs(self):
        class Native(nn.Module):
            config = SimpleNamespace(expert_num=256, moe_k=10, has_moe_norm=True)

            def forward(self, logits, ids, weights):
                self.called = True

        native = Native()
        fused = Qwen4ExpMoeTopk(native)
        logits = torch.randn(7, 256, device="cuda")
        ids = torch.full((7, 10), -99, dtype=torch.int32, device="cuda")
        weights = torch.full((7, 10), -99.0, device="cuda")
        fused(logits, ids, weights)
        self.assertTrue(native.called)
        self.assertTrue((ids == -99).all())
        self.assertTrue((weights == -99).all())

    def test_unsupported_device_and_output_layout_delegate(self):
        logits = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
        ids = torch.empty(7, 10, device="cuda", dtype=torch.int32)
        weights = torch.empty(7, 10, device="cuda")
        for target, value in (
            ("torch.version.hip", "hip"),
            ("torch.cuda.get_device_capability", (7, 0)),
        ):
            with self.subTest(target=target):
                override = (
                    patch(target, return_value=value)
                    if target.endswith("get_device_capability")
                    else patch(target, value)
                )
                with (
                    override,
                    patch.object(
                        self.native, "forward", wraps=self.native.forward
                    ) as call,
                ):
                    self.fused(logits, ids, weights)
                    call.assert_called_once()
                self.compare(logits, ids, weights)
        with patch.object(self.native, "forward") as call:
            self.fused(logits, ids.long(), weights)
            call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
