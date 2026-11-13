"""Independent routing/quantization and changing-input graph regression tests."""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    per_token_cast_to_fp8_packed_ue8m0,
)
from rtp_llm.models_py.triton_kernels.moe.softmax_gate_pack import (
    fused_pack_mega_moe_softmax_inputs,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SoftmaxGatePackTest(unittest.TestCase):
    def _buffers(self, tokens, dim=8192, experts=768, topk=10):
        torch.manual_seed(171)
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
        outputs = (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty(tokens, dim // 128, device="cuda", dtype=torch.int32),
            torch.empty(tokens, topk, device="cuda", dtype=torch.int64),
            torch.empty(tokens, topk, device="cuda", dtype=torch.float32),
        )
        return x, logits, outputs

    def _run(self, x, logits, outputs):
        fused_pack_mega_moe_softmax_inputs(x, logits, *outputs, topk=outputs[2].size(1))

    def _check(self, x, logits, outputs):
        data, scales, indices, weights = outputs
        probability = torch.softmax(logits.float(), dim=-1)
        expected_ids = torch.argsort(probability, dim=-1, descending=True, stable=True)[
            :, : indices.size(1)
        ]
        expected_weights = probability.gather(-1, expected_ids)
        expected_weights /= expected_weights.sum(-1, keepdim=True)
        torch.testing.assert_close(indices, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)
        expected_data, expected_scales = per_token_cast_to_fp8_packed_ue8m0(
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), gran_k=32
        )
        torch.testing.assert_close(
            data.view(torch.uint8), expected_data.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)

    def test_m11_and_general_expert_counts(self):
        for tokens, dim, experts, topk in (
            (4, 8192, 768, 10),
            (129, 8192, 768, 10),
            (33, 128, 512, 10),
            (17, 256, 37, 7),
            (32768, 128, 768, 10),
        ):
            with self.subTest(tokens=tokens, experts=experts):
                x, logits, outputs = self._buffers(tokens, dim, experts, topk)
                self._run(x, logits, outputs)
                self._check(x, logits, outputs)

    def test_ties_zero_and_empty(self):
        for tokens in (0, 4):
            x, logits, outputs = self._buffers(tokens)
            x.zero_()
            logits.zero_()
            self._run(x, logits, outputs)
            self._check(x, logits, outputs)

    def test_invalid_logits_remain_invalid(self):
        x, logits, outputs = self._buffers(4)
        logits[0, 0] = float("nan")
        logits[1, 5] = float("inf")
        logits[2].fill_(-float("inf"))
        self._run(x, logits, outputs)
        self.assertTrue(torch.isnan(outputs[3][:3]).all().item())
        self.assertTrue(torch.isfinite(outputs[3][3]).all().item())

    def test_matches_native_m11_routing(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models_py.modules.base.cuda.select_topk import SelectTopk

        config = ModelConfig()
        config.expert_num = 768
        config.moe_k = 10
        config.has_moe_norm = True
        native = SelectTopk(config)
        for tokens in (4, 129, 32768):
            x, logits, outputs = self._buffers(tokens, dim=128)
            logits[0].zero_()
            logits[1].fill_(-80.0)
            logits[1, 30:42] = 80.0
            self._run(x, logits, outputs)
            ids = torch.empty_like(outputs[2])
            weights = torch.empty_like(outputs[3])
            native(logits, ids, weights)
            torch.testing.assert_close(outputs[2], ids, rtol=0, atol=0)
            torch.testing.assert_close(outputs[3], weights, rtol=2e-6, atol=2e-7)

    def test_graph_replay_updates_all_inputs(self):
        x, logits, outputs = self._buffers(4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._run(x, logits, outputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run(x, logits, outputs)
        for offset in (0.25, 3.0, -2.0):
            x.copy_(torch.randn_like(x) + offset)
            logits.copy_(torch.randn_like(logits))
            logits[:, 17:27] += 6.0
            graph.replay()
            self._check(x, logits, outputs)


if __name__ == "__main__":
    unittest.main()
