import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import Qwen4GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.hc_prefill_mix import hc_up_mix


class HcPrefillMixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")

    def setUp(self):
        torch.manual_seed(2035)

    @staticmethod
    def random(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.03

    @staticmethod
    def consumer(logits, normed):
        products = (torch.sigmoid(logits) * normed).unflatten(-1, (4, 2560))
        return products.float().sum(-2).div(4).bfloat16()

    def test_fp8_large_prefill_retains_quantizer_and_consumer_boundaries(self):
        weight, scale = scaled_fp8_per_tensor_quant(self.random(10240, 320))
        for rows in (8192, 16384):
            mix = self.random(rows, 320)
            normed = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            quantized, input_scale = scaled_fp8_per_tensor_quant(mix)
            logits = torch._scaled_mm(
                quantized,
                weight.T,
                scale_a=input_scale,
                scale_b=scale,
                out_dtype=torch.bfloat16,
            )
            actual = hc_up_mix(mix, weight, normed, scale)
            torch.testing.assert_close(
                actual, self.consumer(logits, normed), rtol=0, atol=0
            )

    def test_graph_replay_refreshes_fp8_quantization_and_normed_consumer(self):
        rows = 8192
        mix = self.random(rows, 320)
        normed = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
        weight, scale = scaled_fp8_per_tensor_quant(self.random(10240, 320))
        hc_up_mix(mix, weight, normed, scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = hc_up_mix(mix, weight, normed, scale)
        try:
            for gain in (0.1, 1.0, 7.0):
                mix.copy_(self.random(rows, 320) * gain)
                normed.copy_(torch.randn_like(normed))
                quantized, input_scale = scaled_fp8_per_tensor_quant(mix)
                logits = torch._scaled_mm(
                    quantized,
                    weight.T,
                    scale_a=input_scale,
                    scale_b=scale,
                    out_dtype=torch.bfloat16,
                )
                graph.replay()
                torch.testing.assert_close(
                    output, self.consumer(logits, normed), rtol=0, atol=0
                )
        finally:
            graph.reset()

    def test_model_fp8_composition_is_exact_and_decode_falls_back(self):
        gamma = self.random(10240)
        down, sd = scaled_fp8_per_tensor_quant(self.random(320, 10240))
        up, su = scaled_fp8_per_tensor_quant(self.random(10240, 320))
        inject, si = scaled_fp8_per_tensor_quant(self.random(4, 10240))
        weights = (gamma, down, up, inject)
        kwargs = dict(
            hc_mult=4, norm_eps=1e-6, weight_scales=(sd, su, si), role="PREFILL"
        )
        baseline = GatedResidual(*weights, **kwargs)
        with patch.dict("os.environ", RTP_QWEN4_PREFILL_HC_IMPL="upmix"):
            candidate = Qwen4GatedResidual(*weights, **kwargs)
        x = self.random(8192, 10240)
        expected = baseline(x)
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.hc_prefill_mix.hc_up_mix",
            wraps=hc_up_mix,
        ) as fused:
            actual = candidate(x)
            self.assertEqual(fused.call_count, 1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with patch.dict("os.environ", RTP_QWEN4_PREFILL_HC_IMPL="upmix"):
            candidate = Qwen4GatedResidual(*weights, **dict(kwargs, role="DECODE"))
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.hc_prefill_mix.hc_up_mix",
            side_effect=AssertionError("Prefill HC used by Decode"),
        ):
            candidate(self.random(7, 10240))

    def test_bf16_upmix_retains_independent_down_and_injection_projections(self):
        # The rejected source20 merged-down option changed cuBLAS rounding.
        # This path keeps both original projections; no merged-down mode exists.
        for seed in (2035, 1779):
            torch.manual_seed(seed)
            weights = (
                self.random(10240),
                self.random(320, 10240),
                self.random(10240, 320),
                self.random(4, 10240),
            )
            kwargs = dict(hc_mult=4, norm_eps=1e-6, role="PREFILL")
            baseline = GatedResidual(*weights, **kwargs)
            with patch.dict("os.environ", RTP_QWEN4_PREFILL_HC_IMPL="upmix"):
                candidate = Qwen4GatedResidual(*weights, **kwargs)
            for rows in (4093, 8192, 16384):
                x = self.random(rows, 10240)
                expected = baseline(x)
                with patch(
                    "rtp_llm.models_py.modules.qwen4_exp.hc_prefill_mix.hc_up_mix",
                    wraps=hc_up_mix,
                ) as fused:
                    actual = candidate(x)
                    self.assertEqual(fused.call_count, 1)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            with patch.dict("os.environ", RTP_QWEN4_PREFILL_HC_IMPL="upmix_merged"):
                with self.assertRaises(ValueError):
                    Qwen4GatedResidual(*weights, **kwargs)


if __name__ == "__main__":
    unittest.main()
