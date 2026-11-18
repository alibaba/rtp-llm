"""Guard coverage plus bounded ROCm numerical/capture checks (not a load test)."""

import types
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc import rocm_decode_qwen3 as decode


class DecodeFusionGuardTest(unittest.TestCase):
    def test_prefill_capture_and_non_graph_decode_do_not_enter_fast_path(self):
        cases = [
            (True, False, False),
            (True, True, False),
            (False, False, False),
            (False, True, True),
        ]
        with patch.object(torch.version, "hip", "7.2"), torch.inference_mode():
            for prefill, graph, capture in cases:
                inputs = types.SimpleNamespace(
                    attention_inputs=types.SimpleNamespace(
                        is_prefill=prefill, is_cuda_graph=graph
                    ),
                    capture_hidden_states=capture,
                )
                self.assertIsNone(decode.forward_decode(object(), inputs))
            self.assertIsNone(
                decode.forward_decode(
                    object(), types.SimpleNamespace(attention_inputs={})
                )
            )

    def test_module_and_global_hooks_are_not_bypassed(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.assertFalse(decode._has_hooks(model))
        with model[0].register_forward_pre_hook(lambda module, args: None):
            self.assertTrue(decode._has_hooks(model))
        self.assertFalse(decode._has_hooks(model))
        with torch.nn.modules.module.register_module_forward_hook(
            lambda module, args, output: None
        ):
            self.assertTrue(decode._has_hooks(model))

    def test_fallback_reasons_are_deduplicated(self):
        model = types.SimpleNamespace(_rocm_decode_fallback_reasons=set())
        with patch.object(decode.logging, "info") as log:
            for _ in range(3):
                self.assertIsNone(decode._fallback(model, "module_hooks"))
            self.assertEqual(log.call_count, 1)


@unittest.skipUnless(torch.version.hip, "requires the compiled ROCm runtime")
class DecodeFusionRocmTest(unittest.TestCase):
    @torch.inference_mode()
    def test_bf16_residual_rounding_and_native_norm_parity(self):
        from aiter.ops.rmsnorm import rmsnorm

        from rtp_llm.models_py.model_desc.rocm_rounded_add_norm import rounded_add_norm

        with torch.random.fork_rng(devices=[0]):
            for rows in (1, 2, 4, 8):
                for seed in range(8):
                    torch.manual_seed(seed)
                    x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                    residual = torch.randn_like(x)
                    weight = torch.randn(2560, device="cuda", dtype=x.dtype)
                    expected_residual = x + residual
                    expected = torch.empty_like(x)
                    rmsnorm(expected, expected_residual, weight, 1e-6)
                    actual, added = rounded_add_norm(x, residual, weight, 1e-6)
                    self.assertTrue(torch.equal(added, expected_residual))
                    self.assertTrue(
                        torch.allclose(actual, expected, rtol=0.008, atol=0)
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured, captured_residual = rounded_add_norm(
                        x, residual, weight, 1e-6
                    )
                x.add_(0.125)
                graph.replay()
                expected_residual = x + residual
                rmsnorm(expected, expected_residual, weight, 1e-6)
                self.assertTrue(torch.equal(captured_residual, expected_residual))
                self.assertTrue(torch.allclose(captured, expected, rtol=0.008, atol=0))

    @torch.inference_mode()
    def test_rounding_probe_accepts_native_and_rejects_rne_reference(self):
        from rtp_llm.models_py.modules.base.rocm.norm import RMSNorm

        weight = torch.ones(2560, device="cuda", dtype=torch.bfloat16)
        model = types.SimpleNamespace(_rocm_decode_rounding_state=None)
        self.assertTrue(
            decode._rounding_supported(model, RMSNorm(weight), weight.device)
        )

        class RoundToNearestNorm:
            def __init__(self, weight, eps=1e-6):
                self.weight = weight
                self.variance_epsilon = eps

            def __call__(self, x):
                v = x.float()
                y = v * torch.rsqrt(
                    v.square().mean(-1, keepdim=True) + self.variance_epsilon
                )
                return (y * self.weight.float()).to(x.dtype)

        model._rocm_decode_rounding_state = None
        self.assertFalse(
            decode._rounding_supported(model, RoundToNearestNorm(weight), weight.device)
        )


if __name__ == "__main__":
    unittest.main()
