"""SM120 numerical boundaries required by the released V4.1 reference."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import roundtrip_swa
from rtp_llm.models_py.modules.dsv4.moe.gate import Gate
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb
from rtp_llm.utils.model_weight import W


def norm_rope_reference(x, weight, frequencies, inverse=False):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    if weight is not None:
        value = value * weight.float()
    value = value.to(x.dtype)
    apply_rotary_emb(value[..., -64:].unsqueeze(0), frequencies, inverse=inverse)
    return value


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class V41ReferenceMathSM120Test(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("requires SM120")
        torch.manual_seed(913)

    @torch.no_grad()
    def test_fp32_router_keeps_near_tie_expert_choice(self):
        x = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
        x[0, :2] = torch.tensor([1, 0.125], device="cuda")
        weight = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
        weight[:2, 0] = 1
        weight[1, 1] = 1 / 256
        bias = torch.zeros(256, device="cuda")
        bias[0] = 0.00001
        layers = {W.v4_router_w: weight, W.v4_router_bias: bias}
        bf16 = Gate(0, 256, 256, 1, layer_weights=layers)
        fp32 = Gate(0, 256, 256, 1, layer_weights=layers, fp32_logits=True)
        expected = (
            torch.nn.functional.softplus(x.float() @ weight.float().T).sqrt() + bias
        ).argmax(-1)
        self.assertEqual(int(expected[0]), 1)
        with patch.dict(os.environ, {"DSV4_GATE_FP32": "0"}):
            self.assertEqual(int(bf16(x)[1][0, 0]), 0)
            for fused in ("0", "1"):
                with patch.dict(os.environ, {"DSV4_GATE_FUSED": fused}):
                    weights, indices = fp32(x)
                    torch.testing.assert_close(indices[:, 0], expected, rtol=0, atol=0)
                    torch.testing.assert_close(weights, torch.ones_like(weights))
            empty_weights, empty_indices = fp32(x[:0])
            self.assertEqual(empty_weights.shape, (0, 1))
            self.assertEqual(empty_indices.shape, (0, 1))

    @torch.no_grad()
    def test_fresh_swa_matches_reference_group32_rounding(self):
        for rows in (0, 1, 137, 503):
            values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
            if rows:
                values[0, :32] = 0
                values[:, 32:64] *= 0.001
                values[:, 64:96] *= 32
            saved = values.clone()
            groups = values.float().reshape(rows, 16, 32)
            scales = (groups.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
            expected = (
                (
                    (groups / scales[..., None]).to(torch.float8_e4m3fn).float()
                    * scales[..., None]
                )
                .reshape(values.shape)
                .to(values.dtype)
            )
            actual = roundtrip_swa(values)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(values, saved, rtol=0, atol=0)
            torch.testing.assert_close(roundtrip_swa(actual), actual, rtol=0, atol=0)

    @torch.no_grad()
    def test_norm_rounding_matches_separate_reference(self):
        for heads in (1, 4):
            x = torch.randn(97, heads, 512, device="cuda", dtype=torch.bfloat16)
            angles = torch.randn(97, 32, device="cuda")
            frequencies = torch.polar(torch.ones_like(angles), angles)
            for weighted in (False, True):
                weight = (
                    torch.randn(512, device="cuda", dtype=torch.bfloat16)
                    if weighted
                    else None
                )
                for inverse in (False, True):
                    expected = norm_rope_reference(x, weight, frequencies, inverse)
                    original = fused_rmsnorm_rope(
                        x, weight, frequencies, 64, inverse=inverse, group_heads=heads
                    )
                    actual = fused_rmsnorm_rope(
                        x,
                        weight,
                        frequencies,
                        64,
                        inverse=inverse,
                        group_heads=heads,
                        round_norm_to_input_dtype=True,
                    )
                    torch.testing.assert_close(
                        actual[..., :-64], original[..., :-64], rtol=0, atol=0
                    )
                    error = (actual.float() - expected.float()).norm()
                    old_error = (original.float() - expected.float()).norm()
                    self.assertLess(float(error), float(old_error) / 10)
                    self.assertLess(float(error / expected.float().norm()), 0.0003)
                    output = torch.empty_like(x)
                    result = fused_rmsnorm_rope(
                        x,
                        weight,
                        frequencies,
                        64,
                        inverse=inverse,
                        group_heads=heads,
                        out=output,
                        round_norm_to_input_dtype=True,
                    )
                    self.assertEqual(result.data_ptr(), output.data_ptr())
                    torch.testing.assert_close(result, actual, rtol=0, atol=0)

    @torch.no_grad()
    def test_rounded_inplace_and_changed_input_graph(self):
        x = torch.randn(17, 1, 512, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(17, 32, device="cuda")
        frequencies = torch.polar(torch.ones_like(angles), angles)

        def run(values, **kwargs):
            return fused_rmsnorm_rope(
                values,
                weight,
                frequencies,
                64,
                round_norm_to_input_dtype=True,
                **kwargs,
            )

        expected = run(x)
        inplace = x.clone()
        result = run(inplace, inplace=True)
        self.assertEqual(result.data_ptr(), inplace.data_ptr())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        output = torch.empty_like(x)
        run(x, out=output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(x, out=output)
        x.normal_()
        expected = run(x)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
