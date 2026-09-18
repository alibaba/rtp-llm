import os
import unittest

import torch
import torch.nn.functional as F
from standalone_load import load_component

_math = load_component("rtp_v41_math_test", "models_py/modules/dsv41/math.py")
dequantize_block32 = _math.dequantize_block32
engram_inject = _math.engram_inject
grouped_wo_a = _math.grouped_wo_a
hc_post = _math.hc_post
hc_pre = _math.hc_pre
identity_pre_mix = _math.identity_pre_mix
moe_gate = _math.moe_gate
rms_norm = _math.rms_norm
swiglu_activation = _math.swiglu_activation


def reference_hc_mixes(hidden, weight, scale, base, iterations=20, eps=1e-6):
    hc = hidden.shape[-2]
    flat = hidden.flatten(-2).float()
    mixes = F.linear(flat, weight.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-20
    )
    pre = (mixes[..., :hc] * scale[0] + base[:hc]).sigmoid() + eps
    post = (mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]).sigmoid() * 2
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class MathTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))

    def test_wo_a_preserves_bf16_group_boundaries(self):
        attention = torch.ones(2, 8, 4096, dtype=torch.bfloat16, device=self.device)
        weight = torch.zeros(8, 1024, 4096, dtype=torch.bfloat16, device=self.device)
        weight[:, :, 0] = torch.arange(1, 9, device=self.device)[:, None]
        result = grouped_wo_a(attention, weight)
        expected = (
            torch.arange(1, 9, device=self.device)
            .view(1, 8, 1)
            .expand(2, 8, 1024)
            .bfloat16()
        )
        self.assertTrue(torch.equal(result, expected))
        with self.assertRaises(ValueError):
            grouped_wo_a(attention.float(), weight)

    def test_dense_block32_uses_both_axes(self):
        weight = torch.ones(64, 96, dtype=torch.float8_e4m3fn, device=self.device)
        scale = torch.tensor(
            [[1, 2, 4], [8, 16, 32]], dtype=torch.float32, device=self.device
        )
        decoded = dequantize_block32(weight, scale)
        for row in range(2):
            for col in range(3):
                self.assertTrue(
                    torch.all(
                        decoded[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
                        == scale[row, col]
                    ).item()
                )

    def test_swiglu_negative_gate_and_route_before_cast(self):
        gate = torch.tensor(
            [[-40, -12, -1.125, 0, 4.25, 11]], device=self.device
        ).bfloat16()
        up = torch.tensor(
            [[-40, 12, -2.25, 1, 1.25, 40]], device=self.device
        ).bfloat16()
        route = torch.tensor([0.3125], device=self.device)
        expected = (
            F.silu(gate.float().clamp(max=10))
            * up.float().clamp(-10, 10)
            * route[:, None]
        ).bfloat16()
        actual = swiglu_activation(gate, up, route)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual[0, 3].item(), 0)
        self.assertGreater(actual[0, 0].item(), 0)

    def test_hc_source_destination_and_one_hot(self):
        hidden = torch.arange(4 * 5120, device=self.device).view(1, 4, 5120).bfloat16()
        self.assertTrue(
            torch.equal(hc_pre(hidden, identity_pre_mix(hidden)), hidden[:, 0])
        )
        comb = torch.zeros(1, 4, 4, device=self.device)
        comb[:, 0, 3] = 1
        result = hc_post(
            torch.zeros(1, 5120, device=self.device).bfloat16(),
            hidden,
            torch.zeros(1, 4, device=self.device),
            comb,
        )
        self.assertTrue(torch.equal(result[:, 3], hidden[:, 0]))
        self.assertTrue(torch.all(result[:, :3] == 0).item())

    @torch.inference_mode()
    def test_hc_mixes_preserves_fp32_formula(self):
        if self.device.type != "cuda":
            self.skipTest("TileLang split/Sinkhorn requires CUDA")
        torch.manual_seed(83)
        for shape, iterations, eps in (
            ((1, 4, 5120), 20, 1e-6),
            ((1, 31, 4, 5120), 20, 1e-6),
            ((2, 3, 4, 5120), 1, 1e-6),
            ((7, 4, 5120), 3, 1e-4),
        ):
            with self.subTest(shape=shape, iterations=iterations, eps=eps):
                hidden = torch.randn(shape, device=self.device).bfloat16()
                weight = torch.randn(24, 4 * 5120, device=self.device) * 0.02
                scale = torch.tensor([0.13, 0.27, 0.31], device=self.device)
                base = torch.randn(24, device=self.device)
                actual = _math.hc_mixes(
                    hidden, weight, scale, base, iterations, hc_eps=eps
                )
                expected = reference_hc_mixes(
                    hidden, weight, scale, base, iterations, eps
                )
                for left, right in zip(actual, expected):
                    self.assertEqual(left.dtype, torch.float32)
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_hc_mixes_graph_uses_updated_inputs(self):
        if self.device.type != "cuda":
            self.skipTest("CUDA Graph requires CUDA")
        hidden = torch.randn(7, 4, 5120, device=self.device).bfloat16()
        weight = torch.randn(24, 4 * 5120, device=self.device) * 0.02
        scale = torch.tensor([0.13, 0.27, 0.31], device=self.device)
        base = torch.randn(24, device=self.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _math.hc_mixes(hidden, weight, scale, base)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = _math.hc_mixes(hidden, weight, scale, base)
        pointers = [value.data_ptr() for value in actual]
        for step in range(3):
            hidden.normal_()
            weight.mul_(-1)
            scale.fill_(0.5 + step)
            base.fill_(step - 1)
            graph.replay()
            expected = reference_hc_mixes(hidden, weight, scale, base)
            self.assertEqual([value.data_ptr() for value in actual], pointers)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)

    def test_hc_mixes_preserves_gradient_fallback(self):
        hidden = torch.randn(3, 4, 17, device=self.device, requires_grad=True)
        weight = torch.randn(24, 4 * 17, device=self.device)
        scale = torch.ones(3, device=self.device)
        base = torch.zeros(24, device=self.device)
        actual = _math.hc_mixes(hidden, weight, scale, base)
        expected = reference_hc_mixes(hidden, weight, scale, base)
        for left, right in zip(actual, expected):
            self.assertTrue(left.requires_grad)
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_image_bias_selects_but_does_not_weight(self):
        hidden = torch.zeros(2, 5120, device=self.device).bfloat16()
        gate = torch.zeros(384, 5120, device=self.device).bfloat16()
        bias = torch.arange(384, device=self.device).float()
        weights, indices = moe_gate(
            hidden,
            gate,
            bias,
            -bias,
            torch.tensor([False, True], device=self.device),
            6,
        )
        self.assertEqual(indices[0].tolist(), [383, 382, 381, 380, 379, 378])
        self.assertEqual(indices[1].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertTrue(torch.equal(weights[0], weights[1]))

    def test_image_mask_suppresses_engram_injection(self):
        hidden = torch.ones(2, 4, 5120, device=self.device).bfloat16()
        projected = torch.ones(2, 5 * 5120, device=self.device).bfloat16()
        qk = torch.ones(4, 5120, device=self.device).bfloat16()
        result = engram_inject(
            hidden, projected, qk, qk, torch.tensor([False, True], device=self.device)
        )
        self.assertTrue(torch.equal(result[0], hidden[0]))
        self.assertTrue(torch.all(result[1] > hidden[1]).item())

    @torch.inference_mode()
    def test_rms_norm_native_matches_reference(self):
        if self.device.type != "cuda" or torch.cuda.get_device_capability(0)[0] != 10:
            self.skipTest("native rmsnorm requires a Blackwell CUDA device")
        shapes = [(2, 5120), (7, 1280), (2048, 512), (4, 512, 5120), (3, 128), (1, 5120)]
        for index, shape in enumerate(shapes):
            torch.manual_seed(83 + index)
            hidden = torch.randn(*shape, device=self.device).bfloat16()
            weight = (torch.randn(shape[-1], device=self.device) * 0.5 + 1).bfloat16()
            expected = (
                hidden.float()
                * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-20)
                * weight.float()
            ).to(hidden.dtype)
            native = rms_norm(hidden, weight)
            ai = native.view(torch.int16).to(torch.int32)
            bi = expected.view(torch.int16).to(torch.int32)
            ulp = (
                torch.where(ai >= 0, ai, -32768 - ai)
                - torch.where(bi >= 0, bi, -32768 - bi)
            ).abs()
            exact = (native == expected).float().mean().item()
            self.assertGreaterEqual(exact, 0.999, (shape, exact))
            self.assertLessEqual(int(ulp.max().item()), 1, (shape, int(ulp.max())))

    @torch.inference_mode()
    def test_rms_norm_native_default_on(self):
        # R4-3: the native kernel is the code-default path on Blackwell CUDA13.
        if self.device.type != "cuda" or torch.cuda.get_device_capability(0)[0] != 10:
            self.skipTest("native rmsnorm requires a Blackwell CUDA device")
        torch.manual_seed(91)
        hidden = torch.randn(2048, 512, device=self.device).bfloat16()
        weight = (torch.randn(512, device=self.device) * 0.5 + 1).bfloat16()
        self.assertTrue(_math._rms_norm_native_supported(hidden, weight))

    @torch.inference_mode()
    def test_rms_norm_native_fallbacks(self):
        hidden = torch.randn(4, 5120, device=self.device).bfloat16()
        weight = torch.randn(5120, device=self.device).bfloat16()
        if self.device.type == "cuda" and torch.cuda.get_device_capability(0)[0] == 10:
            self.assertTrue(_math._rms_norm_native_supported(hidden, weight))
            self.assertFalse(
                _math._rms_norm_native_supported(hidden, weight.float())
            )
            self.assertFalse(
                _math._rms_norm_native_supported(hidden.float(), weight)
            )
            wide = torch.randn(4, 10240, device=self.device).bfloat16()
            self.assertFalse(
                _math._rms_norm_native_supported(wide[:, :5120], weight)
            )
            self.assertFalse(
                _math._rms_norm_native_supported(hidden, weight.cpu())
            )
        mixed = rms_norm(hidden, weight.float())
        expected = (
            hidden.float()
            * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-20)
            * weight.float()
        ).to(hidden.dtype)
        self.assertTrue(torch.equal(mixed, expected))


if __name__ == "__main__":
    unittest.main()
