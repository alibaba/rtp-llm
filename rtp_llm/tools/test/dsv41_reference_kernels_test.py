"""Opt-in SM120 numerical checks of the unmodified released TileLang kernels."""

import importlib
import os
import sys
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(
    os.environ.get("DSV41_TEST_CHECKPOINT"), "needs official NAS source"
)
class OfficialKernelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Reference qualification requires SM120")
        root = Path(os.environ["DSV41_TEST_CHECKPOINT"]) / "inference"
        sys.path.insert(0, str(root))
        cls.kernel = importlib.import_module("kernel")
        if Path(cls.kernel.__file__).resolve() != (root / "kernel.py").resolve():
            raise RuntimeError("Unexpected reference kernel module")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(1941)

    def test_fp8_activation(self):
        x = torch.randn(7, 512, device="cuda")
        x[0].zero_()
        quant, scales = self.kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        groups = x.cpu().float().reshape(7, 16, 32)
        expected_scales = (
            (groups.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
        )
        expected = (
            (groups / expected_scales[..., None]).to(torch.float8_e4m3fn).reshape_as(x)
        )
        torch.testing.assert_close(
            scales.cpu().float(), expected_scales, rtol=0, atol=0
        )
        self.assertTrue(
            torch.equal(quant.cpu().view(torch.uint8), expected.view(torch.uint8))
        )
        inplace = self.kernel.act_quant(
            x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True
        )
        decoded = (
            (expected.float().reshape_as(groups) * expected_scales[..., None])
            .reshape_as(x)
            .bfloat16()
        )
        torch.testing.assert_close(inplace.cpu(), decoded, rtol=0, atol=0)

    def test_fp4_cache_quantization(self):
        x = torch.randn(7, 512, device="cuda")
        x[0].zero_()
        actual = self.kernel.fp4_act_quant(x.clone(), 16, True, torch.float8_e4m3fn)
        groups = x.cpu().float().reshape(7, 32, 16)
        scales = (
            (groups.abs().amax(-1).clamp_min(6 * 2**-9) / 6)
            .to(torch.float8_e4m3fn)
            .float()
        )
        normalized = (groups / scales[..., None]).clamp(-6, 6)
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        distances = (normalized.abs()[..., None] - levels).abs()
        # FP4 rounds midpoint ties to the even encoded value.
        even_first = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        closest = even_first[distances[..., even_first].argmin(-1)]
        expected = (
            (levels[closest].copysign(normalized) * scales[..., None])
            .reshape_as(x)
            .bfloat16()
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_fp8_and_fp4_gemm(self):
        levels = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
        )
        for rows, output_dim, input_dim in (
            (1, 128, 256),
            (7, 128, 5120),
            (1, 5120, 2304),
        ):
            x = torch.randn(rows, input_dim, device="cuda")
            quant, scales = self.kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
            decoded_x = quant.cpu().float() * scales.cpu().float().repeat_interleave(
                32, -1
            )
            weight = torch.randn(output_dim, input_dim, device="cuda").to(
                torch.float8_e4m3fn
            )
            weight_scales = (
                torch.randint(-2, 3, (output_dim // 32, input_dim // 32), device="cuda")
                .float()
                .exp2()
                .to(torch.float8_e8m0fnu)
            )
            actual = self.kernel.fp8_gemm(
                quant, scales, weight, weight_scales, torch.float8_e8m0fnu, 32
            )
            decoded_w = (
                weight.cpu().float()
                * weight_scales.cpu()
                .float()
                .repeat_interleave(32, 0)
                .repeat_interleave(32, 1)
            )
            torch.testing.assert_close(
                actual.cpu().float(), decoded_x @ decoded_w.T, rtol=0.008, atol=0.03
            )
            packed = torch.randint(
                0, 256, (output_dim, input_dim // 2), dtype=torch.uint8
            )
            weights_fp4 = packed.to("cuda").view(torch.float4_e2m1fn_x2)
            weights_scale = (
                torch.randint(-2, 3, (output_dim, input_dim // 32), device="cuda")
                .float()
                .exp2()
                .to(torch.float8_e8m0fnu)
            )
            actual = self.kernel.fp4_gemm(
                quant, scales, weights_fp4, weights_scale, torch.float8_e8m0fnu, 32
            )
            decoded_w = torch.stack(
                (levels[(packed & 15).long()], levels[(packed >> 4).long()]), -1
            ).flatten(1)
            decoded_w *= weights_scale.cpu().float().repeat_interleave(32, -1)
            torch.testing.assert_close(
                actual.cpu().float(), decoded_x @ decoded_w.T, rtol=0.008, atol=0.05
            )

    def test_sparse_attention(self):
        query = torch.randn(1, 3, 8, 512, device="cuda")
        cache = torch.randn(1, 129, 512, device="cuda")
        sinks = torch.randn(8, device="cuda", dtype=torch.float32)
        indices = (
            torch.arange(129, device="cuda", dtype=torch.int32).expand(1, 3, -1).clone()
        )
        indices[:, 0] = -1
        indices[:, 1, 17:] = -1
        actual = self.kernel.sparse_attn(query, cache, sinks, indices, 512**-0.5)
        expected = torch.zeros_like(query, device="cpu")
        for row, count in ((1, 17), (2, 129)):
            q, kv = query.cpu()[0, row].float(), cache.cpu()[0, :count].float()
            logits = q @ kv.T * 512**-0.5
            probabilities = torch.cat((logits, sinks.cpu()[:, None]), -1).softmax(-1)[
                :, :-1
            ]
            expected[0, row] = probabilities @ kv
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.005)

    def test_sinkhorn(self):
        mixes = torch.randn(2, 3, 24, device="cuda", dtype=torch.float32)
        scales = torch.tensor([0.3, 0.2, 0.5], device="cuda", dtype=torch.float32)
        base = torch.randn(24, device="cuda", dtype=torch.float32)
        pre, post, comb = self.kernel.hc_split_sinkhorn(mixes, scales, base)
        x, s, b = mixes.cpu(), scales.cpu(), base.cpu()
        expected_pre = (x[..., :4] * s[0] + b[:4]).sigmoid() + 1e-6
        expected_post = 2 * (x[..., 4:8] * s[1] + b[4:8]).sigmoid()
        expected_comb = (x[..., 8:] * s[2] + b[8:]).reshape(2, 3, 4, 4).softmax(
            -1
        ) + 1e-6
        expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
        for _ in range(19):
            expected_comb /= expected_comb.sum(-1, keepdim=True) + 1e-6
            expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
        for actual, expected in (
            (pre, expected_pre),
            (post, expected_post),
            (comb, expected_comb),
        ):
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
