"""SM80 query RMSNorm+E4M3+UE8M0 fallback contract."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41._v41_query_quant import (
    _query_norm_quant_reference,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA FP8 conversion required")
class QueryNormQuantSm80Test(unittest.TestCase):
    def test_matches_bf16_boundary_and_packed_scales(self):
        torch.manual_seed(17)
        device = "cuda"
        x = torch.randn(4, 1280, device=device, dtype=torch.bfloat16)
        weight = torch.randn(1280, device=device, dtype=torch.bfloat16)
        normalized, (quantized, scales) = _query_norm_quant_reference(
            x, weight, 1e-6
        )
        flat = x.float()
        inv = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = (flat * inv * weight.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(normalized, expected))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scales.shape), (4, 10))
        self.assertEqual(scales.stride(), (1, 4))

        groups = expected.float().reshape(4, 40, 32)
        amax = groups.abs().amax(-1)
        raw = (amax / 448.0).clamp_min(1.0e-10)
        bits = raw.view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scale = (exponent << 23).view(torch.float32)
        exp_q = (
            (groups / scale[..., None])
            .clamp(-448.0, 448.0)
            .reshape(4, 1280)
            .to(torch.float8_e4m3fn)
        )
        self.assertTrue(torch.equal(quantized.view(torch.uint8), exp_q.view(torch.uint8)))
        packed = torch.zeros((4, 10), dtype=torch.int32, device=device)
        for shift in range(4):
            packed |= exponent[:, shift::4] << (8 * shift)
        self.assertTrue(torch.equal(scales, packed))


if __name__ == "__main__":
    unittest.main()
