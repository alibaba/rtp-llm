import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    _transform_scale_ue8m0,
    per_block_cast_to_fp8,
)
from rtp_llm.models_py.modules.kimi_k3.attention import linear
from rtp_llm.utils.model_weight import W


class KimiK3Fp8LinearTest(unittest.TestCase):
    def test_padded_kda_projection_uses_deepgemm(self):
        if not torch.cuda.is_available():
            self.fail("K3 FP8 GEMM test requires a GPU")
        # Production TP8 KDA has 6284 logical outputs. The fused FP8 weight
        # pads that dimension to 6400 so DeepGEMM can execute it.
        rows, columns = 6400, 7168
        quant = torch.zeros((rows, columns), dtype=torch.float8_e4m3fn, device="cuda")
        scales = torch.ones((rows // 128, columns // 128), dtype=torch.float32, device="cuda")
        weights = {
            K3W.KDA_INPUT: quant,
            K3W.KDA_INPUT_S: _transform_scale_ue8m0(scales, mn=rows),
        }
        kernel = linear(weights, K3W.KDA_INPUT)
        for tokens in (1, 128):
            with self.subTest(tokens=tokens):
                inputs = torch.randn(tokens, columns, dtype=torch.bfloat16, device="cuda")
                actual = kernel(inputs)
                self.assertEqual(actual.shape, (tokens, rows))
                self.assertEqual(torch.count_nonzero(actual).item(), 0)

    def test_kda_and_mla_projections_use_fp8_gemm(self):
        if not torch.cuda.is_available():
            self.fail("K3 FP8 GEMM test requires a GPU")
        torch.manual_seed(37)
        for name, scale_name in (
            (K3W.KDA_INPUT, K3W.KDA_INPUT_S),
            (W.mla_q_b_w, W.mla_q_b_s),
        ):
            with self.subTest(name=name):
                dense = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda") * 0.03
                inputs = torch.randn(16, 256, dtype=torch.bfloat16, device="cuda")
                quant, scales = per_block_cast_to_fp8(dense, use_ue8m0=True)
                weights = {name: quant, scale_name: _transform_scale_ue8m0(scales, mn=256)}
                kernel = linear(weights, name)
                actual = kernel(inputs)
                expected = F.linear(inputs, dense)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertNotEqual(type(kernel).__name__, "KimiK3Bf16Linear")
                torch.testing.assert_close(actual, expected, rtol=0.2, atol=0.2)


if __name__ == "__main__":
    unittest.main()
