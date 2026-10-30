"""Native packed KV-up interface for BF16 and FP8 weights."""

import unittest

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)


class KvUpProjectionTest(unittest.TestCase):
    def test_packed_projection_writes_only_k_nope_and_v(self):
        torch.manual_seed(713)
        heads, latent, splits = 12, 512, (128, 64, 128)
        left, middle, right = splits
        output_head = left + middle + right
        columns = heads * (left + right)
        args = dict(device="cuda", dtype=torch.bfloat16)
        hidden = torch.empty((17, latent), **args).random_(-1, 2)
        weight = torch.empty((columns, latent), **args).random_(-1, 2)
        scales = torch.empty_strided(
            (columns, 1), (1, columns), device=weight.device, dtype=torch.int32
        ).fill_(0x7F7F7F7F)

        projections = (
            CudaF16Linear(weight.T.contiguous()),
            CudaFp8GEMMLinear(
                weight=weight.to(torch.float8_e4m3fn), weight_scales=scales
            ),
        )
        expected = (hidden.float() @ weight.float().T).to(torch.bfloat16)
        expected = expected.view(-1, heads, left + right)
        for projection in projections:
            with self.subTest(precision=type(projection).__name__):
                output = torch.full(
                    (len(hidden), heads * output_head), float("nan"), **args
                )
                actual = projection.forward_skip_head_mid(hidden, splits, output=output)
                self.assertIs(actual, output)
                actual = actual.view(-1, heads, output_head)
                torch.testing.assert_close(actual[..., :left], expected[..., :left])
                torch.testing.assert_close(actual[..., -right:], expected[..., -right:])
                self.assertTrue(torch.isnan(actual[..., left : left + middle]).all())


if __name__ == "__main__":
    unittest.main()
