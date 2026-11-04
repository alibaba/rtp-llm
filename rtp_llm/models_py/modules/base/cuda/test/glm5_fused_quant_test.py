import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
)
from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant,
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
)
from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
    fused_strided_rmsnorm,
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
)


class Glm5FusedQuantTest(unittest.TestCase):
    def test_empty_strided_norm_preserves_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for width in (1536, 16384):
                x = torch.empty((0, width * 2), device="cuda", dtype=dtype)[:, ::2]
                weight = torch.ones(width, device="cuda", dtype=dtype)
                result = fused_strided_rmsnorm(x, weight)
                self.assertEqual(result.shape, x.shape)
                self.assertEqual(result.dtype, dtype)

    def check_quant(self, actual, scales, expected, packed):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(scales.dtype, torch.int32 if packed else torch.float32)
        if not expected.numel():
            return
        reference, reference_scales = sgl_per_token_group_quant_fp8(
            expected,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=packed,
        )
        torch.testing.assert_close(scales, reference_scales)
        torch.testing.assert_close(actual.float(), reference.float(), rtol=0, atol=0)

    def test_add_norm_quant_single_and_dual_output(self):
        torch.manual_seed(17)
        for rows in (0, 1, 33):
            for magnitude in (0.0, 1e-6, 1.0):
                for packed in (False, True):
                    with self.subTest(rows=rows, magnitude=magnitude, packed=packed):
                        hidden = (
                            torch.randn(rows, 2048, device="cuda") * magnitude
                        ).bfloat16()[:, :1536]
                        residual = torch.randn_like(hidden) * magnitude
                        weight = torch.ones(1536, device="cuda", dtype=torch.bfloat16)
                        summed = hidden.float() + residual.float()
                        normed = (
                            summed
                            * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
                        ).bfloat16()
                        for dual in (False, True):
                            working = residual.clone()
                            pointer = working.data_ptr()
                            function = (
                                fused_add_rmsnorm_fp8_quant_with_bf16_output
                                if dual
                                else fused_add_rmsnorm_fp8_quant
                            )
                            result = function(
                                hidden, working, weight, scale_ue8m0=packed
                            )
                            self.assertEqual(working.data_ptr(), pointer)
                            torch.testing.assert_close(
                                working, summed.bfloat16(), rtol=0, atol=0
                            )
                            if dual:
                                torch.testing.assert_close(result[0], normed)
                            self.check_quant(result[-2], result[-1], normed, packed)

    def test_strided_norm_quant(self):
        torch.manual_seed(19)
        for rows in (0, 1, 33):
            for packed in (False, True):
                for column_stride in (1, 2):
                    with self.subTest(
                        rows=rows, packed=packed, column_stride=column_stride
                    ):
                        x = torch.randn(
                            rows, 3072, device="cuda", dtype=torch.bfloat16
                        )[:, : 1536 * column_stride : column_stride]
                        weight = torch.ones(1536, device="cuda", dtype=torch.bfloat16)
                        expected = (
                            x.float()
                            * torch.rsqrt(
                                x.float().square().mean(-1, keepdim=True) + 1e-6
                            )
                        ).bfloat16()
                        normed, quant, scales = (
                            fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                                x, weight, scale_ue8m0=packed
                            )
                        )
                        torch.testing.assert_close(normed, expected)
                        self.check_quant(quant, scales, expected, packed)

    def test_silu_quant(self):
        torch.manual_seed(23)
        for rows in (0, 1, 33):
            for magnitude in (0.0, 1e-6, 1.0):
                for packed in (False, True):
                    with self.subTest(rows=rows, magnitude=magnitude, packed=packed):
                        x = (
                            torch.randn(rows, 3072, device="cuda") * magnitude
                        ).bfloat16()
                        gate, up = x.float().chunk(2, dim=-1)
                        expected = (torch.nn.functional.silu(gate) * up).bfloat16()
                        quant, scales = (
                            silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                                x, scale_ue8m0=packed
                            )
                        )
                        self.check_quant(quant, scales, expected, packed)


if __name__ == "__main__":
    unittest.main()
