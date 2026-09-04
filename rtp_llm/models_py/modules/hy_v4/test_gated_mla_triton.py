import math
import unittest

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.hy_v4.gated_mla_triton import (
    maybe_fused_gated_mla_proj_mxfp8,
)
from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
    sigmoid_mul_fp8_quant_fwd,
)


def _sm10_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@unittest.skipUnless(_sm10_available(), "SM10x CUDA GPU required")
class Hy4GatedMlaTritonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20260904)
        cls.k = 6144
        cls.n = 16384
        cls.hidden = torch.randn(16, cls.k, device="cuda", dtype=torch.bfloat16)
        cls.attn = torch.randn(16, cls.n, device="cuda", dtype=torch.bfloat16)
        weight_storage = torch.randn(
            cls.k, cls.n, device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(cls.k)
        # Match CudaF16Linear: logical [N, K] view over contiguous [K, N].
        cls.weight = weight_storage.contiguous().T

    def test_matches_existing_gated_mla_mxfp8_boundary_bitwise(self):
        for m in (1, 2, 4, 8, 12, 16):
            with self.subTest(m=m):
                hidden = self.hidden[:m]
                attn = self.attn[:m]
                gate = F.linear(hidden, self.weight)
                expected_q, expected_scale = sigmoid_mul_fp8_quant_fwd(
                    attn,
                    gate,
                    quant_group_size=32,
                    scale_ue8m0=True,
                    round_scale_to_pow2=True,
                    column_major_scales=True,
                )

                actual = maybe_fused_gated_mla_proj_mxfp8(
                    hidden, self.weight, attn
                )
                self.assertIsNotNone(actual)
                assert actual is not None
                actual_q, actual_scale = actual
                self.assertTrue(
                    torch.equal(
                        expected_q.view(torch.uint8), actual_q.view(torch.uint8)
                    )
                )
                self.assertTrue(torch.equal(expected_scale, actual_scale))

    def test_unsupported_layout_and_prefill_fall_back(self):
        self.assertIsNone(
            maybe_fused_gated_mla_proj_mxfp8(
                self.hidden[:4], self.weight.contiguous(), self.attn[:4]
            )
        )
        self.assertIsNone(
            maybe_fused_gated_mla_proj_mxfp8(
                torch.randn(17, self.k, device="cuda", dtype=torch.bfloat16),
                self.weight,
                torch.randn(17, self.n, device="cuda", dtype=torch.bfloat16),
            )
        )


if __name__ == "__main__":
    unittest.main()
