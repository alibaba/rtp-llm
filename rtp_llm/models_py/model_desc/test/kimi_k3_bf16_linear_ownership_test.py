import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.linear import bf16_linear


class KimiK3Bf16LinearOwnershipTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_two_dimensional_result_owns_storage_for_inplace_activation(self):
        hidden = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
        output = bf16_linear(hidden, weight)
        self.assertTrue(output.is_contiguous())
        self.assertIsNone(output._base)


if __name__ == "__main__":
    unittest.main()
