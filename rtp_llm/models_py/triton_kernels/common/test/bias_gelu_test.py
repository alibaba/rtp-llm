import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.common.activation import bias_gelu


class BiasGeluTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA/ROCm required")
        self.device = "cuda"

    def _run_case(self, m: int, n: int, dtype: torch.dtype = torch.bfloat16):
        torch.manual_seed(0)
        x = torch.randn(m, n, dtype=dtype, device=self.device)
        bias = torch.randn(n, dtype=dtype, device=self.device)

        output = bias_gelu(x, bias)
        reference = F.gelu(x + bias)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)

    def test_visionbert_fc1_shape(self):
        self._run_case(8192, 3072)

    def test_small_batch_shape(self):
        self._run_case(32, 3072)

    def test_non_power_of_two_hidden_shape(self):
        self._run_case(128, 771)


if __name__ == "__main__":
    unittest.main()
