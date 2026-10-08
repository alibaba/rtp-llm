import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.router import KimiK3RouterProjection


class KimiK3RouterTest(unittest.TestCase):
    def test_native_router_compiles_and_matches_fp32_accumulation(self):
        self.assertTrue(torch.cuda.is_available(), "Native router requires CUDA")
        self.assertGreaterEqual(torch.cuda.get_device_capability()[0], 9)
        torch.manual_seed(137)
        # Full-model router geometry; weights and inputs are BF16, logits FP32.
        weight = torch.randn(7168, 896, device="cuda", dtype=torch.bfloat16) * 0.03
        router = KimiK3RouterProjection(weight)
        self.assertIsNotNone(router._native_router)
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for tokens in (1, 2, 8, 16):
                with self.subTest(tokens=tokens):
                    hidden = (
                        torch.randn(tokens, 7168, device="cuda", dtype=torch.bfloat16)
                        * 0.1
                    )
                    actual = router(hidden)
                    expected = hidden.float() @ weight.float()
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_tf32


if __name__ == "__main__":
    unittest.main()
