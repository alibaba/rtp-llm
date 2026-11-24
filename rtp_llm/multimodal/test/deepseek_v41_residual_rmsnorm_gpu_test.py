"""SM100 numerical gate for the out-of-place vision residual RMSNorm."""

import json
import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.vision_residual_add import (
    apply_vision_residual_rmsnorm,
)


class ResidualRMSNormGPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("the fused residual RMSNorm gate requires CUDA")
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("the production specialization targets SM100")
        cls.device = torch.device("cuda:0")

    @staticmethod
    def _relative_l2(actual, expected, chunk_rows=65536):
        squared_error = torch.zeros((), device=actual.device, dtype=torch.float64)
        squared_reference = torch.zeros_like(squared_error)
        max_abs = torch.zeros((), device=actual.device, dtype=torch.float32)
        for start in range(0, actual.shape[0], chunk_rows):
            stop = min(start + chunk_rows, actual.shape[0])
            difference = actual[start:stop].float() - expected[start:stop].float()
            reference = expected[start:stop].float()
            squared_error += difference.double().square().sum()
            squared_reference += reference.double().square().sum()
            max_abs = torch.maximum(max_abs, difference.abs().max())
        return (
            (squared_error / squared_reference).sqrt().item(),
            max_abs.item(),
        )

    @torch.inference_mode()
    def test_matches_bf16_add_then_fp32_rmsnorm(self):
        self.assertEqual(os.environ.get("GPU_COUNT"), "1")
        self.assertEqual(os.environ.get("WORLD_SIZE"), "1")
        torch.manual_seed(20261021)
        weight = torch.randn(1024, device=self.device, dtype=torch.bfloat16)
        results = []
        for rows in (7, 4096, 1058400):
            with self.subTest(rows=rows):
                x = torch.randn(
                    rows, 1024, device=self.device, dtype=torch.bfloat16
                )
                residual = torch.randn_like(x)
                expected_sum = x + residual
                value = expected_sum.float()
                normalized = value * torch.rsqrt(
                    value.square().mean(-1, keepdim=True) + 1e-6
                )
                expected_norm = (weight * normalized).to(torch.bfloat16)
                actual_sum, actual_norm = apply_vision_residual_rmsnorm(
                    x, residual, weight, 1e-6
                )
                self.assertTrue(torch.equal(actual_sum, expected_sum))
                self.assertNotEqual(actual_sum.data_ptr(), x.data_ptr())
                self.assertNotEqual(actual_sum.data_ptr(), residual.data_ptr())
                self.assertNotEqual(actual_norm.data_ptr(), x.data_ptr())
                self.assertNotEqual(actual_norm.data_ptr(), residual.data_ptr())
                relative_l2, max_abs = self._relative_l2(
                    actual_norm, expected_norm
                )
                results.append(
                    {
                        "rows": rows,
                        "relative_l2": relative_l2,
                        "max_abs": max_abs,
                        "sum_byte_equal": True,
                    }
                )
                print(json.dumps(results[-1]), flush=True)
                self.assertLessEqual(relative_l2, 2e-5)
                # Across more than one billion random BF16 values, the largest
                # outlier may land one BF16 ULP from the reference.
                self.assertLessEqual(max_abs, 0.03125)
                del (
                    x,
                    residual,
                    expected_sum,
                    value,
                    normalized,
                    expected_norm,
                    actual_sum,
                    actual_norm,
                )
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    unittest.main()
