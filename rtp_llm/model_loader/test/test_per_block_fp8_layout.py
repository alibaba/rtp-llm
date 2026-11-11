import unittest

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import per_block_cast_to_fp8


class SquareBlockQuantTest(unittest.TestCase):
    def test_scale_shape_and_reconstruction(self):
        weight = torch.arange(256 * 384, dtype=torch.float32).reshape(256, 384)
        weight = (weight.remainder(31) - 15) * torch.tensor(
            [0.01, 0.1, 1.0]
        ).repeat_interleave(128)
        quantized, scales = per_block_cast_to_fp8(weight, 128)
        self.assertEqual(quantized.shape, weight.shape)
        self.assertEqual(scales.shape, (2, 3))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        reconstructed = quantized.float() * scales.repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        self.assertLess(((reconstructed - weight).norm() / weight.norm()).item(), 0.03)
        self.assertTrue(torch.all(scales[:, 0] < scales[:, 1]))
        self.assertTrue(torch.all(scales[:, 1] < scales[:, 2]))


if __name__ == "__main__":
    unittest.main()
