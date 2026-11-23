"""Real floating checkpoints and native W8A16 kernels; never skip a wrong card."""

import os
import unittest

import torch
from test_weight_only_int8 import load_fixture

from rtp_llm.models_py.quant_methods.weight_only_int8 import (
    apply_packed_weight_only_int8,
)


class WeightOnlyInt8NativeTest(unittest.TestCase):
    def test_mixed_gemm_layout_and_projection(self):
        self.assertEqual(torch.cuda.get_device_capability(), (int(os.environ["W8A16_EXPECT_SM"]), 0))
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer, q, scale, bias = load_fixture(
                    "cuda:0", dtype=dtype, rows=160, width=96, startup=True,
                )
                dense = (q * scale).cuda()  # independently computed [K,N]
                # Basis vectors inspect every output channel and every packed
                # lane, including both 64-row interleaves and both TP pads.
                eye = torch.eye(96, device="cuda", dtype=dtype)
                actual = apply_packed_weight_only_int8(
                    eye, layer.weight, layer.weight_scale, 160
                )
                torch.testing.assert_close(actual, dense, atol=0, rtol=0)
                self.assertEqual(layer(eye[:0]).shape, (0, 160))
                # Include a partial M tile, multidimensional input and fused
                # bias as well as the normal row-parallel post-GEMM bias.
                x = torch.linspace(-1, 1, 3 * 7 * 96, device="cuda", dtype=dtype).reshape(3, 7, 96)
                reference = torch.nn.functional.linear(x, dense.T, bias.cuda())
                actual = apply_packed_weight_only_int8(
                    x, layer.weight, layer.weight_scale, 160, bias.cuda()
                )
                torch.testing.assert_close(actual, reference, atol=0.05, rtol=0.01)
                reference = torch.nn.functional.linear(x, dense.T) + bias.cuda()
                torch.testing.assert_close(layer(x), reference, atol=0.05, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
