import unittest

import torch
from test_weight_only_int8 import load_fixture


class WeightOnlyInt8SM8xTest(unittest.TestCase):
    def test_native_forward_from_float_checkpoint(self):
        # This target is GPU-tagged: a wrong worker is a failure, never a skip.
        self.assertEqual(torch.cuda.get_device_capability()[0], 8)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer, q, scale, bias = load_fixture("cuda:0", dtype=dtype)
                x = torch.linspace(
                    -1, 1, 2 * 3 * 256, device="cuda", dtype=dtype
                ).reshape(2, 3, 256)
                actual = layer(x)
                # Row-parallel owners add bias *after* the GEMM (and any TP reduction).
                reference = torch.nn.functional.linear(
                    x, (q * scale).T.to(x.device)
                ) + bias.to(x.device)
                torch.testing.assert_close(
                    actual, reference,
                    atol=0.13 if dtype == torch.bfloat16 else 0.02,
                    rtol=0.01,
                )
                self.assertEqual(layer(x[:0]).shape, (0, 3, 192))

    def test_native_forward_preserves_unpadded_activation_widths(self):
        self.assertEqual(torch.cuda.get_device_capability()[0], 8)
        layer, q, scale, bias = load_fixture(
            "cuda:0", rows=160, width=96, startup=True
        )
        x = torch.linspace(-1, 1, 4 * 96, dtype=torch.float16, device="cuda").view(4, 96)
        expected = torch.nn.functional.linear(x, (q * scale).T.cuda()) + bias.cuda()
        torch.testing.assert_close(layer(x), expected, atol=0.05, rtol=0.01)
        self.assertEqual(layer(x[:0]).shape, (0, 160))


if __name__ == "__main__":
    unittest.main()
