import unittest

import torch
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0


class NativeUe8m0ScaleTest(unittest.TestCase):
    def test_checkpoint_scales_pack_identically_to_float_scales(self):
        self.assertTrue(torch.cuda.is_available())
        exponents = torch.tensor(
            [[-120, -1, 0, 120], [-100, -10, 10, 100]], device="cuda"
        )
        values = torch.exp2(exponents.float())
        native = values.to(torch.float8_e8m0fnu)
        torch.testing.assert_close(native.float(), values, rtol=0, atol=0)
        for rows in (129, 256):
            expected = _transform_scale_ue8m0(values, rows)
            actual = _transform_scale_ue8m0(native, rows)
            self.assertEqual(actual.dtype, torch.int32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
