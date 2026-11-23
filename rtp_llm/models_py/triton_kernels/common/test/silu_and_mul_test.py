import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul


class SiluAndMulTest(unittest.TestCase):
    def test_large_row_offsets(self):
        width = 1024
        row_stride = 2 * width
        boundary_row = (1 << 31) // row_stride
        shape = (boundary_row + 2, row_stride)
        inputs = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
        # Use the same stride for the output so both row offsets cross 2**31.
        outputs = torch.zeros_like(inputs)[:, :width]
        boundary_rows = inputs[boundary_row - 1 :]
        boundary_rows[:, 0] = torch.tensor([3, 5, 7], device="cuda")
        expected = silu_and_mul(torch.empty_like(outputs[-3:]), boundary_rows)

        silu_and_mul(outputs, inputs)
        torch.testing.assert_close(outputs[-3:], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
