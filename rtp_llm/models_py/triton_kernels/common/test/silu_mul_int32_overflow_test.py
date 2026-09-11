"""Guard the row-offset arithmetic in _silu_and_mul_kernel against int32 overflow.

The kernel indexes its row with `tl.program_id(axis=0) * input_row_stride`. Both
operands are int32, so the product wraps once a row offset passes 2**31 elements and
is then sign-extended into the pointer, which puts the whole row outside the tensor.
On the grouped-GEMM MoE path the row count is the number of (token, expert) pairs
landing on one rank, so a large prefill batch with skewed routing reaches that
threshold: at stride 3072 it takes 699051 rows.

This test drives the kernel just past the boundary and checks the last row, which is
the only row whose offset overflows. Before the fix this either faults or silently
writes elsewhere; both are caught here. Read and write are covered separately: the
output stride is half the input's, so a shape that overflows the read offset leaves
the write offset near 2**30 and would miss an output-only regression.

The shape is large by necessity -- the overflow is a property of the offset magnitude,
so it cannot be reproduced on a small tensor. The test skips when the device does not
have the room.
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul

INT32_MAX = 2**31 - 1


class SiluMulInt32OverflowTest(unittest.TestCase):

    # stride(0) of the input, i.e. 2*N. 3072 is GLM-4.x MoE (2 * moe_intermediate_size).
    STRIDE = 3072

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        self.device = torch.device("cuda")

    def _rows_to_overflow(self, stride: int) -> int:
        # One row past the boundary is enough; going further only costs memory.
        return INT32_MAX // stride + 2

    def _require_room(self, rows: int) -> None:
        # bf16 in (rows x STRIDE) + bf16 out (rows x STRIDE/2), plus slack.
        needed = rows * self.STRIDE * 2 * 3 // 2 + (1 << 30)
        free, _total = torch.cuda.mem_get_info(self.device)
        if free < needed:
            raise unittest.SkipTest(
                "needs ~%.1f GiB free to cross the 2**31 row offset, have %.1f GiB"
                % (needed / 2**30, free / 2**30)
            )

    def _assert_boundary_row_is_addressed(self, rows: int, overflowing: str) -> None:
        n = self.STRIDE // 2
        last = rows - 1
        offsets = {"input": last * self.STRIDE, "output": last * n}
        self.assertGreater(
            offsets[overflowing],
            INT32_MAX,
            "the %s row offset does not actually cross the boundary, "
            "the test would prove nothing" % overflowing,
        )

        inp = torch.zeros((rows, self.STRIDE), device=self.device, dtype=torch.bfloat16)
        # zeros, not empty: the "nothing else was written" assertion below only means
        # something if the untouched rows started at a known value.
        out = torch.zeros((rows, n), device=self.device, dtype=torch.bfloat16)

        # Only the overflowing row carries a signal. Everything else stays zero, so a
        # wrapped offset that writes into another row is visible as a nonzero elsewhere
        # rather than being mistaken for a correct result.
        value = 2.0
        gate = 1.0
        inp[last, :n] = value
        inp[last, n:] = gate

        silu_and_mul(out, inp)
        torch.cuda.synchronize()

        expected = float(gate * torch.sigmoid(torch.tensor(gate)).item() * value)
        got = out[last].float()
        self.assertTrue(
            torch.allclose(got, torch.full_like(got, expected), atol=1e-2),
            "row %d (%s offset %d > INT32_MAX) got %s, expected %s"
            % (last, overflowing, offsets[overflowing], got[:4].tolist(), expected),
        )

        # A wrapped write would land in a row that should still be zero.
        self.assertEqual(
            float(out[:last].float().abs().sum().item()),
            0.0,
            "rows below the boundary were written; the row offset wrapped",
        )

    def test_input_row_offset_past_int32_is_still_addressed(self) -> None:
        rows = self._rows_to_overflow(self.STRIDE)
        self._require_room(rows)
        self._assert_boundary_row_is_addressed(rows, "input")

    def test_output_row_offset_past_int32_is_still_addressed(self) -> None:
        # Sized off the output stride, which is half the input's. The input-side
        # case above leaves the write offset at ~2**30, so it cannot catch an
        # output address calculation that degraded to int32 on its own.
        rows = self._rows_to_overflow(self.STRIDE // 2)
        self._require_room(rows)
        self._assert_boundary_row_is_addressed(rows, "output")


if __name__ == "__main__":
    unittest.main()
