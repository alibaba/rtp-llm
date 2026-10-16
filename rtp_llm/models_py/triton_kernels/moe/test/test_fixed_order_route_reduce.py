import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.fixed_order_route_reduce import (
    fixed_order_fp32_route_reduce,
    make_route_local_ids,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/HIP is required")
class FixedOrderRouteReduceTest(unittest.TestCase):
    def test_make_route_local_ids(self):
        token_num = 3
        topk = 2
        packed_ids = torch.tensor(
            [0, (1 << 24) | 2, 3, (7 << 24) | 0xFFFFFF],
            dtype=torch.int32,
            device="cuda",
        )

        actual = make_route_local_ids(packed_ids, token_num, topk)

        torch.testing.assert_close(
            actual.cpu(), torch.tensor([0, 5, 6, 6], dtype=torch.int32)
        )

    def test_make_route_local_ids_high_slot_stays_positive(self):
        # slot >= 128 sets bit 31 of the packed id; an arithmetic int32 shift
        # would sign-extend it into a negative route row.  The extraction must
        # stay logical so large-TopK slots keep mapping to positive rows.
        token_num, topk, slot = 2, 255, 200
        # torch.tensor rejects values above INT32_MAX for an int32 dtype, so
        # wrap the bit-31-set packed id into its signed int32 representation
        # (same bits, e.g. 0xC8000001) instead of relying on silent overflow.
        packed_value = (slot << 24) | 1
        if packed_value >= 1 << 31:
            packed_value -= 1 << 32
        packed_ids = torch.tensor(
            [0, packed_value], dtype=torch.int32, device="cuda"
        )

        actual = make_route_local_ids(packed_ids, token_num, topk)

        torch.testing.assert_close(
            actual.cpu(),
            torch.tensor([0, 1 * topk + slot], dtype=torch.int32),
        )

    def test_reduce_route_rows_consumes_leading_rows_only(self):
        # The deterministic MoE caller passes the route buffer together with
        # its trailing zeroed sentinel scratch row; route_rows must consume
        # exactly the leading token*topk rows and ignore the scratch.
        torch.manual_seed(11)
        token_num, topk, hidden_size = 3, 4, 65
        route_output = torch.randn(
            token_num * topk + 1,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        route_output[-1] = 0  # the scratch row starts zeroed
        output = torch.empty(
            token_num, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        fixed_order_fp32_route_reduce(
            route_output, output, topk, route_rows=token_num * topk
        )
        reference = torch.empty_like(output)
        fixed_order_fp32_route_reduce(
            route_output[: token_num * topk], reference, topk
        )
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        # A row count that disagrees with token_num*topk fails loudly instead
        # of silently consuming the wrong rows.
        with self.assertRaises(ValueError):
            fixed_order_fp32_route_reduce(
                route_output, output, topk, route_rows=token_num * topk - 1
            )

    def test_reduce_matches_fixed_fp32_order_and_repeats_exactly(self):
        torch.manual_seed(7)
        token_num, topk, hidden_size = 5, 8, 513
        route_output = torch.randn(
            token_num * topk,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        route_view = route_output.view(token_num, topk, hidden_size)
        expected_fp32 = torch.zeros(
            token_num, hidden_size, dtype=torch.float32, device="cuda"
        )
        for slot in range(topk):
            expected_fp32.add_(route_view[:, slot].float())
        expected = expected_fp32.bfloat16()

        first = None
        for _ in range(20):
            actual = torch.empty_like(expected)
            fixed_order_fp32_route_reduce(route_output, actual, topk)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if first is None:
                first = actual.clone()
            else:
                torch.testing.assert_close(actual, first, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
