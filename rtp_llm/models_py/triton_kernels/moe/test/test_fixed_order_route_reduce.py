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
