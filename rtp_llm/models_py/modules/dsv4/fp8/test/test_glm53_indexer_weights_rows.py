"""FP32 indexer projection invariance across cold/cache and ragged row layouts."""

import unittest

import torch
from rtp_llm.models_py.modules.dsv4.fp8.indexer import _linear_fixed_rows


class IndexerWeightsRowsTest(unittest.TestCase):
    def test_cold_cached_and_packed_rows_are_bitwise_identical(self):
        self.assertTrue(torch.cuda.is_available())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(531011)
        x = torch.randn(8610, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(32, 4096, device="cuda") * 0.01
        cold = _linear_fixed_rows(x, weight, 512)
        tail = x[-418:]
        cached = _linear_fixed_rows(tail, weight, 512)
        torch.testing.assert_close(cold[-418:], cached, rtol=0, atol=0)
        packed = torch.cat((x[:31], tail, tail, x[:17]))
        actual = _linear_fixed_rows(packed, weight, 512)
        for start in (31, 449):
            torch.testing.assert_close(
                actual[start : start + 418], cached, rtol=0, atol=0
            )
        self.assertEqual(actual.dtype, torch.float32)

    def test_empty_and_batched_shapes(self):
        weight = torch.randn(32, 128, device="cuda")
        empty = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(_linear_fixed_rows(empty, weight, 512).shape, (0, 32))
        x = torch.randn(3, 75, 128, device="cuda", dtype=torch.bfloat16)
        result = _linear_fixed_rows(x, weight, 512)
        flat = _linear_fixed_rows(x.reshape(-1, 128), weight, 512)
        torch.testing.assert_close(result.reshape(-1, 32), flat, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
