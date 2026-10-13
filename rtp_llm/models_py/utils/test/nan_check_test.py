import os
import unittest

import torch

from rtp_llm.models_py.utils.nan_check import (
    NanCheckMetadata,
    NanCheckStage,
    detect_nan,
    detect_nan_tensor,
)


class NanCheckTest(unittest.TestCase):
    def test_first_value_per_token_and_first_stage_wins(self):
        values = torch.tensor(
            [[1.0, float("inf"), float("nan")], [float("nan"), 0.0, 2.0], [1, 2, 3]]
        )
        records = torch.zeros((3, 4), dtype=torch.int64)
        detect_nan(values, NanCheckMetadata(records, 2, NanCheckStage.ATTENTION))
        self.assertEqual(records.tolist(), [[2, 1, 1, 2], [2, 1, 0, 1], [0, 0, 0, 0]])
        values.fill_(float("-inf"))
        detect_nan(values, NanCheckMetadata(records, 2, NanCheckStage.FFN))
        self.assertEqual(records.tolist(), [[2, 1, 1, 2], [2, 1, 0, 1], [2, 2, 0, 3]])

    def test_verify_rows_stay_independent(self):
        values = torch.zeros((2, 3, 4))
        values[0, 2, 1] = float("nan")
        values[1, 0, 3] = float("-inf")
        records = torch.zeros((6, 4), dtype=torch.int64)
        detect_nan_tensor(values, records, 3, 2)
        self.assertEqual(records[:, 3].tolist(), [0, 0, 1, 3, 0, 0])
        self.assertEqual(records[2].tolist(), [3, 2, 1, 1])
        self.assertEqual(records[3].tolist(), [3, 2, 3, 3])
        # Acceptance filtering belongs to the engine: rejected rows remain
        # observable here without contaminating another token's record.
        self.assertFalse(records.reshape(2, 3, 4)[0, :2, 3].any())

    def test_disabled_does_not_touch_tensor(self):
        detect_nan(None, None)
        detect_nan_tensor(None, None, 0, 1)

    def test_empty_and_noncontiguous(self):
        records = torch.zeros((2, 4), dtype=torch.int64)
        detect_nan_tensor(torch.empty((2, 0)), records, 0, 1)
        self.assertFalse(records.any())
        values = torch.zeros((5, 2))
        values[4, 1] = float("inf")
        detect_nan_tensor(values.t(), records, 0, 1)
        self.assertEqual(records[1].tolist(), [0, 1, 4, 2])


class NanCheckCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            if os.environ.get("GPU_COUNT") == "1":
                raise AssertionError(
                    "The GPU test target requires a visible CUDA device"
                )
            raise unittest.SkipTest("CUDA is required")

    def test_cuda_matches_cpu_across_tiles(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            values = torch.zeros((4, 8193), dtype=dtype)
            values[0, 8192] = float("nan")
            values[1, 4097] = float("inf")
            values[1, 1023] = float("-inf")
            values[2, 2048] = float("inf")
            values[3, 0] = torch.finfo(dtype).max
            expected = torch.zeros((4, 4), dtype=torch.int64)
            detect_nan_tensor(values, expected, 7, 2)
            actual = torch.zeros_like(expected, device="cuda")
            detect_nan_tensor(values.cuda(), actual, 7, 2)
            torch.testing.assert_close(actual.cpu(), expected)

    def test_cuda_strided_rows_keep_first_site(self):
        values = torch.zeros((33, 3), device="cuda").t()
        values[0, 32] = float("nan")
        values[1, 7] = float("inf")
        records = torch.zeros((3, 4), dtype=torch.int64, device="cuda")
        detect_nan_tensor(values, records, 5, 1)
        values.fill_(float("-inf"))
        detect_nan_tensor(values, records, 6, 2)
        torch.testing.assert_close(
            records.cpu(),
            torch.tensor([[5, 1, 32, 1], [5, 1, 7, 2], [6, 2, 0, 3]]),
        )

    def test_graph_replay_reads_new_values_and_reset_records(self):
        values = torch.zeros((3, 32), device="cuda")
        records = torch.zeros((3, 4), dtype=torch.int64, device="cuda")
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            detect_nan_tensor(values, records, 4, 1)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            detect_nan_tensor(values, records, 4, 1)
        values[1, 17] = float("nan")
        graph.replay()
        self.assertEqual(records[1].tolist(), [4, 1, 17, 1])
        values.zero_()
        records.zero_()
        graph.replay()
        self.assertFalse(records.any().item())


if __name__ == "__main__":
    unittest.main()
