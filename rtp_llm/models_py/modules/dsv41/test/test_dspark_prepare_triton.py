"""SM90 contracts for DSpark proposal and commit-row preparation fusion."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.speculative._dspark_prepare_triton import (
    try_map_context_rows,
    try_prepare_proposal,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkPrepareTritonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 DSpark preparation contract")

    def test_query_inputs_match_eager_for_both_prefix_dtypes(self):
        for dtype in (torch.int32, torch.int64):
            prefix = torch.tensor([9, 0, 17], device="cuda", dtype=dtype)
            ids = torch.tensor(
                [[5, 6, 7, 8], [11, 12, 13, 14], [21, 22, 23, 24]],
                device="cuda",
                dtype=torch.int64,
            )
            actual = try_prepare_proposal(ids, prefix, width=4, noise_token_id=3)
            self.assertIsNotNone(actual)
            query, positions, prefix32, active = actual
            expected_query = torch.full_like(ids, 3, dtype=torch.int32)
            expected_query[:, 0] = ids[:, 0].to(torch.int32)
            expected_positions = prefix.to(torch.int64)[:, None] + torch.arange(
                4, device="cuda", dtype=torch.int64
            )
            self.assertTrue(torch.equal(query, expected_query))
            self.assertTrue(torch.equal(positions, expected_positions))
            self.assertTrue(torch.equal(prefix32, prefix.to(torch.int32)))
            self.assertTrue(torch.equal(active, prefix > 0))

    def test_commit_rows_match_searchsorted_with_duplicate_starts(self):
        starts = torch.tensor([0, 0, 3, 7], device="cuda", dtype=torch.int64)
        lengths = torch.tensor([0, 3, 4, 2], device="cuda", dtype=torch.int64)
        ends = torch.tensor([8, 13, 24, 31], device="cuda", dtype=torch.int64)
        rows = torch.arange(10, device="cuda", dtype=torch.int64)
        request = torch.searchsorted(starts, rows, right=True) - 1
        safe = request.clamp(0, starts.numel() - 1)
        valid = (
            (request >= 0)
            & (rows >= starts[safe])
            & (rows < starts[safe] + lengths[safe])
        )
        expected_request = torch.where(valid, request, -1).to(torch.int32)
        expected_position = torch.where(
            valid, ends[safe] - lengths[safe] + rows - starts[safe], -1
        ).to(torch.int32)
        actual = try_map_context_rows(starts, lengths, ends, rows.numel())
        self.assertIsNotNone(actual)
        self.assertTrue(torch.equal(actual[0], expected_request))
        self.assertTrue(torch.equal(actual[1], expected_position))

    def test_feature_gates_preserve_fallback(self):
        values = torch.ones(1, device="cuda", dtype=torch.int64)
        ids = torch.ones(4, device="cuda", dtype=torch.int32)
        with patch.dict(os.environ, {"DSV4_FUSED_DSPARK_QUERY_INPUTS": "0"}):
            self.assertIsNone(
                try_prepare_proposal(ids, values, width=4, noise_token_id=0)
            )
        with patch.dict(os.environ, {"DSV4_FUSED_DSPARK_COMMIT_ROWS": "0"}):
            self.assertIsNone(try_map_context_rows(values, values, values, 1))


if __name__ == "__main__":
    unittest.main()
