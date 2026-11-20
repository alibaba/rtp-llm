"""Exact selection and graph replay beyond the former score-capacity boundary."""

import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.kernels.cuda.qsa.test.reference.topk_reference import assert_qsa_selection_close
from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import (
    try_select_qsa_tokens,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaSm90CapacityTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")
        torch.manual_seed(20261117)

    def select(self, scores, lengths, **options):
        # Matching native output alone would also pass after silently taking
        # the unfused torch.topk path. Require fused dispatch during capture too.
        with patch.object(
            torch, "topk", side_effect=AssertionError("Unexpected native TopK fallback")
        ):
            actual = try_select_qsa_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048, **options
            )
        self.assertIsNotNone(actual)
        return actual

    @staticmethod
    def reference(scores, lengths, finite):
        masked = scores.masked_fill(
            torch.arange(scores.shape[1], device=scores.device)[None, :]
            >= lengths[:, None] // 4,
            float("-inf"),
        )
        values, ids = torch.topk(masked, 512, dim=-1)
        valid = torch.isfinite(values) if finite else values > float("-inf")
        tokens = ids[..., None] * 4 + torch.arange(4, device=scores.device)
        tokens = torch.where(valid[..., None], tokens, -1).flatten(1).int()
        tail = lengths[:, None] // 4 * 4 + torch.arange(3, device=scores.device)
        return torch.cat(
            (tokens, torch.where(tail < lengths[:, None], tail, -1).int()), 1
        )

    def test_capacity_and_distributions(self):
        for columns in (512, 8191, 8192, 8193, 8447, 8704, 8705, 16384, 32768, 131073):
            for rows in (1, 2, 16, 32, 64, 128):
                scores = torch.empty((rows, columns), device="cuda")
                lengths = torch.randint(
                    0, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
                )
                lengths[0] = columns * 4 + 3
                for kind in ("random", "ties", "zero", "nonfinite"):
                    scores.normal_()
                    if kind == "ties":
                        scores.copy_(torch.randint(-3, 4, scores.shape, device="cuda"))
                    if kind == "zero":
                        scores.zero_()
                        scores.view(torch.int32)[:, ::2] = -2147483648
                    if kind == "nonfinite":
                        scores[:, ::13] = float("inf")
                        scores[:, 1::17] = float("-inf")
                        scores[:, 2::19] = float("nan")
                        scores.view(torch.int32)[:, 3::23] = -4194303
                    for finite in (True, False):
                        with self.subTest(
                            columns=columns, rows=rows, kind=kind, finite=finite
                        ):
                            actual = self.select(scores, lengths, finite_only=finite)
                            assert_qsa_selection_close(actual, self.reference(scores, lengths, finite), scores, lengths)

    def test_signed_zero_above_and_at_selection_threshold(self):
        for columns in (1024, 8192, 8447, 8705, 32768):
            for rows in (1, 32):
                lengths = torch.full(
                    (rows,), columns * 4, device="cuda", dtype=torch.int32
                )
                order = torch.randperm(columns, device="cuda")
                for positives in (0, 255, 511):
                    for zeros in (2, 511, 512, 513, 1024):
                        zeros = min(zeros, columns - positives)
                        scores = torch.full((rows, columns), -1.0, device="cuda")
                        scores[:, order[:zeros]] = 0.0
                        scores.view(torch.int32)[:, order[:zeros:2]] = -2147483648
                        scores[:, order[zeros : zeros + positives]] = 1.0
                        with self.subTest(
                            columns=columns, rows=rows, positives=positives, zeros=zeros
                        ):
                            actual = self.select(scores, lengths)
                            assert_qsa_selection_close(actual, self.reference(scores, lengths, True), scores, lengths)

    def test_wide_strided_scores_use_native_selector(self):
        scores = torch.randn(2, 65536, device="cuda")[:, ::2]
        lengths = torch.full((2,), 32000, device="cuda", dtype=torch.int32)
        actual = self.select(scores, lengths)
        assert_qsa_selection_close(actual, self.reference(scores, lengths, True), scores, lengths)

    def test_graph_live_lengths_and_fallback(self):
        for rows in (1, 32, 64, 128):
            scores = torch.randn(rows, 32768, device="cuda")
            lengths = torch.full((rows,), 8192, device="cuda", dtype=torch.int32)

            def select():
                return self.select(scores, lengths)

            select()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = select()
            for length in (0, 3, 2047, 8193, 32771, 33791, 131071):
                lengths.fill_(length)
                scores.normal_()
                graph.replay()
                assert_qsa_selection_close(actual, self.reference(scores, lengths, True), scores, lengths)


if __name__ == "__main__":
    unittest.main()
