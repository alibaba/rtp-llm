import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa_runtime import select_qsa_paged_tokens
from rtp_llm.models_py.modules.qwen4_exp.qsa_topk_triton import try_select_qsa_tokens


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaTopkTritonTest(unittest.TestCase):
    def reference(self, scores, lengths, **options):
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_FUSED_QSA_TOPK": "0"}):
            return select_qsa_paged_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048, **options
            )

    def test_exact_native_order_for_ties_nonfinite_values_and_ragged_lengths(self):
        torch.manual_seed(821)
        for rows, columns in (
            (1, 1),
            (7, 31),
            (17, 513),
            (32, 1024),
            (127, 2048),
            (4093, 1024),
        ):
            with self.subTest(rows=rows, columns=columns):
                scores = torch.randn(rows, columns, device="cuda")
                scores[0] = 0
                if rows > 1:
                    scores[1, : min(5, columns)] = float("nan")
                    scores[2, 0] = float("inf")
                    scores[3] = float("-inf")
                lengths = torch.randint(
                    0, columns * 4 + 4, (rows,), device="cuda", dtype=torch.int32
                )
                expected = self.reference(scores, lengths)
                actual = try_select_qsa_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_graph_replay_refreshes_strided_scores_and_lengths(self):
        scores = torch.randn(17, 2048, device="cuda")[:, ::2]
        lengths = torch.full((34,), 4096, device="cuda", dtype=torch.int32)[::2]
        try_select_qsa_tokens(scores, lengths, compress_ratio=4, token_budget=2048)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_select_qsa_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
        for step in range(3):
            scores.copy_(torch.randn_like(scores))
            scores[0] = 0
            lengths.copy_(torch.arange(17, device="cuda", dtype=torch.int32) * 7 + step)
            graph.replay()
            torch.testing.assert_close(
                actual, self.reference(scores, lengths), atol=0, rtol=0
            )

    def test_validation_fallback_and_unvalidated_negative_lengths(self):
        scores = torch.randn(7, 31, device="cuda")
        lengths = torch.tensor(
            [-2147483648, -7, -1, 0, 1, 124, 2147483647],
            device="cuda",
            dtype=torch.int32,
        )
        expected = self.reference(scores, lengths, validate_lengths=False)
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_FUSED_QSA_TOPK": "1"}):
            actual = select_qsa_paged_tokens(
                scores,
                lengths,
                compress_ratio=4,
                token_budget=2048,
                validate_lengths=False,
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                select_qsa_paged_tokens(
                    scores, lengths, compress_ratio=4, token_budget=2048
                )
            with self.assertRaisesRegex(ValueError, "do not cover"):
                select_qsa_paged_tokens(
                    scores,
                    torch.full_like(lengths, 128),
                    compress_ratio=4,
                    token_budget=2048,
                )
            for unsupported in (scores.double(), scores[:, :0], scores.cpu()):
                self.assertIsNone(
                    try_select_qsa_tokens(
                        unsupported,
                        lengths.to(unsupported.device),
                        compress_ratio=4,
                        token_budget=2048,
                    )
                )
            for ratio, budget in ((2, 2048), (4, 128)):
                self.assertIsNone(
                    try_select_qsa_tokens(
                        scores, lengths, compress_ratio=ratio, token_budget=budget
                    )
                )


if __name__ == "__main__":
    unittest.main()
