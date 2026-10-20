import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda import qwen4_qsa_topk as cub
from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
    try_select_qsa_tokens,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaCubTopkTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 pinned Torch ordering")
        self.env = patch.dict(os.environ, {"QWEN4_QSA_CUB_TOPK": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(143201)

    @staticmethod
    def reference(scores, lengths, finite=True):
        masked = scores.masked_fill(
            torch.arange(scores.shape[1], device=scores.device)[None, :]
            >= lengths[:, None] // 4,
            float("-inf"),
        )
        values, indices = torch.topk(masked, 512, dim=-1)
        valid = torch.isfinite(values) if finite else values > float("-inf")
        tokens = (
            torch.where(
                valid[..., None],
                indices[..., None] * 4 + torch.arange(4, device=scores.device),
                -1,
            )
            .flatten(1)
            .int()
        )
        tail = lengths[:, None] // 4 * 4 + torch.arange(3, device=scores.device)
        return torch.cat(
            (tokens, torch.where(tail < lengths[:, None], tail, -1).int()), 1
        )

    def test_native_bits_ties_nan_signs_ragged_and_partial_columns(self):
        for rows in (1, 3, 4, 8, 16, 20, 24, 28, 32):
            for columns in (2047, 2048):
                scores = torch.randn(rows, columns, device="cuda")
                lengths = torch.arange(rows, device="cuda", dtype=torch.int32) * 379
                lengths[0] = columns * 4 - 1
                for kind in (
                    "random",
                    "equal",
                    "ties",
                    "nonfinite",
                    "negative_nan",
                    "signed_zero",
                ):
                    scores.normal_()
                    if kind == "equal":
                        scores.zero_()
                    if kind == "ties":
                        scores.copy_(torch.randint(-3, 4, scores.shape, device="cuda"))
                    if kind == "nonfinite":
                        scores[:, ::3] = float("inf")
                        scores[:, 1::5] = float("nan")
                        scores[:, 2::7] = float("-inf")
                    if kind == "negative_nan":
                        scores[:, ::5] = float("nan")
                        scores.view(torch.int32)[:, ::7] = -4194303
                    if kind == "signed_zero":
                        scores.zero_()
                        scores.view(torch.int32)[:, ::2] = -2147483648
                    for finite in (True, False):
                        actual = cub.try_cub_qsa_tokens(
                            scores, lengths, finite_only=finite
                        )
                        self.assertIsNotNone(
                            actual,
                            "Required CUDA compiler or CUB candidate unavailable",
                        )
                        torch.testing.assert_close(
                            actual,
                            self.reference(scores, lengths, finite),
                            rtol=0,
                            atol=0,
                        )

    def test_two_graphs_live_lengths_nan_masks_and_nondefault_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for columns in (2047, 2048):
                scores = torch.randn(4, columns, device="cuda")
                lengths = torch.full(
                    (4,), columns * 4, device="cuda", dtype=torch.int32
                )
                for _ in range(3):
                    cub.try_cub_qsa_tokens(scores, lengths)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = cub.try_cub_qsa_tokens(scores, lengths)
                for step in range(6):
                    scores.normal_()
                    scores[0, ::5] = float("nan")
                    scores.view(torch.int32)[0, ::7] = -4194303
                    lengths.copy_(
                        torch.tensor(
                            [129 + step, 4099, columns * 4 - 1, 7],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    graph.replay()
                    torch.testing.assert_close(
                        actual, self.reference(scores, lengths), rtol=0, atol=0
                    )
                graph.reset()
        torch.cuda.current_stream().wait_stream(stream)

    def test_owner_masked_contract_and_unqualified_fallbacks(self):
        scores = torch.randn(4, 2048, device="cuda")
        lengths = torch.tensor([129, 4099, 8191, 7], device="cuda", dtype=torch.int32)
        scores.masked_fill_(
            torch.arange(2048, device="cuda")[None, :] >= lengths[:, None] // 4,
            float("-inf"),
        )
        actual = try_select_qsa_tokens(
            scores, lengths, compress_ratio=4, token_budget=2048, scores_are_masked=True
        )
        torch.testing.assert_close(
            actual, self.reference(scores, lengths), rtol=0, atol=0
        )
        self.assertIsNone(cub.try_cub_qsa_tokens(scores[:, ::2], lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.double(), lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.requires_grad_(), lengths))
        self.assertIsNone(cub.try_cub_qsa_tokens(scores.detach().cpu(), lengths.cpu()))
        with patch.dict(os.environ, {"QWEN4_QSA_CUB_TOPK": "0"}):
            self.assertIsNone(cub.try_cub_qsa_tokens(scores.detach(), lengths))

    def test_unprimed_capture_or_failed_build_keeps_original_selector(self):
        scores = torch.randn(4, 2048, device="cuda")
        lengths = torch.full((4,), 8192, device="cuda", dtype=torch.int32)
        with patch.object(cub, "_MODULE", None), patch.object(
            cub, "_BUILD_FAILED", False
        ), patch("torch.cuda.is_current_stream_capturing", return_value=True):
            self.assertIsNone(cub.try_cub_qsa_tokens(scores, lengths))
        with patch.object(cub, "_MODULE", None), patch.object(
            cub, "_BUILD_FAILED", True
        ):
            actual = try_select_qsa_tokens(
                scores,
                lengths,
                compress_ratio=4,
                token_budget=2048,
                scores_are_masked=True,
            )
        torch.testing.assert_close(
            actual, self.reference(scores, lengths), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
