import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa import histogram_topk
from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton import (
    try_select_qsa_tokens,
)


class HistogramTopKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")

    def setUp(self):
        torch.manual_seed(2052)
        self.env = patch.dict(
            "os.environ",
            RTP_QWEN4_PREFILL_TOPK_IMPL="radix",
            RTP_QWEN4_PREFILL_TOPK_VERIFY="0",
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    @staticmethod
    def reference(scores, lengths, finite=True):
        complete = torch.div(lengths, 4, rounding_mode="floor")
        columns = torch.arange(scores.shape[1], device=scores.device)
        masked = torch.where(
            columns[None, :] < complete[:, None], scores, -float("inf")
        )
        values, blocks = torch.topk(masked, 512, dim=-1)
        valid = (
            torch.isfinite(values)
            if finite
            else ~torch.isnan(values) & ~torch.isneginf(values)
        )
        valid &= blocks < complete[:, None]
        lanes = torch.arange(4, device=scores.device)
        selected = blocks[:, :, None] * 4 + lanes
        selected = torch.where(valid[:, :, None], selected, -1).flatten(1)
        recent = complete[:, None] * 4 + lanes[:3]
        recent = torch.where(lanes[:3] < (lengths - complete * 4)[:, None], recent, -1)
        return torch.cat((selected, recent), dim=1).to(torch.int32)

    def test_exact_order_positive_buckets_ties_and_special_values(self):
        for cols in (6144, 8192, 16384):
            lengths = (
                torch.arange(128, device="cuda", dtype=torch.int32) * (cols * 4 // 128)
                + 1
            )
            for kind in ("normal", "positive", "narrow", "bf16", "ties", "special"):
                scores = torch.randn(128, cols, device="cuda")
                if kind == "positive":
                    scores.add_(5)
                elif kind == "narrow":
                    scores.add_(20)
                elif kind == "bf16":
                    scores = scores.bfloat16().float()
                elif kind == "ties":
                    scores.fill_(1)
                elif kind == "special":
                    scores[:, :4] = torch.tensor(
                        [float("nan"), -0.0, float("inf"), -float("inf")], device="cuda"
                    )
                for finite in (True, False) if kind == "special" else (True,):
                    output = histogram_topk.try_histogram_qsa_tokens(
                        scores, lengths, finite_only=finite
                    )
                    self.assertIsNotNone(output)
                    torch.testing.assert_close(
                        output, self.reference(scores, lengths, finite), rtol=0, atol=0
                    )

    def test_real_8192_rows_and_optional_runtime_validation(self):
        histogram_topk._VERIFIED.clear()
        for cols in (6144, 8192, 12288, 16384):
            scores = torch.randn(8192, cols, device="cuda")
            if cols == 8192:
                scores.add_(5)
            elif cols == 12288:
                scores.add_(20)
            lengths = (
                torch.arange(1, 8193, device="cuda", dtype=torch.int32)
                + cols * 4
                - 8192
            )
            with patch.dict("os.environ", RTP_QWEN4_PREFILL_TOPK_VERIFY="1"):
                output = histogram_topk.try_histogram_qsa_tokens(scores, lengths)
            torch.testing.assert_close(
                output, self.reference(scores, lengths), rtol=0, atol=0
            )
            self.assertIn(
                (scores.device.index, 8192, cols, True), histogram_topk._VERIFIED
            )

    def test_graph_refreshes_fallback_and_input_scores(self):
        scores = torch.randn(128, 16384, device="cuda")
        lengths = torch.full((128,), 65536, dtype=torch.int32, device="cuda")
        histogram_topk.try_histogram_qsa_tokens(scores, lengths)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = histogram_topk.try_histogram_qsa_tokens(scores, lengths)
        try:
            for gain, length in ((0.0, 65536), (1.0, 65536), (0.1, 1025), (7.0, 32769)):
                scores.copy_(torch.randn_like(scores) * gain)
                lengths.fill_(length)
                graph.replay()
                torch.testing.assert_close(
                    output, self.reference(scores, lengths), rtol=0, atol=0
                )
        finally:
            graph.reset()

    def test_real_dense_dispatch_avoids_torch_and_unsupported_inputs_decline(self):
        scores = torch.randn(8192, 16384, device="cuda")
        lengths = torch.arange(1, 8193, device="cuda", dtype=torch.int32) + 57344
        expected = self.reference(scores, lengths)
        histogram_topk.try_histogram_qsa_tokens(scores, lengths)
        with (
            patch("torch.topk", side_effect=AssertionError("native TopK fallback")),
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.qsa.cub_topk.try_cub_qsa_tokens",
                side_effect=AssertionError("histogram dispatch missed"),
            ),
        ):
            actual = try_select_qsa_tokens(
                scores, lengths, compress_ratio=4, token_budget=2048
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(
            histogram_topk.try_histogram_qsa_tokens(scores[:, ::2], lengths)
        )
        self.assertIsNone(
            histogram_topk.try_histogram_qsa_tokens(scores[:2], lengths[:2])
        )
        self.assertIsNone(
            histogram_topk.try_histogram_qsa_tokens(scores.bfloat16(), lengths)
        )


if __name__ == "__main__":
    unittest.main()
