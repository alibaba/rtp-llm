"""Check scores, top-k visibility, and live Graph replay for prefill scoring."""

import math
import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.qwen4_exp.indexer_prefill_score import try_prefill_score


class IndexerPrefillScoreTest(unittest.TestCase):
    def inputs(self, length=131, blocks=37, strided=False):
        torch.manual_seed(73)
        shape = (2, length, 4, 128)
        if strided:
            q = torch.randn(2, length, 8, 128, device="cuda", dtype=torch.bfloat16)[
                :, :, ::2
            ]
            k = torch.randn(2, blocks * 2, 128, device="cuda", dtype=torch.bfloat16)[
                :, ::2
            ]
        else:
            q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(2, blocks, 128, device="cuda", dtype=torch.bfloat16)
        return q, k

    def reference(self, q, k):
        return torch.relu(
            torch.matmul(q.float(), k.float().transpose(-1, -2).unsqueeze(1)).transpose(
                -1, -2
            )
        ).sum(-1) / math.sqrt(128)

    def test_ragged_strided_scores_match_fp64_reference(self):
        for strided in (False, True):
            q, k = self.inputs(strided=strided)
            with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_PREFILL_SCORE": "1"}):
                actual = try_prefill_score(q, k)
            self.assertIsNotNone(actual)
            expected = torch.relu(
                torch.matmul(
                    q.double(), k.double().transpose(-1, -2).unsqueeze(1)
                ).transpose(-1, -2)
            ).sum(-1) / math.sqrt(128)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual.double(), expected, atol=5e-6, rtol=5e-6)

    def test_long_prefill_preserves_topk_selection_and_invisible_blocks(self):
        q, k = self.inputs(length=4093, blocks=1023)
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_PREFILL_SCORE": "1"}):
            actual = try_prefill_score(q, k)
        expected = self.reference(q, k)
        complete = (torch.arange(q.shape[1], device=q.device) + 1) // 4
        visible = torch.arange(k.shape[1], device=q.device)[None] < complete[:, None]
        actual = actual.masked_fill(~visible[None], float("-inf"))
        expected = expected.masked_fill(~visible[None], float("-inf"))
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        actual_top = actual.topk(512, dim=-1).indices.sort(-1).values
        expected_top = expected.topk(512, dim=-1).indices.sort(-1).values
        self.assertTrue(torch.equal(actual_top, expected_top))

    def test_graph_replay_refreshes_queries_and_keys(self):
        q, k = self.inputs(strided=True)
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_PREFILL_SCORE": "1"}):
            try_prefill_score(q, k)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    try_prefill_score(q, k)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = try_prefill_score(q, k)
            for scale in (0.1, 0.5, 1.0):
                q.copy_(torch.randn_like(q) * scale)
                k.copy_(torch.randn_like(k) * scale)
                graph.replay()
                torch.testing.assert_close(
                    actual, self.reference(q, k), atol=5e-6, rtol=5e-6
                )

    def test_gate_and_unsupported_inputs_fall_back(self):
        q, k = self.inputs()
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_PREFILL_SCORE": "0"}):
            self.assertIsNone(try_prefill_score(q, k))
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_PREFILL_SCORE": "1"}):
            self.assertIsNone(try_prefill_score(q[:, :1], k))
            self.assertIsNone(try_prefill_score(q.float(), k.float()))
            with patch("torch.version.hip", "hip"):
                self.assertIsNone(try_prefill_score(q, k))
            with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
                self.assertIsNone(try_prefill_score(q, k))


if __name__ == "__main__":
    unittest.main()
