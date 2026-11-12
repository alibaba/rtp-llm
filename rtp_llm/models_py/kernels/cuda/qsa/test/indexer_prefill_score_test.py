"""Check scores, top-k visibility, and live Graph replay for prefill scoring."""

import math
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa.indexer_prefill_score import try_prefill_score
from rtp_llm.models_py.modules.qwen4_exp.qsa import indexer as indexer_module
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer


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

    def test_automatic_selection_and_unsupported_inputs_fall_back(self):
        q, k = self.inputs()
        self.assertIsNotNone(try_prefill_score(q, k))
        self.assertIsNone(try_prefill_score(q[:, :1], k))
        self.assertIsNone(try_prefill_score(q.float(), k.float()))
        with patch("torch.version.hip", "hip"):
            self.assertIsNone(try_prefill_score(q, k))
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            self.assertIsNone(try_prefill_score(q, k))

    def reference_selection(self, q, keys, past_length):
        batch, length = q.shape[:2]
        positions = past_length + torch.arange(length, device=q.device)
        complete = (positions + 1) // 4
        scores = self.reference(q, keys)
        visible = torch.arange(keys.shape[1], device=q.device)[None] < complete[:, None]
        values, blocks = scores.masked_fill(~visible[None], float("-inf")).topk(
            min(512, keys.shape[1]), dim=-1
        )
        tokens = blocks[..., None] * 4 + torch.arange(4, device=q.device)
        tokens = torch.where(values[..., None] > float("-inf"), tokens, -1)
        output = torch.full(
            (batch, length, 2051), -1, device=q.device, dtype=torch.int32
        )
        output[..., : tokens.shape[-2] * 4] = tokens.flatten(-2).int()
        tail = complete[:, None] * 4 + torch.arange(3, device=q.device)
        output[..., 2048:] = torch.where(tail <= positions[:, None], tail, -1).int()
        return output

    def test_bounded_fallback_keeps_absolute_positions_ties_and_tail(self):
        for past_length, tied in ((0, False), (5, False), (0, True), (5, True)):
            with self.subTest(past_length=past_length, tied=tied):
                q, _ = self.inputs(length=131, strided=True)
                raw = torch.randn(2, 131, 128, device=q.device, dtype=q.dtype)
                past = torch.randn(2, past_length, 128, device=q.device, dtype=q.dtype)
                if tied:
                    q.zero_()
                cosine = torch.ones(
                    2, past_length + 131, 64, device=q.device, dtype=q.dtype
                )
                sine = torch.zeros_like(cosine)
                indexer = Qwen4ExpQSAIndexer(
                    torch.zeros(640, 8, device=q.device, dtype=q.dtype),
                    torch.zeros(128, device=q.device, dtype=q.dtype),
                    torch.zeros(128, device=q.device, dtype=q.dtype),
                    n_heads=4,
                    kv_heads=1,
                    head_dim=128,
                    token_budget=2048,
                    compress_ratio=4,
                    norm_eps=1e-6,
                )
                keys = indexer.pooled_block_keys_all(
                    torch.cat((past, raw), 1), cosine, sine
                )
                expected = self.reference_selection(q, keys, past_length)
                bytes_per_query = 2 * keys.shape[1] * (2 * 4 + 2) * 4
                with (
                    patch(
                        "rtp_llm.models_py.kernels.cuda.qsa.indexer_prefill_score.try_prefill_score",
                        return_value=None,
                    ),
                    patch.object(
                        indexer_module,
                        "_MAX_SCORE_WORKSPACE_BYTES",
                        bytes_per_query * 7,
                    ),
                    patch.object(torch, "matmul", wraps=torch.matmul) as matmul,
                ):
                    actual = indexer.forward_causal_from_projected(
                        q,
                        raw,
                        cosine,
                        sine,
                        past if past_length else None,
                        precomputed_block_keys=keys if not past_length else None,
                    )
                self.assertGreater(matmul.call_count, 1)
                self.assertTrue(
                    all(call.args[0].shape[1] <= 7 for call in matmul.call_args_list)
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
