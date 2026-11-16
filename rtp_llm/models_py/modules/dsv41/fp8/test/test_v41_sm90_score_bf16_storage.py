"""Exact score storage, causal bounds and graph replay on packed-FP4 SM90."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as decode
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_indexer as prefill
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as prefill_topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_decode_candidate as candidate
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_indexer as scorer
from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import quantize_rows_fp4
from test_v41_sm90_decode_candidate import make_case


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90ScoreBF16StorageTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")

    def score(self, case, **kwargs):
        return scorer.paged_score(
            case["q"], case["q_scale"], case["pool"], case["weights"],
            case["table"], case["lengths"], case["capacity"],
            case["logical_entries"], **kwargs,
        )

    def check_scores_and_ids(self, actual, expected, case, padded):
        lengths = case["lengths"].clamp(0, case["capacity"])
        mask = torch.arange(case["capacity"], device="cuda")[None] < lengths[:, None]
        torch.testing.assert_close(actual.float()[mask], expected[mask], rtol=0, atol=0)
        if padded:
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            topk.select_tokens(actual, lengths, 512),
            topk.select_tokens(expected, lengths, 512), rtol=0, atol=0,
        )

    def test_paged_bf16_masks_ties_candidates_and_fp4_cache_unchanged(self):
        for physical in (64, 128):
            for ratio in (1, 2):
                with self.subTest(physical=physical, ratio=ratio):
                    c, _ = make_case(b=16, s=4, capacity=16391, physical=physical, ratio=ratio)
                    c["lengths"].copy_(torch.linspace(-3, 16400, 64, device="cuda").int())
                    c["table"][0, 0] = 0
                    c["table"][1, 1] = -1
                    c["weights"][2].zero_()
                    cache_before = c["pool"].clone()
                    expected = self.score(c)
                    self.assertEqual(expected.dtype, torch.float32)
                    for dtype in (torch.float32, torch.bfloat16):
                        for padded in (True,):
                            actual = self.score(c, output_dtype=dtype)
                            self.check_scores_and_ids(actual, expected, c, padded)
                            lengths = c["lengths"].clamp(0, c["capacity"])
                            actual_groups = topk.select_candidates(actual, lengths, 8, 512)
                            expected_groups = topk.select_candidates(expected, lengths, 8, 512)
                            for actual_group, expected_group in zip(actual_groups, expected_groups):
                                torch.testing.assert_close(actual_group, expected_group, rtol=0, atol=0)
                    candidate_expected = candidate.try_score(**c)
                    candidate_actual = candidate.try_score(**c, output_dtype=torch.bfloat16)
                    self.assertEqual(candidate_actual.dtype, torch.bfloat16)
                    self.check_scores_and_ids(candidate_actual, candidate_expected, c, True)
                    self.assertEqual(c["pool"].dtype, torch.uint8)
                    self.assertEqual(c["q_scale"].dtype, torch.int32)
                    self.assertEqual(c["pool"].shape[-1], 68)
                    self.assertTrue(torch.equal(c["pool"], cache_before))

    def test_one_million_graph_dynamic_lengths(self):
        c, _ = make_case(b=16, s=4, capacity=2053, physical=128, ratio=2)
        table = torch.zeros((16, 1048576 // 64), device="cuda", dtype=torch.int32)
        table[:, :c["table"].shape[1]].copy_(c["table"])
        c.update(table=table, capacity=1048576)
        fn = lambda: self.score(c, output_dtype=torch.bfloat16)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fn()
        self.assertEqual(actual.numel() * actual.element_size(), 128 * 1024**2)
        for length in (0, 1, 513, 1048576, 17):
            c["lengths"].fill_(length)
            graph.replay()
            self.check_scores_and_ids(actual, self.score(c), c, True)

    def test_candidate_graph_changes_payload_and_metadata(self):
        c, raw_q = make_case(b=2, s=4, capacity=2053, ratio=2, blocks=2048)
        table = torch.zeros((2, 1048576 // 64), device="cuda", dtype=torch.int32)
        table[:, :c["table"].shape[1]].copy_(c["table"])
        c.update(table=table, capacity=1048576)

        def run(dtype):
            q, sf = quantize_rows_fp4(raw_q)
            current = c | {"q": q.view(2, 4, 32, 64), "q_scale": sf.view(2, 4, 32)}
            logits = candidate.try_score(**current, output_dtype=dtype)
            return logits, topk.select_tokens(logits, c["lengths"], 512)

        for _ in range(3):
            run(torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_ids = run(torch.bfloat16)
        self.assertEqual(captured.dtype, torch.bfloat16)
        for iteration in range(3):
            raw_q.copy_(torch.randn_like(raw_q) * (iteration + 1) / 4)
            c["weights"].copy_(torch.rand_like(c["weights"]) / 64)
            c["table"].copy_(c["table"].flip(-1))
            c["lengths"].copy_(torch.tensor(
                [0, 1, 511, 512, 513, 2053, 32768, 1048576],
                device="cuda", dtype=torch.int32,
            ))
            c["candidates"].random_(-3, 4096)
            if iteration == 0:
                c["candidates"].fill_(-1)
            graph.replay()
            original, original_ids = run(torch.float32)
            torch.testing.assert_close(captured.float(), original, rtol=0, atol=0)
            torch.testing.assert_close(captured_ids, original_ids, rtol=0, atol=0)

    def test_decode_existing_options_are_forwarded(self):
        c, _ = make_case(b=1, s=3, capacity=2053, physical=128, ratio=2)
        with patch.object(decode, "is_supported", return_value=True), patch.object(
            decode, "prepare_indexer_q", return_value=(c["q"], c["q_scale"])
        ):
            args = (c["q"], c["weights"], None, c["pool"], c["table"], c["lengths"])
            kwargs = dict(max_ctx_len=c["capacity"], logical_entries_per_block=c["logical_entries"])
            baseline = decode.score_decode_indexer(*args, **kwargs)
            self.assertEqual(baseline.dtype, torch.float32)
            actual = decode.score_decode_indexer(
                *args, **kwargs, preserve_score_dtype=True, pad_invisible=False
            )
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.check_scores_and_ids(actual, baseline, c, False)
            actual = decode.score_decode_indexer(
                *args, **kwargs, preserve_score_dtype=True, candidate_blocks=c["candidates"]
            )
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.check_scores_and_ids(actual, candidate.try_score(**c), c, True)

    def test_prefill_bf16_fixed_chunk_and_two_query_path(self):
        self.check_prefill(17, 2053)
        self.check_prefill(2048, 32768)

    def check_prefill(self, rows, width):
        torch.manual_seed(261026)
        if width == 32768:
            self.assertEqual(prefill.logits_chunk_rows(width), rows)
        q, qs = quantize_rows_fp4(torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * .25)
        k, ks = quantize_rows_fp4(torch.randn(width, 128, device="cuda", dtype=torch.bfloat16) * .25)
        weights = torch.rand(rows, 32, device="cuda", dtype=torch.bfloat16) / 64
        starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
        ends = torch.linspace(0, width, rows, device="cuda").int()
        starts[::3] = ends[::3] // 2
        weights[::7] = 0
        args = (q, qs, k, ks, weights, ends)
        baseline = prefill.score_indexer_chunk(*args, bounds=(starts, ends))
        actual = prefill.score_indexer_chunk(*args, bounds=(starts, ends), return_bf16=True)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.numel() * actual.element_size(), baseline.numel() * baseline.element_size() // 2)
        torch.testing.assert_close(actual.float(), baseline, rtol=0, atol=0)
        actual_ids = prefill_topk.try_select_tokens(actual, ends, bounds=(starts, ends))
        baseline_ids = prefill_topk.try_select_tokens(baseline, ends, bounds=(starts, ends))
        self.assertIsNotNone(actual_ids)
        self.assertIsNotNone(baseline_ids)
        torch.testing.assert_close(actual_ids, baseline_ids, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
