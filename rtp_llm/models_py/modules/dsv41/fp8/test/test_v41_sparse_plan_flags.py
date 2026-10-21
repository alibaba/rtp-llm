"""Candidate membership reuse: dispatch, exact parity, dynamic graph inputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_decode as sparse
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as topk


class SparsePlanFlagsCPU(unittest.TestCase):
    def test_only_complete_engine_plan_omits_compatibility_flags(self):
        candidates = torch.empty(6, 512, dtype=torch.int32)
        lengths = torch.ones(6, dtype=torch.int32)
        table = torch.ones(2, 256, dtype=torch.int32)
        flags = torch.empty(6, 4096, dtype=torch.uint8)
        valid = torch.tensor([True, False])
        backend = SimpleNamespace(get_paged_sparse_mqa_logits_metadata=Mock())
        with patch.object(sparse, "deep_gemm", backend), \
             patch.object(sparse, "_prepare_prefix_plan_kernel", MagicMock()) as prefix, \
             patch.object(sparse, "_prepare_plan_kernel", MagicMock()) as sorter:
            for complete in (False, True):
                for mask in (None, valid):
                    plan = sparse.make_plan(
                        candidates, lengths, table, query_len=3,
                        candidate_mask=(candidates, flags, 8, "sorted_prefix") if complete else None,
                        valid_requests=mask,
                    )
                    self.assertEqual(plan.repair_rows is None, complete and mask is not None)
                    self.assertEqual(plan.lengths.data_ptr(), lengths.data_ptr())
                    kernel = prefix if complete else sorter
                    self.assertIs(kernel.__getitem__.return_value.call_args.kwargs["valid_requests"], mask)
            for bad in (valid.int(), valid[:1], valid[:, None]):
                with self.assertRaisesRegex(ValueError, "request validity"):
                    sparse.make_plan(candidates, lengths, table, query_len=3, valid_requests=bad)

    def test_sorted_prefix_requires_current_producer_provenance(self):
        candidates = torch.empty(6, 2048, dtype=torch.int32)
        flags = torch.empty(6, 4096, dtype=torch.uint8)
        lengths = torch.ones(6, dtype=torch.int32)
        table = torch.ones(1, 256, dtype=torch.int32)
        backend = SimpleNamespace(get_paged_sparse_mqa_logits_metadata=Mock())
        with patch.object(sparse, "deep_gemm", backend), \
             patch.object(sparse, "_prepare_prefix_plan_kernel", MagicMock()) as prefix, \
             patch.object(sparse, "_prepare_plan_kernel", MagicMock()) as sorter, \
             patch.object(sparse, "_prepare_flags_plan_kernel", MagicMock()) as compact:
            for mask, wanted in (
                ((candidates, flags, 8, "sorted_prefix"), True),
                ((candidates.clone(), flags, 8, "sorted_prefix"), False),
                ((candidates, flags, 16, "sorted_prefix"), False),
                ((candidates, flags, 8), False),
            ):
                prefix.reset_mock()
                sorter.reset_mock()
                compact.reset_mock()
                sparse.make_plan(candidates, lengths, table, query_len=6, candidate_mask=mask)
                self.assertEqual(prefix.__getitem__.call_count, int(wanted))
                self.assertEqual(sorter.__getitem__.call_count + compact.__getitem__.call_count, int(not wanted))

    def test_only_matching_bounded_flags_select_compaction(self):
        candidates = torch.empty(6, 2048, dtype=torch.int32)
        lengths = torch.ones(6, dtype=torch.int32)
        table = torch.ones(1, 265, dtype=torch.int32)
        flags = torch.empty(6, 4096, dtype=torch.uint8)
        cases = [
            ((candidates, flags, 8), True),
            (None, False),
            ((candidates.clone(), flags, 8), False),
            ((candidates, flags, 16), False),
            ((candidates, flags.int(), 8), False),
            ((candidates, flags[:3], 8), False),
            ((candidates, flags[:, ::2], 8), False),
            ((candidates, torch.empty(6, 4224, dtype=torch.uint8), 8), True),
            ((candidates, torch.empty(6, 4240, dtype=torch.uint8), 8), False),
        ]
        backend = SimpleNamespace(get_paged_sparse_mqa_logits_metadata=Mock())
        with patch.object(sparse, "deep_gemm", backend), \
             patch.object(sparse, "_prepare_plan_kernel", MagicMock()) as sorter, \
             patch.object(sparse, "_prepare_flags_plan_kernel", MagicMock()) as compact:
            for mask, wanted in cases:
                with self.subTest(wanted=wanted):
                    sorter.reset_mock()
                    compact.reset_mock()
                    sparse.make_plan(
                        candidates, lengths, table, query_len=6, candidate_mask=mask
                    )
                    self.assertEqual(compact.__getitem__.call_count, int(wanted))
                    self.assertEqual(sorter.__getitem__.call_count, int(not wanted))
            for k in (512, 1024):
                c = candidates[:, :k].contiguous()
                self.assertIsNone(sparse._plan_candidate_flags(c, table, (c, flags, 8)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparsePlanFlagsCUDA(unittest.TestCase):
    def test_compatibility_plans_also_honor_dummy_request_mask(self):
        query_len, k = 6, 2048
        rows = 2 * query_len
        candidates = torch.arange(k, device="cuda", dtype=torch.int32).expand(rows, k).clone()
        candidates[:, 512:] = -1
        if not sparse.is_supported(candidates):
            self.skipTest("SM10x and native Sparse MQA required")
        flags = torch.ones(rows, 512, device="cuda", dtype=torch.uint8)
        table = torch.ones(2, 32, device="cuda", dtype=torch.int32)
        table[1] = 0
        lengths = torch.full((rows,), 4093, device="cuda", dtype=torch.int32)
        valid = torch.tensor([True, False], device="cuda")
        for mask in (None, (candidates, flags, 8)):
            plan = sparse.make_plan(
                candidates, lengths, table, query_len=query_len,
                candidate_mask=mask, valid_requests=valid,
            )
            torch.testing.assert_close(plan.lengths, lengths)
            torch.testing.assert_close(plan.compact_lengths[:query_len], lengths[:query_len])
            self.assertTrue(bool((plan.compact_lengths[query_len:] == 0).all()))
            self.assertFalse(bool(plan.repair_rows[query_len:].any()))

    def test_dummy_compact_end_refreshes_without_mutating_context_lengths(self):
        batch, k, cols = 3, 512, 32
        for query_len in (1, 3, 6):
            rows = batch * query_len
            candidates = torch.arange(k, device="cuda", dtype=torch.int32).expand(rows, k).clone()
            if not sparse.is_supported(candidates):
                self.skipTest("SM10x and native Sparse MQA required")
            flags = torch.ones(rows, k, device="cuda", dtype=torch.uint8)
            lengths = torch.full((rows,), 4093, device="cuda", dtype=torch.int32)
            table = torch.ones(batch, cols, device="cuda", dtype=torch.int32)
            valid = torch.ones(batch, device="cuda", dtype=torch.bool)
            # Keep a nontrivial row stride; dummy bytes must never reach TopK.
            raw = torch.randn(rows, k * 8 + 512, device="cuda", dtype=torch.bfloat16)[:, :k * 8]
            original_lengths = lengths.clone()

            def run():
                plan = sparse.make_plan(
                    candidates, lengths, table, query_len=query_len,
                    candidate_mask=(candidates, flags, 8, "sorted_prefix"),
                    valid_requests=valid,
                )
                return plan, sparse.select_tokens(raw, plan, inplace=True)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan, output = run()
            end_ptr = plan.compact_lengths.data_ptr()
            for live in (3, 1, 0, 2, 3):
                valid.copy_(torch.arange(batch, device="cuda") < live)
                table.fill_(1)
                table[live:] = 0
                raw.normal_()
                raw[live * query_len:] = torch.nan
                graph.replay()
                self.assertIsNone(plan.repair_rows)
                self.assertEqual(plan.compact_lengths.data_ptr(), end_ptr)
                torch.testing.assert_close(plan.lengths, original_lengths)
                torch.testing.assert_close(
                    plan.compact_lengths,
                    torch.where(valid.repeat_interleave(query_len), original_lengths, 0),
                )
                self.assertTrue(bool((output[live * query_len:] == -1).all()))
                # Compare live rows with the retained copy/clean compatibility
                # path; compare values rather than arbitrary tied TopK IDs.
                reference = sparse.select_tokens(raw, plan)
                for row in range(live * query_len):
                    self.assertTrue(bool((output[row] >= 0).all()))
                    actual_scores = raw[row, output[row].long()].sort().values
                    expected_scores = raw[row, reference[row].long()].sort().values
                    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)

    def test_complete_prefix_matches_generic_plan_on_graph_replay(self):
        probe = torch.empty(6, 2048, device="cuda", dtype=torch.int32)
        if not sparse.is_supported(probe):
            self.skipTest("SM10x and native Sparse MQA required")
        source = torch.randn(6, 32768, device="cuda")
        lengths = torch.full((6,), 32768, device="cuda", dtype=torch.int32)
        table = torch.ones(1, 256, device="cuda", dtype=torch.int32)

        def run():
            candidates, flags = topk.select_candidates(source, lengths, 8, 2048, complete=True)
            return sparse.make_plan(
                candidates, lengths, table, query_len=6,
                candidate_mask=(candidates, flags, 8, "sorted_prefix"),
            ), candidates

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan, candidates = run()
        for mode in ("complete", "nonfinite", "short", "restored"):
            if mode == "nonfinite":
                source.fill_(-torch.inf)
                source[:, :128] = torch.nan
            elif mode == "short":
                lengths.copy_(torch.tensor([0, 1, 7, 8, 511, 513], device="cuda"))
                table[:, ::3] = 0
            elif mode == "restored":
                source.normal_()
                lengths.fill_(32768)
                table.fill_(1)
            graph.replay()
            reference = sparse.make_plan(candidates, lengths, table, query_len=6)
            for actual, expected in zip(plan[:4] + plan[5:], reference[:4] + reference[5:]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_exact_outputs_and_graph_replay(self):
        probe = torch.empty(6, 2048, device="cuda", dtype=torch.int32)
        if not sparse.is_supported(probe):
            self.skipTest("SM10x and native Sparse MQA required")
        torch.manual_seed(20261020)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for cols in (32, 128, 256, 264, 265):
                domain = cols * 16
                source = torch.randn(6, domain * 8, device="cuda")
                lengths = torch.full((6,), domain * 8, device="cuda", dtype=torch.int32)
                table = torch.arange(cols + 3, device="cuda", dtype=torch.int32)[None, :cols]

                def run():
                    candidates, flags = topk.select_candidates(source, lengths, 8, 2048)
                    return sparse.make_plan(
                        candidates, lengths, table, query_len=6,
                        candidate_mask=(candidates, flags, 8),
                    ), candidates

                run()  # DG workspace/JIT warm-up on the capture stream.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    plan, candidates = run()
                for mode in ("complete", "holes", "empty", "short", "restored"):
                    if mode == "holes":
                        source[:, ::2] = torch.nan
                        source[:, :domain * 4] = -torch.inf
                    elif mode == "empty":
                        source.fill_(-torch.inf)
                    elif mode == "short":
                        lengths.copy_(torch.tensor([0, 1, 7, 8, 511, 513], device="cuda"))
                        table[:, ::3] = -1
                    elif mode == "restored":
                        source.normal_()
                        lengths.fill_(domain * 8)
                    graph.replay()
                    reference = sparse.make_plan(candidates, lengths, table, query_len=6)
                    # Metadata scheduling order can vary; compare its exact
                    # inputs rather than requiring identical scheduler bytes.
                    for actual, expected in zip(plan[:4], reference[:4]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for actual, expected in zip(plan[5:], reference[5:]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    unittest.main()
