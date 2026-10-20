"""Candidate membership reuse: dispatch, exact parity, dynamic graph inputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_decode as sparse
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as topk


class SparsePlanFlagsCPU(unittest.TestCase):
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
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    unittest.main()
