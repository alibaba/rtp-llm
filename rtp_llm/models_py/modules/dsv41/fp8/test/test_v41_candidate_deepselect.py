"""Sorted K2048 candidates, lazy compatibility flags and replay bounds.

GPU tests require rebuilding the new native candidate binding. No timing claims.
"""

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_candidate_deepselect as native
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as selector
from rtp_llm.ops.compute_ops import rtp_llm_ops


class CandidateAdmissionTest(unittest.TestCase):
    def test_cpu_does_not_query_cuda(self):
        scores = torch.empty((2, 32768), dtype=torch.bfloat16)
        lengths = torch.zeros(2, dtype=torch.int32)
        with mock.patch.object(native, "_device_supported") as query:
            self.assertFalse(native.is_supported(scores, lengths, 8, 2048))
        query.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CandidateDeepSelectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise unittest.SkipTest("SM100/SM103 required")
        available = getattr(rtp_llm_ops, "deepselect_bf16_available", None)
        if (
            not callable(available)
            or not available()
            or not callable(getattr(rtp_llm_ops, "deepselect_bf16_candidates", None))
        ):
            raise unittest.SkipTest("Rebuild the native candidate binding first")

    def assert_prefix(self, output, ends, width):
        for row, end in enumerate(ends.cpu().clamp(0, width).tolist()):
            count = min(end, 2048)
            ids = output[row].cpu().long()
            self.assertTrue(bool(((ids[:count] >= 0) & (ids[:count] < end)).all()))
            self.assertTrue(bool((ids[1:count] > ids[: max(count - 1, 0)]).all()))
            self.assertTrue(bool((ids[count:] == -1).all()))
            if count:
                self.assertEqual(ids[count - 1].item(), end - 1)

    def test_metadata_fallbacks_and_native_errors(self):
        scores = torch.zeros((2, 32768), dtype=torch.bfloat16, device="cuda")
        ends = torch.full((2,), 32768, dtype=torch.int32, device="cuda")
        with mock.patch.dict(os.environ, {"DSV41_DECODE_CANDIDATE_DEEPSELECT": "1"}):
            self.assertTrue(native.is_supported(scores, ends, 8, 2048))
            self.assertFalse(native.is_supported(scores.float(), ends, 8, 2048))
            self.assertFalse(native.is_supported(scores, ends, 8, 1024))
            self.assertFalse(native.is_supported(scores, ends.long(), 8, 2048))
            with mock.patch.object(rtp_llm_ops, "deepselect_bf16_candidates", None):
                self.assertFalse(native.is_supported(scores, ends, 8, 2048))
        with mock.patch.dict(os.environ, {"DSV41_DECODE_CANDIDATE_DEEPSELECT": "0"}):
            self.assertFalse(native.is_supported(scores, ends, 8, 2048))
        with mock.patch.object(
            rtp_llm_ops,
            "deepselect_bf16_candidates",
            side_effect=RuntimeError("native failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                native.select(scores, ends)

    def test_padding_nonfinite_and_runtime_k_smaller_than_capacity(self):
        # Both initial-window and permuted/reconstruct paths. Invalid allocator
        # contents must not enter ties with real -INF values or become indices.
        for width in (2560, 4096, 16896, 32768):
            ends = torch.tensor(
                [0, 1, 2047, 2048, 2049, width - 1, width],
                device="cuda",
                dtype=torch.int32,
            )
            for value in (-float("inf"), float("inf"), float("nan"), -3.0):
                with self.subTest(width=width, value=value):
                    scores = torch.full(
                        (7, width), value, dtype=torch.bfloat16, device="cuda"
                    )
                    # Poison the masked part, even when it shares a TMA tile.
                    columns = torch.arange(width, device="cuda")
                    scores.masked_fill_(columns[None] >= ends[:, None], float("nan"))
                    self.assert_prefix(native.select(scores, ends), ends, width)

    def test_one_incomer_does_not_select_reconstruct_padding(self):
        # The first 24 permuted 512-element segments share the 16K init
        # window with the tail. Put one hit in the next segment: the final
        # reconstruction must process exactly one incoming pair plus padding.
        width, end, perm_segments = 32768, 32767, 56
        segment = (24 * (0xB559EB75 % perm_segments) + 0x22262226) % perm_segments
        hit = segment * 512
        scores = torch.full((1, width), -torch.inf, device="cuda", dtype=torch.bfloat16)
        scores[0, hit] = 1
        scores[0, end - 1] = torch.inf
        scores[0, end:] = torch.nan
        ends = torch.tensor([end], device="cuda", dtype=torch.int32)
        output = native.select(scores, ends)
        self.assert_prefix(output, ends, width)
        self.assertTrue(bool((output == hit).any()))
        self.assertEqual(int((scores[0, output[0].long()] > -torch.inf).sum()), 2)

    def test_score_multiset_and_last_block_pin(self):
        torch.manual_seed(2048)
        scores = torch.randn((4, 32768), dtype=torch.bfloat16, device="cuda")
        ends = torch.tensor(
            [2049, 4097, 16385, 32767], dtype=torch.int32, device="cuda"
        )
        for row, end in enumerate(ends.tolist()):
            scores[row, end - 1] = float("inf")
        output = native.select(scores, ends)
        self.assert_prefix(output, ends, scores.shape[1])
        for row, end in enumerate(ends.tolist()):
            actual = scores[row, output[row].long()].float().sort().values
            expected = scores[row, :end].float().topk(2048).values.sort().values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pool_dispatch_has_no_prefix_or_eager_flags_and_replays(self):
        logits = torch.randn((4, 32768), device="cuda", dtype=torch.bfloat16)
        lengths = torch.full((4,), 32768, device="cuda", dtype=torch.int32)
        with mock.patch.dict(
            os.environ, {"DSV41_DECODE_CANDIDATE_DEEPSELECT": "1"}
        ), mock.patch.object(
            selector, "_candidate_store_prefix_kernel"
        ) as legacy, mock.patch.object(
            selector, "build_candidate_flags"
        ) as eager_flags:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    selector.select_candidates(
                        logits, lengths, 8, 2048, complete=True, build_flags=False
                    )
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                output, flags = selector.select_candidates(
                    logits, lengths, 8, 2048, complete=True, build_flags=False
                )
            ptr = output.data_ptr()
            self.assertIsNone(flags)
            for bounds in ([0, 1, 16385, 32767], [32768, 0, 17, 16384]):
                lengths.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
                logits.neg_()
                graph.replay()
                self.assertEqual(output.data_ptr(), ptr)
                self.assert_prefix(output, (lengths + 7) // 8, 4096)
                # Length checks alone cannot detect a graph using stale scores.
                cpu_logits = logits.cpu().float()
                for row, bound in enumerate(bounds):
                    visible = (bound + 7) // 8
                    count = min(visible, 2048)
                    cpu_logits[row, bound:] = -torch.inf
                    pooled = cpu_logits[row].view(4096, 8).amax(1)
                    if visible:
                        pooled[visible - 1] = torch.inf
                    ids = output[row, :count].cpu().long()
                    torch.testing.assert_close(
                        pooled[ids].sort().values,
                        pooled[:visible].topk(count).values.sort().values,
                        rtol=0,
                        atol=0,
                    )
            legacy.__getitem__.assert_not_called()
            eager_flags.assert_not_called()
        # A dense fallback can materialize its bitmap without sorting again.
        flags_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            selector.build_candidate_flags(output, logits.shape[1], 8)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(flags_graph, stream=side):
            bitmap = selector.build_candidate_flags(output, logits.shape[1], 8)
        flags_graph.replay()
        for row in range(output.shape[0]):
            ids = output[row][output[row] >= 0].long()
            self.assertEqual(bitmap[row].sum().item(), ids.numel())
            self.assertTrue(bool((bitmap[row, ids] == 1).all()))
        output.fill_(-1)
        flags_graph.replay()
        self.assertEqual(bitmap.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
