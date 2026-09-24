"""Prefill slots preserve CP ownership, physical pages, and STATE tail writes."""

import itertools
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_metadata as fused
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
    cp_kv_slot_mapping,
    cp_state_slot_mapping,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillMetadataTest(unittest.TestCase):
    def test_chunk_metadata_preserves_long_arithmetic(self):
        for batch in (2, 32, 128):
            for dtype in (torch.int32, torch.int64):
                p = ((torch.arange(batch * 2, device="cuda", dtype=dtype) - 2) * 513)[
                    ::2
                ]
                n = (torch.arange(batch * 2, device="cuda", dtype=dtype) * 137 + 1)[::2]
                req = (
                    (torch.arange(514, device="cuda", dtype=torch.int64) % batch)
                    - batch
                )[1::2]
                for ratio, replay, window in itertools.product(
                    (1, 2), (False, True), (1, 128)
                ):
                    prefix, length = p.long(), n.long()
                    tail = (
                        torch.zeros_like(prefix)
                        if replay
                        else prefix.clamp(max=window - 1)
                    )
                    sizes = (prefix + length) // ratio
                    chunks = sizes + length + tail
                    expected = (
                        (chunks.cumsum(0) - chunks)[req, None],
                        sizes[req, None],
                        (prefix - tail)[req, None],
                    )
                    actual = fused.try_chunk_metadata(p, n, req, ratio, window, replay)
                    self.assertIsNotNone(actual)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
        p = torch.tensor([2**31 - 1, 2**31 - 2], device="cuda", dtype=torch.int32)
        n = torch.tensor([101, 259], device="cuda", dtype=torch.int32)
        req = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        sizes = (p.long() + n.long()) // 2
        actual = fused.try_chunk_metadata(p, n, req, 2, 128, True)
        torch.testing.assert_close(actual[1][:, 0], sizes, rtol=0, atol=0)
        self.assertIsNone(fused.try_chunk_metadata(p, n.long(), req, 2, 128, True))

    def test_chunk_metadata_graph_and_empty_rows(self):
        p = torch.tensor([512, 16384, 30720], device="cuda", dtype=torch.int64)
        n = torch.tensor([1, 7, 8], device="cuda", dtype=torch.int64)
        req = torch.tensor([2, 1, 0], device="cuda", dtype=torch.int64)
        fused.try_chunk_metadata(p, n, req, 2, 128, False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused.try_chunk_metadata(p, n, req, 2, 128, False)
        p.add_(17)
        n.add_(3)
        req.copy_(req.flip(0))
        graph.replay()
        expected = fused.try_chunk_metadata(p, n, req, 2, 128, False)
        for got, want in zip(captured, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        empty = fused.try_chunk_metadata(p, n, req[:0], 2, 128, False)
        self.assertTrue(all(t.shape == (0, 1) for t in empty))

    def _inputs(self, dtype):
        # Strided vectors and tables, ragged prefixes, unallocated pages,
        # and positions beyond the final allocated physical page.
        pos = torch.arange(4097, device="cuda", dtype=dtype)[::2]
        req = (torch.arange(4097, device="cuda", dtype=dtype) % 3)[::2]
        table = torch.arange(3 * 37, device="cuda", dtype=dtype).view(3, 37)[:, :33]
        table[:, 0] = 0
        return pos, req, table

    def test_full_kv_physical_owner_and_kernel_page(self):
        for dtype in (torch.int32, torch.int64):
            pos, req, table = self._inputs(dtype)
            for ratio in (1, 2):
                positions = pos + (ratio - 1)
                for cp in (1, 2, 4):
                    for rank in range(cp):
                        for owner_tpb in (32, 256):
                            with self.subTest(
                                dtype=dtype,
                                ratio=ratio,
                                cp=cp,
                                rank=rank,
                                owner=owner_tpb,
                            ):
                                actual = fused.try_slot_mapping(
                                    positions,
                                    req,
                                    table,
                                    32 // ratio,
                                    32,
                                    ratio,
                                    cp,
                                    rank,
                                    owner_tokens_per_block=owner_tpb,
                                )
                                expected = cp_kv_slot_mapping(
                                    positions,
                                    table,
                                    req,
                                    32,
                                    32 // ratio,
                                    ratio,
                                    cp,
                                    rank,
                                    owner_tokens_per_block=owner_tpb,
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )

    def test_state_snapshot_and_per_block_tail(self):
        for dtype in (torch.int32, torch.int64):
            pos, req, table = self._inputs(dtype)
            for cp in (1, 2, 4):
                for rank in range(cp):
                    for ends in (
                        None,
                        torch.tensor([315, 511, 3001], device="cuda", dtype=dtype),
                    ):
                        actual = fused.try_slot_mapping(
                            pos,
                            req,
                            table,
                            4,
                            256,
                            2,
                            cp,
                            rank,
                            state=True,
                            seq_ends=ends,
                        )
                        expected = cp_state_slot_mapping(
                            pos, table, req, 4, 256, cp, rank, ends
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_state_owner_span_independent_of_checkpoint_span(self):
        for dtype in (torch.int32, torch.int64):
            positions = torch.arange(0, 2049, device="cuda", dtype=dtype)
            requests = positions % 3
            table = torch.arange(1, 16, device="cuda", dtype=dtype).reshape(3, 5)
            table[1, 1] = 0
            for rank in range(4):
                for end in (
                    None,
                    torch.tensor([513, 1024, 2049], device="cuda", dtype=dtype),
                ):
                    actual = fused.try_slot_mapping(
                        positions,
                        requests,
                        table,
                        3,
                        512,
                        2,
                        4,
                        rank,
                        owner_tokens_per_block=128,
                        state=True,
                        seq_ends=end,
                    )
                    self.assertIsNotNone(actual)
                    expected = cp_state_slot_mapping(
                        positions, table, requests, 3, 512, 4, rank, end
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertIsNone(
                fused.try_slot_mapping(
                    positions,
                    requests,
                    table,
                    64,
                    512,
                    2,
                    4,
                    0,
                    owner_tokens_per_block=128,
                    state=False,
                )
            )

    def test_bounds_clamp_before_int32_conversion(self):
        for ratio in (1, 2):
            positions = torch.tensor(
                [-3, -1, 0, 1, 511, 512, 524287, 2**40],
                device="cuda",
                dtype=torch.int64,
            )
            starts, ends = fused.try_score_bounds(positions, 524288, ratio)
            expected = ((positions + 1).clamp_min(0) // ratio).clamp_max(524288).int()
            torch.testing.assert_close(ends, expected, rtol=0, atol=0)
            self.assertEqual(starts.count_nonzero().item(), 0)

    def test_gate_and_graph_dynamic_metadata(self):
        pos, req, table = self._inputs(torch.int64)
        fused.try_slot_mapping(pos, req, table, 16, 32, 2, 4, 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fused.try_slot_mapping(pos, req, table, 16, 32, 2, 4, 1)
        pos.add_(1)
        table.add_(100)
        graph.replay()
        expected = cp_kv_slot_mapping(pos, table, req, 32, 16, 2, 4, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(fused.try_slot_mapping(pos.float(), req, table, 16, 32, 2))
        self.assertIsNone(fused.try_score_bounds(pos.float(), 1024, 2))

    def test_shared_bounds_and_direct_topk_output(self):
        rows, width = 19, 2049
        pos = torch.linspace(-1, width * 2, rows, device="cuda").long()
        bounds = fused.try_score_bounds(pos, width, 2)
        logits = torch.randn(rows, width + 127, device="cuda")[:, :width]
        logits.masked_fill_(
            torch.arange(width, device="cuda")[None] >= bounds[1][:, None], -torch.inf
        )
        backing = torch.full((rows + 2, 512), 999, device="cuda", dtype=torch.int32)
        out = backing[1:-1]
        actual = topk.try_select_tokens(logits, bounds[1], bounds=bounds, out=out)
        self.assertIs(actual, out)
        values = logits.gather(1, actual.long().clamp_min(0)).masked_fill(
            actual < 0, -torch.inf
        )
        torch.testing.assert_close(
            values.sort(-1).values,
            logits.topk(512).values.sort(-1).values,
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(backing[[0, -1]] == 999).item())


if __name__ == "__main__":
    unittest.main()
