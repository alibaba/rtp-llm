"""Packed draft addressing must preserve grouped attention and cache padding."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_layout_triton import (
    try_pack_inputs,
    try_restore_selection,
    try_unpack_output,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_metadata_triton import draft_rows
from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_paged_fmha import (
    _merge_draft_metadata_splits,
    try_draft_paged_attention,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
    _merge_sparse_paged_splits_kernel,
    sparse_paged_gqa_attn,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DraftPagedFmhaTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {
                "QWEN4_DRAFT_QSA_LAYOUT": "1",
                "QWEN4_DRAFT_QSA_DIRECT": "1",
                "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(20261020)

    def test_two_graphs_ragged_padded_pools_live_pages_and_scales(self):
        for batch in (1, 2, 3, 4, 8):
            rows, pages = batch * 4, batch * 4 + 1
            qkv = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)[:, :3584]
            selected = torch.randint(
                -1, 512, (rows, 2064), device="cuda", dtype=torch.int32
            )[:, :2051]
            lengths = torch.full((batch,), 4, device="cuda", dtype=torch.int32)
            prefixes = torch.full_like(lengths, 127)
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
            table = torch.arange(1, pages, device="cuda", dtype=torch.int32).reshape(
                batch, 4
            )
            original_table = table.clone()
            cache = (torch.randn(pages, 65536 + 128, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            scale = torch.rand(pages, 256 + 16, device="cuda") + 0.1
            gate = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
            initial = cache.view(torch.uint8).clone()

            def call(candidate, gated):
                sources, inverse, valid, _, visible = draft_rows(
                    lengths, prefixes, cu, rows
                )
                if candidate:
                    packed = try_restore_selection(selected, inverse, valid)
                else:
                    packed = torch.where(
                        valid[:, None], selected.index_select(0, inverse), -1
                    )
                if candidate:
                    value = try_draft_paged_attention(
                        qkv,
                        packed,
                        lengths,
                        prefixes,
                        cu,
                        cache,
                        table,
                        heads=12,
                        kv_heads=1,
                        head_dim=256,
                        page_size=128,
                        kv_scale=scale,
                        gate=gate if gated else None,
                    )
                    self.assertIsNotNone(value)
                else:
                    query, indices = try_pack_inputs(
                        qkv, packed, sources, visible, heads=12, head_dim=256
                    )
                    value = sparse_paged_gqa_attn(
                        query,
                        cache,
                        table,
                        visible,
                        indices,
                        page_size=128,
                        kv_head_num=1,
                        kv_scale=scale,
                        graph_capture=True,
                    )
                    value = try_unpack_output(
                        value, inverse, valid, gate=gate if gated else None
                    )
                return packed, value

            graphs = []
            for gated in (False, True):
                call(True, gated)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = call(True, gated)
                graphs.append((gated, graph, actual))
            for step in range(32):
                lengths.copy_(
                    (torch.arange(batch, device="cuda", dtype=torch.int32) + step) % 5
                )
                prefixes.copy_(
                    125
                    + (torch.arange(batch, device="cuda", dtype=torch.int32) + step) % 7
                )
                cu[0] = 0
                cu[1:].copy_(lengths.cumsum(0))
                table.copy_(original_table.flip(1) if step % 2 else original_table)
                if step % 3 == 0:
                    table[:, 1] = 0
                if step % 5 == 0:
                    table[:, 2] = pages + 3
                qkv.normal_()
                selected.random_(-1, 512)
                scale.uniform_(0.1, 1.0)
                gate.normal_()
                for gated, graph, actual in graphs:
                    expected = call(False, gated)
                    graph.replay()
                    for got, ref in zip(actual, expected):
                        self.assertTrue(
                            torch.equal(
                                got.contiguous().view(torch.uint8),
                                ref.contiguous().view(torch.uint8),
                            )
                        )
                self.assertTrue(torch.equal(cache.view(torch.uint8), initial))
            for _, graph, _ in graphs:
                graph.reset()

    def test_merge_every_gate_bit_and_empty_partitions(self):
        gate = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
            .repeat(2)[: 32 * 3072]
            .reshape(32, 3072)
        )
        lengths = torch.tensor(
            [4, 0, 4, 3, 4, 3, 4, 4], device="cuda", dtype=torch.int32
        )
        cu = torch.cat(
            (torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0).int())
        )
        _, inverse, valid, _, _ = draft_rows(lengths, torch.zeros_like(lengths), cu, 32)
        for splits in (8, 32):
            partial = torch.randn(8, 12, 4, splits, 258, device="cuda")
            partial[..., 257].abs_()
            partial[:, :, 1, :, 257] = 0
            partial[:, :, 1, :, 256] = float("-inf")
            baseline = torch.empty(8, 12, 4, 256, device="cuda", dtype=torch.bfloat16)
            _merge_sparse_paged_splits_kernel[(32 * 12,)](
                partial,
                baseline,
                baseline,
                D=256,
                NUM_SPLITS=splits,
                HEADS=12,
                QUERIES=4,
                WITH_GATE=False,
            )
            expected = try_unpack_output(baseline, inverse, valid, gate=gate)
            actual = torch.empty_like(expected)
            _merge_draft_metadata_splits[(32, 12)](
                partial, cu, gate, actual, BATCH=8, RB=8, SPLITS=splits, WITH_GATE=True
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            finite = ~torch.isnan(expected)
            self.assertTrue(
                torch.equal(
                    actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
                )
            )

    def test_unqualified_metadata_and_gate_fall_back(self):
        qkv = torch.zeros(4, 3584, device="cuda", dtype=torch.bfloat16)
        selected = torch.zeros(4, 2051, device="cuda", dtype=torch.int32)
        sources = torch.arange(4, device="cuda")
        valid = torch.ones(4, device="cuda", dtype=torch.bool)
        visible = torch.ones(1, 4, device="cuda", dtype=torch.int32)
        cache = torch.zeros(3, 65536, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.ones(3, 256, device="cuda")
        table = torch.ones(1, 2, device="cuda", dtype=torch.int32)
        lengths = torch.tensor([4], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
        args = (qkv, selected, lengths, lengths, cu, cache, table)
        kwargs = dict(heads=12, kv_heads=1, head_dim=256, page_size=128, kv_scale=scale)
        for change in (
            {"heads": 24},
            {"kv_scale": scale.double()},
            {"gate": qkv[:, :1]},
            {"page_size": 64},
        ):
            self.assertIsNone(
                try_draft_paged_attention(*args, **dict(kwargs, **change))
            )
        self.assertIsNone(try_restore_selection(selected.long(), sources, valid))
        self.assertIsNone(try_restore_selection(selected, sources.int(), valid))
        with patch.dict(os.environ, {"QWEN4_DRAFT_QSA_DIRECT": "0"}):
            self.assertIsNone(try_draft_paged_attention(*args, **kwargs))
            self.assertIsNone(try_restore_selection(selected, sources, valid))


if __name__ == "__main__":
    unittest.main()
