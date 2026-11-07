"""Exercise packed request ownership, padding, gate rounding and graph refresh."""

import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_layout_triton import (
    try_pack_inputs,
    try_unpack_output,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_metadata_triton import draft_rows


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DraftQsaLayoutTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(107920)

    @staticmethod
    def reference_pack(qkv, selected, sources, visible):
        query = qkv[:, :3072].index_select(0, sources).view(-1, 4, 12, 256)
        selected = selected.index_select(0, sources).view(-1, 4, 2051)
        return query.transpose(1, 2).contiguous(), torch.where(
            visible.unsqueeze(-1) > 0, selected, -1
        )

    @staticmethod
    def reference_unpack(value, inverse, valid, gate):
        result = value.transpose(1, 2).reshape(-1, 3072).index_select(0, inverse)
        result = torch.where(valid[:, None], result, 0)
        return result if gate is None else result * torch.sigmoid(gate)

    def test_ragged_ownership_and_two_graphs_refresh_all_metadata(self):
        for batch in (1, 3, 8):
            rows = batch * 4
            qkv = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)[:, :3584]
            selection = torch.randint(
                -1, 8192, (rows, 2059), device="cuda", dtype=torch.int32
            )[:, :2051]
            lengths = torch.full((batch,), 4, device="cuda", dtype=torch.int32)
            prefixes = torch.full_like(lengths, 127)
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
            value = torch.randn(
                batch, 4, 12, 256, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
            gate = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
            graphs = []
            for gated in (False, True):

                def call():
                    sources, inverse, valid, _, visible = draft_rows(
                        lengths, prefixes, cu, rows
                    )
                    packed = try_pack_inputs(
                        qkv, selection, sources, visible, heads=12, head_dim=256
                    )
                    unpacked = try_unpack_output(
                        value, inverse, valid, gate=gate if gated else None
                    )
                    assert packed is not None and unpacked is not None
                    return (*packed, unpacked)

                call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = call()
                graphs.append((gated, graph, actual))
            for step in range(32):
                lengths.copy_(
                    (torch.arange(batch, device="cuda", dtype=torch.int32) + step) % 5
                )
                cu[0] = 0
                cu[1:].copy_(lengths.cumsum(0))
                qkv.normal_()
                selection.random_(-1, 8192)
                value.normal_()
                gate.normal_()
                sources, inverse, valid, _, visible = draft_rows(
                    lengths, prefixes, cu, rows
                )
                query, selected = self.reference_pack(qkv, selection, sources, visible)
                for gated, graph, actual in graphs:
                    graph.replay()
                    expected = (
                        query,
                        selected,
                        self.reference_unpack(
                            value, inverse, valid, gate if gated else None
                        ),
                    )
                    for got, ref in zip(actual, expected):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)
            for _, graph, _ in graphs:
                graph.reset()

    def test_every_bfloat16_gate_bits_including_nonfinite_and_padding(self):
        values = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        gate = values.repeat(2)[: 32 * 3072].reshape(32, 3072)
        value = torch.randn(8, 12, 4, 256, device="cuda", dtype=torch.bfloat16)
        inverse = torch.randperm(32, device="cuda")
        for padding in (False, True):
            valid = torch.full((32,), not padding, device="cuda", dtype=torch.bool)
            actual = try_unpack_output(value, inverse, valid, gate=gate)
            expected = self.reference_unpack(value, inverse, valid, gate)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            finite = ~torch.isnan(expected)
            self.assertTrue(
                torch.equal(
                    actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
                )
            )

    def test_unqualified_layouts_fall_back(self):
        qkv = torch.zeros(4, 3584, device="cuda", dtype=torch.bfloat16)
        selected = torch.zeros(4, 2051, device="cuda", dtype=torch.int32)
        sources = torch.arange(4, device="cuda")
        visible = torch.ones(1, 4, device="cuda", dtype=torch.int32)
        self.assertIsNone(
            try_pack_inputs(qkv, selected, sources, visible, heads=6, head_dim=256)
        )
        self.assertIsNone(
            try_pack_inputs(
                qkv, selected, sources, visible.long(), heads=12, head_dim=256
            )
        )
        self.assertIsNone(
            try_pack_inputs(
                qkv[:, ::2], selected, sources, visible, heads=12, head_dim=256
            )
        )


if __name__ == "__main__":
    unittest.main()
