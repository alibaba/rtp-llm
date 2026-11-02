"""Fused draft addressing/norm keeps all old preparation outputs byte exact."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_metadata_triton import draft_rows
from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_prepare_triton import (
    try_prepare,
    try_prepare_metadata,
)
from rtp_llm.ops import RopeStyle


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DraftQsaMetadataTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 measured domain")
        torch.manual_seed(20261020)

    def test_live_metadata_raw_gains_padded_rows_and_two_graphs(self):
        frequency = 10000000.0 ** (
            -2.0 * torch.arange(32, device="cuda", dtype=torch.float32) / 64
        )
        for batch in range(1, 9):
            rows = batch * 4
            q = torch.randn(rows, 600, device="cuda", dtype=torch.bfloat16)[
                :, :512
            ].view(rows, 4, 128)
            lengths = torch.full((batch,), 4, device="cuda", dtype=torch.int32)
            prefixes = torch.full_like(lengths, 4093)
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
            gamma = torch.randn(
                128, device="cuda", dtype=torch.bfloat16 if batch % 2 else torch.float32
            )
            rope = SimpleNamespace(
                style=RopeStyle.Base,
                index_factor=3,
                dim=64,
                scale=(1.0, 0.5, 2.0)[batch % 3],
            )
            graphs = []
            for raw in (False, True):

                def call():
                    return try_prepare_metadata(
                        q,
                        lengths,
                        prefixes,
                        cu,
                        frequency,
                        rope_config=rope,
                        q_gamma=gamma if raw else None,
                    )

                self.assertIsNotNone(call())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = call()
                graphs.append((raw, graph, actual))
            for generation in range(32):
                requests = torch.arange(batch, device="cuda", dtype=torch.int32)
                lengths.copy_((requests + generation) % 5)
                cu[0] = 0
                cu[1:].copy_(lengths.cumsum(0))
                prefixes.copy_(
                    requests * 13
                    + (0, 125, 8188, 2**24 + 27, 2**30 + 3)[generation % 5]
                )
                q.normal_(0, (0.0, 0.001, 1.0, 10.0)[generation % 4])
                gamma.normal_()
                sources, _, _, positions, visible = draft_rows(
                    lengths, prefixes, cu, rows
                )
                for raw, graph, actual in graphs:
                    query = exact_head_rms_norm(q, gamma, 1e-6) if raw else q
                    expected = try_prepare(
                        query, positions, sources, visible, frequency, rope_config=rope
                    ) + (visible,)
                    graph.replay()
                    for got, ref in zip(actual, expected):
                        self.assertTrue(
                            torch.equal(
                                got.contiguous().view(torch.uint8),
                                ref.contiguous().view(torch.uint8),
                            ),
                            (batch, generation, raw),
                        )
            for _, graph, _ in graphs:
                graph.reset()

    def test_unqualified_metadata_norm_and_rope_fall_back(self):
        q = torch.randn(4, 4, 128, device="cuda", dtype=torch.bfloat16)
        lengths = torch.tensor([4], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
        frequency = torch.ones(32, device="cuda")
        gamma = torch.ones(128, device="cuda")
        rope = SimpleNamespace(style=RopeStyle.Base, index_factor=3, dim=64, scale=1.0)
        args = (q, lengths, lengths, cu, frequency)
        for kwargs in (
            {"q_gamma": gamma.long()},
            {"q_gamma": gamma.cpu()},
            {"q_gamma": gamma, "norm_eps": float("nan")},
            {"q_gamma": gamma, "norm_eps": -1.0},
            {"rope_config": SimpleNamespace(**dict(vars(rope), dim=32))},
        ):
            self.assertIsNone(
                try_prepare_metadata(*args, **dict({"rope_config": rope}, **kwargs))
            )
        self.assertIsNone(
            try_prepare_metadata(
                q, lengths.long(), lengths, cu, frequency, rope_config=rope
            )
        )
        self.assertIsNone(
            try_prepare_metadata(
                q, lengths, lengths, cu[:1], frequency, rope_config=rope
            )
        )
        self.assertIsNone(
            try_prepare_metadata(q, lengths, lengths, cu, None, rope_config=rope)
        )


if __name__ == "__main__":
    unittest.main()
