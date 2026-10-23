"""Exact owner projections against the established 4096-row cuBLAS path."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8._v41_compact_cp_producer import (
    produce_owner_rows,
    project_owner,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32
from rtp_llm.ops.compute_ops import rtp_llm_ops


class CompactCpProjectionTest(unittest.TestCase):
    def setUp(self):
        self.kind = rtp_llm_ops.V41CompactCpProjection
        self.assertTrue(self.kind.supported(torch.cuda.current_device()))
        self.op = self.kind(torch.cuda.current_device())

    def exact(self, expected, actual):
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(
            torch.equal(
                expected.contiguous().view(torch.uint8),
                actual.contiguous().view(torch.uint8),
            )
        )

    @staticmethod
    def indices(rank):
        return torch.cat(
            (
                torch.arange(rank * 512, (rank + 1) * 512, device="cuda"),
                torch.arange((7 - rank) * 512, (8 - rank) * 512, device="cuda"),
            )
        )

    def check_owners(self, x, weight, repeats=8):
        expected = _linear_bf16_bf16_fp32(x, weight)
        for rank in range(4):
            indices = self.indices(rank)
            local = x.index_select(0, indices)
            reference = expected.index_select(0, indices)
            for _ in range(repeats):
                self.exact(reference, self.op.forward(local, weight))

    def test_random_and_adversarial(self):
        torch.manual_seed(8947)
        x = torch.randn(4096, 5120, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(512, 5120, device="cuda", dtype=torch.bfloat16)
        self.check_owners(x, weight)
        self.check_owners(torch.zeros_like(x), weight)
        # Equal positive/negative terms expose partition and reduction-order changes.
        x[:, 1::2] = x[:, ::2]
        weight[:, 1::2] = -weight[:, ::2]
        self.check_owners(x, weight)
        self.check_owners(x * (2.0**-120), weight * (2.0**-8))
        self.check_owners(x * (2.0**80), weight * (2.0**-60))
        sparse = torch.zeros_like(x)
        sparse[:, ::127] = x[:, ::127]
        self.check_owners(sparse, weight)

    def test_captured_projections(self):
        if not os.environ.get("DSV41_CP_FIXTURES"):
            self.skipTest("Captured-operand qualification requires DSV41_CP_FIXTURES")
        root = Path(os.environ["DSV41_CP_FIXTURES"])
        paths = sorted(root.glob("layer*-chunk*.pt"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            fixture = torch.load(path, map_location="cuda", weights_only=True)
            weights = torch.load(
                root / f'weights-layer{fixture["layer"]}.pt',
                map_location="cuda",
                weights_only=True,
            )
            self.check_owners(
                fixture["tensors"]["x"], weights["global_wkv"], repeats=12
            )
            if fixture["ratio"] == 2:
                self.check_owners(
                    fixture["tensors"]["x"], weights["global_wgate"], repeats=12
                )
            owner = SimpleNamespace(**weights)
            tensors, ratio = fixture["tensors"], fixture["ratio"]
            for rank in range(4):
                indices = self.indices(rank)
                boundary = indices[ratio - 1 :: ratio] // ratio
                values, scores = project_owner(
                    self.op, tensors["x"].index_select(0, indices), owner, ratio
                )
                result = produce_owner_rows(
                    values,
                    scores,
                    owner,
                    fixture["eps"],
                    tensors["freqs"][boundary],
                    fixture["rope_dim"],
                    ratio,
                    keep_tail=(rank == 0 and ratio == 2),
                )
                self.exact(tensors["main_bytes"][boundary], result.main)
                self.exact(tensors["index_payload"][boundary], result.index_payload)
                self.exact(tensors["index_scales"][boundary], result.index_scales)
                if rank == 0 and ratio == 2:
                    self.exact(tensors["state_tail"], result.state_tail)
                else:
                    self.assertIsNone(result.state_tail)

    def test_stream_and_graph_lifetime(self):
        x = torch.randn(4096, 5120, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(512, 5120, device="cuda", dtype=torch.bfloat16)
        expected = _linear_bf16_bf16_fp32(x, weight)[:1024]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs = []
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.extend(self.op.forward(x[:1024], weight) for _ in range(32))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for output in outputs:
            self.exact(expected, output)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.op.forward(x[:1024], weight)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = self.op.forward(x[:1024], weight)
        for _ in range(8):
            x.normal_()
            expected = _linear_bf16_bf16_fp32(x, weight)[:1024]
            graph.replay()
            self.exact(expected, captured)
        pending = self.op.forward(x[:1024], weight)
        del self.op
        self.exact(expected, pending)

    def test_reject_unsupported_inputs(self):
        self.assertFalse(self.kind.supported(-1))
        self.assertFalse(self.kind.supported(torch.cuda.device_count()))
        with self.assertRaises(RuntimeError):
            self.kind(-1)
        x = torch.empty(1024, 5120, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(512, 5120, device="cuda", dtype=torch.bfloat16)
        misaligned = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[
            1:
        ].view_as(x)
        for bad in (x[:512], x.float(), x.cpu(), x.t(), misaligned):
            with self.subTest(shape=bad.shape, dtype=bad.dtype, device=bad.device):
                with self.assertRaises(RuntimeError):
                    self.op.forward(bad, weight)


if __name__ == "__main__":
    unittest.main()
