"""Per-forward rotary metadata sharing, including captured producer replay."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.rope_metadata import decode_rope_metadata


class DecodeRopeMetadataTest(unittest.TestCase):
    def check_device(self, device):
        table = torch.randn(128, 32, dtype=torch.complex64, device=device)
        positions = torch.tensor([0, 2, 7, 127], dtype=torch.int32, device=device)
        shared = {}
        converted, first = decode_rope_metadata(shared, table, positions)
        second_positions, second = decode_rope_metadata(shared, table, positions[:])
        self.assertEqual(converted.data_ptr(), second_positions.data_ptr())
        self.assertEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, table[positions.long()], rtol=0, atol=0)
        other_table = table * 2
        _, other = decode_rope_metadata(shared, other_table, positions)
        torch.testing.assert_close(other, other_table[positions.long()], rtol=0, atol=0)
        self.assertNotEqual(first.data_ptr(), other.data_ptr())
        positions.add_(-1).clamp_min_(0)
        shared.pop("decode_rope_metadata")
        _, next_forward = decode_rope_metadata(shared, table, positions)
        torch.testing.assert_close(
            next_forward, table[positions.long()], rtol=0, atol=0
        )
        with patch.dict(os.environ, {"DSV41_REUSE_DECODE_ROPE": "0"}):
            _, fallback = decode_rope_metadata(shared, table, positions)
        torch.testing.assert_close(next_forward, fallback, rtol=0, atol=0)

    def test_sharing_fallback_and_invalidation(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_sharing(self):
        self.check_device("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_graph_producer_reads_new_positions(self):
        table = torch.randn(128, 32, dtype=torch.complex64, device="cuda")
        positions = torch.zeros(24, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                decode_rope_metadata({}, table, positions)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        shared = {}
        with torch.cuda.graph(graph, stream=stream):
            converted, first = decode_rope_metadata(shared, table, positions)
            _, second = decode_rope_metadata(shared, table, positions)
            output = first + second
        self.assertEqual(first.data_ptr(), second.data_ptr())
        for offset in (1, 17, 103):
            positions.fill_(offset)
            graph.replay()
            torch.testing.assert_close(converted, positions.long(), rtol=0, atol=0)
            torch.testing.assert_close(
                output, table[positions.long()] * 2, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
