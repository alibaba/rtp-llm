"""Rotary table initialization and per-forward metadata sharing/replay."""

import math
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8.decode.rope_metadata import (
    decode_rope_metadata,
)
from rtp_llm.models_py.modules.dsv41.rope import precompute_freqs_cis


def reference_frequency_rows(positions, base, *, original_seq_len=0):
    """FP32 device-local reference used by DeepInfer's rotary layers.

    Match the reference inverse-frequency/outer/polar boundaries rather than
    transferring a CPU-generated table. Factor 16 is the checkpoint's YaRN
    scale; it preserves the existing interpolation expression's rounding.
    """
    dim = 64
    channels = torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device)
    pos_freqs = base ** (channels / dim)
    inv_freq = 1.0 / pos_freqs
    if original_seq_len:
        low = math.floor(
            dim * math.log(original_seq_len / (32 * 2 * math.pi)) / (2 * math.log(base))
        )
        high = math.ceil(
            dim * math.log(original_seq_len / (1 * 2 * math.pi)) / (2 * math.log(base))
        )
        low, high = max(low, 0), min(high, dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=positions.device) - low)
            / (high - low)
        ).clamp(0, 1)
        mask = 1 - ramp
        inv_freq = 1.0 / (16 * pos_freqs) * (1 - mask) + inv_freq * mask
    angles = torch.outer(positions.float(), inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


class RopeTableInitializationTest(unittest.TestCase):
    def test_cpu_reference_and_table_reuse(self):
        positions = torch.tensor([0, 1, 127, 239], dtype=torch.int64)
        for base, original in ((10000.0, 0), (160000.0, 65536)):
            args = (64, 240, original, base, 16.0, 32, 1)
            table = precompute_freqs_cis(*args)
            self.assertIs(table, precompute_freqs_cis(*args))
            self.assertEqual(table.device.type, "cpu")
            self.assertEqual(table.dtype, torch.complex64)
            torch.testing.assert_close(
                table[positions],
                reference_frequency_rows(positions, base, original_seq_len=original),
                rtol=0,
                atol=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_long_positions_match_device_local_reference(self):
        # These include the online first-draft position and the two committed
        # keys whose decoded FP8 values differed with CPU inverse frequencies.
        positions = torch.tensor(
            [0, 1, 100807, 100869, 100881, 100935, 100939],
            dtype=torch.int64,
            device="cuda",
        )
        for base, original in ((10000.0, 0), (160000.0, 65536)):
            table = precompute_freqs_cis(
                64, 100940, original, base, 16.0, 32, 1, device=positions.device
            )
            self.assertEqual(table.device, positions.device)
            self.assertEqual(table.dtype, torch.complex64)
            self.assertTrue(table.is_contiguous())
            torch.testing.assert_close(
                table.index_select(0, positions),
                reference_frequency_rows(positions, base, original_seq_len=original),
                rtol=0,
                atol=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_current_device_and_explicit_index_share_table(self):
        device = torch.device("cuda", torch.cuda.current_device())
        args = (64, 241, 0, 10000.0, 16.0, 32, 1)
        first = precompute_freqs_cis(*args, device="cuda")
        explicit = precompute_freqs_cis(*args, device=device)
        repeated = precompute_freqs_cis(*args, device=str(device))
        self.assertIs(first, explicit)
        self.assertIs(first, repeated)
        self.assertEqual(first.device, device)
        cpu = precompute_freqs_cis(*args, device="cpu")
        self.assertEqual(cpu.device.type, "cpu")
        self.assertIsNot(first, cpu)


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
