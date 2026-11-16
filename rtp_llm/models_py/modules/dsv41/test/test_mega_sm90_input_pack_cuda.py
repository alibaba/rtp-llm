# SPDX-License-Identifier: Apache-2.0
"""Hopper correctness and CUDA-graph gates for the fused SM90 input packer."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.moe._mega_sm90_input_pack_triton import (
    fused_pack_mega_moe_sm90_inputs,
)
from rtp_llm.models_py.modules.dsv41.moe.mega_sm90_input_packer import (
    MegaMoeSM90InputPacker,
)


class _Buffer:
    def __init__(self, tokens: int, hidden: int, topk: int, device):
        self.x = torch.empty(
            (tokens, hidden), dtype=torch.float8_e4m3fn, device=device
        )
        self.x_sf = torch.empty((tokens, hidden // 128), device=device)
        self.topk_idx = torch.empty(
            (tokens, topk), dtype=torch.int64, device=device
        )
        self.topk_weights = torch.empty((tokens, topk), device=device)


def _reference(x, weights, indices, buf):
    MegaMoeSM90InputPacker().pack(x, weights, indices, buf, x.shape[0])


def _fused(x, weights, indices, buf):
    fused_pack_mega_moe_sm90_inputs(
        x,
        weights,
        indices,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MegaSm90InputPackCudaTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _case(self, tokens, *, seed=20261028):
        hidden, topk = 5120, 6
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(
            (tokens, hidden),
            dtype=torch.bfloat16,
            device=self.device,
            generator=generator,
        )
        if tokens >= 4:
            x[0, 0] = float("nan")
            x[1, 1] = float("inf")
            x[2, 2] = -float("inf")
            x[3, 128:256] = 0
        weights = torch.randn(
            (tokens, topk), device=self.device, generator=generator
        )
        indices = torch.randint(
            0,
            384,
            (tokens, topk),
            dtype=torch.int64,
            device=self.device,
            generator=generator,
        )
        return x, weights, indices

    def assert_buffers_bitwise_equal(self, expected, actual):
        self.assertTrue(torch.equal(expected.x.view(torch.uint8), actual.x.view(torch.uint8)))
        self.assertTrue(
            torch.equal(expected.x_sf.view(torch.int32), actual.x_sf.view(torch.int32))
        )
        self.assertTrue(torch.equal(expected.topk_idx, actual.topk_idx))
        self.assertTrue(
            torch.equal(
                expected.topk_weights.view(torch.int32),
                actual.topk_weights.view(torch.int32),
            )
        )

    def test_matches_reference_bitwise(self):
        for tokens in (1, 4, 12, 32, 64, 2048):
            with self.subTest(tokens=tokens):
                x, weights, indices = self._case(tokens)
                expected = _Buffer(tokens, x.shape[1], weights.shape[1], self.device)
                actual = _Buffer(tokens, x.shape[1], weights.shape[1], self.device)
                _reference(x, weights, indices, expected)
                _fused(x, weights, indices, actual)
                torch.cuda.synchronize()
                self.assert_buffers_bitwise_equal(expected, actual)

    def test_capture_replay_reads_mutated_inputs(self):
        tokens = 12
        x, weights, indices = self._case(tokens)
        actual = _Buffer(tokens, x.shape[1], weights.shape[1], self.device)
        # Compile before capture; serving graph warmup follows the same contract.
        _fused(x, weights, indices, actual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fused(x, weights, indices, actual)

        next_x, next_weights, next_indices = self._case(tokens, seed=20261029)
        x.copy_(next_x)
        weights.copy_(next_weights)
        indices.copy_(next_indices)
        graph.replay()
        expected = _Buffer(tokens, x.shape[1], weights.shape[1], self.device)
        _reference(x, weights, indices, expected)
        torch.cuda.synchronize()
        self.assert_buffers_bitwise_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
