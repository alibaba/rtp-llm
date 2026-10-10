"""Bitwise state/output parity and owned output storage for cached prefills."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_cublas,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.reuse_state_graph import (
    _GRAPHS,
    chunk_gated_delta_rule_fwd_h_reuse_graph,
)


class ReuseStateGraphTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.manual_seed(530710)

    def inputs(self, length, heads=16, sequences=1):
        shape = (1, length, heads, 128)
        values = dict(
            k=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            w=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            u=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            gk=-torch.rand(shape, device="cuda", dtype=torch.float32) * 4,
            initial_state=torch.randn(sequences, heads, 128, 128, device="cuda") * 0.1,
            output_final_state=True,
            intermediate_state_dtype=torch.float32,
            chunk_size=64,
            use_exp2=True,
        )
        return values

    def compare(self, args):
        expected = chunk_gated_delta_rule_fwd_h_cublas(**args)
        actual = chunk_gated_delta_rule_fwd_h_reuse_graph(**args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        return actual

    def test_partial_chunks_nonzero_states_and_replays(self):
        for heads in (1, 16):
            for length in (1, 31, 63, 64, 65, 75, 127, 128, 129, 257, 512):
                with self.subTest(heads=heads, length=length):
                    args = self.inputs(length, heads)
                    args["cu_seqlens"] = torch.tensor([0, length], device="cuda", dtype=torch.int32)
                    old = self.compare(args)
                    stream = torch.cuda.current_stream(args["k"].device)
                    key = (args["k"].device.index, stream.cuda_stream,
                           args["k"].shape, args["gk"].dtype)
                    self.assertIn(key, _GRAPHS, "supported inputs must replay a graph")
                    frozen = tuple(t.clone() for t in old)
                    for name in ("k", "w", "u", "gk", "initial_state"):
                        args[name].mul_(0.75)
                    self.compare(args)
                    for a, b in zip(old, frozen):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertLessEqual(len(_GRAPHS), 8)

    def test_ragged_batch_uses_reference_math(self):
        args = self.inputs(149, sequences=3)
        args["cu_seqlens"] = torch.tensor([0, 1, 76, 149], device="cuda", dtype=torch.int32)
        self.compare(args)

    def test_stream_workspace_isolation(self):
        args = self.inputs(75)
        first = self.compare(args)
        snapshots = tuple(t.clone() for t in first)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compare(self.inputs(75))
        torch.cuda.current_stream().wait_stream(stream)
        for a, b in zip(first, snapshots):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
