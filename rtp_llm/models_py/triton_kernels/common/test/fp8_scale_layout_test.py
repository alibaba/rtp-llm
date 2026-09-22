"""GPU regression checks for the FP8 all-gather scale layout."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.fp8_scale_layout import (
    repack_ag_scale_wire,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Fp8ScaleLayoutTest(unittest.TestCase):
    def test_scale_repack_graph_replay_preserves_all_bits_and_zeroes_global_tail(self):
        for ranks, rows in ((8, 1), (8, 2), (8, 4), (8, 8), (8, 16), (8, 32), (3, 3)):
            groups, local_pad = 14, (rows + 3) // 4 * 4
            wire = torch.zeros(
                ranks * groups, local_pad, device="cuda", dtype=torch.int32
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                repack_ag_scale_wire(wire, rows, ranks)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = repack_ag_scale_wire(wire, rows, ranks)
            for step in range(3):
                wire.random_(-(2**31), 2**31)
                actual.fill_(-123)  # Global padding must be rewritten on replay.
                graph.replay()
                expected = torch.zeros_like(actual)
                for rank in range(ranks):
                    expected[:, rank * rows : (rank + 1) * rows] = wire[
                        rank * groups : (rank + 1) * groups, :rows
                    ]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
