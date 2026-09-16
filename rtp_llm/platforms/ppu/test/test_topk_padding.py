"""Inactive DeepEP routes must retain order and weight bits through replay."""

import unittest

import torch

from rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency import pad_topk


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class TopkPaddingTest(unittest.TestCase):
    def assert_routes(self, indices, weights, output):
        out_indices, out_weights = output
        width = indices.shape[1]
        self.assertTrue(out_indices.is_contiguous())
        self.assertTrue(out_weights.is_contiguous())
        self.assertTrue(torch.equal(out_indices[:, :width], indices))
        self.assertTrue(
            torch.equal(
                out_weights[:, :width].view(torch.int32), weights.view(torch.int32)
            )
        )
        self.assertTrue(bool((out_indices[:, width:] == -1).all()))
        self.assertTrue(bool((out_weights[:, width:].view(torch.int32) == 0).all()))

    def test_widths_and_strides_preserve_values(self):
        for rows in (0, 1, 3, 8, 128):
            for width in range(1, 17):
                for strided in (False, True):
                    with self.subTest(rows=rows, width=width, strided=strided):
                        indices = torch.randint(
                            -1, 256, (rows, width * 2), device="cuda"
                        )
                        weights = torch.randn((rows, width * 2), device="cuda")
                        indices, weights = indices[:, ::2], weights[:, ::2]
                        if not strided:
                            indices, weights = (
                                indices.contiguous(),
                                weights.contiguous(),
                            )
                        output = pad_topk(indices, weights)
                        if width in (2, 4, 8, 16):
                            self.assertIs(output[0], indices)
                            self.assertIs(output[1], weights)
                        else:
                            self.assert_routes(indices, weights, output)

    def test_changing_graph_inputs_overwrite_every_output(self):
        for rows in (1, 3, 8, 32, 128):
            with self.subTest(rows=rows):
                indices = torch.zeros((rows, 6), dtype=torch.int64, device="cuda")
                weights = torch.zeros((rows, 6), dtype=torch.float32, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        pad_topk(indices, weights)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = pad_topk(indices, weights)
                torch.cuda.current_stream().wait_stream(stream)
                for iteration in range(5):
                    indices.copy_(torch.randint(-1, 256, indices.shape, device="cuda"))
                    weights.copy_(torch.randn_like(weights))
                    # Include sign-zero and exact subnormal values in the copy contract.
                    weights[:, 0] = -0.0
                    weights[:, 1] = 2**-140
                    if iteration == 2:
                        indices.fill_(-1)
                        weights.zero_()
                    output[0].fill_(999)
                    output[1].fill_(123)
                    graph.replay()
                    self.assert_routes(indices, weights, output)


if __name__ == "__main__":
    unittest.main()
