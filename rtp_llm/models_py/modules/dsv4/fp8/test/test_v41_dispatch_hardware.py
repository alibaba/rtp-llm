"""Exact SM120 EP payload bytes and graph updates; no distributed admission claim."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch


class V41DispatchHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            torch.cuda.is_available(), "Dispatch hardware gate requires CUDA"
        )
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        self.layout = V41DispatchLayout(5120, 6, 384)

    def inputs(self, rows):
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(rows, 6, device="cuda", dtype=torch.float32)
        ids = (torch.arange(rows * 6, device="cuda", dtype=torch.int64) % 384).reshape(
            rows, 6
        )
        valid = torch.arange(rows, device="cuda") % 3 != 1
        x[~valid], weights[~valid] = float("nan"), float("nan")
        return x, weights, ids, valid

    def check(self, inputs, actual):
        expected = self.layout.allocate(actual.shape[0], device="cpu")
        pack_dispatch(
            *(tensor.cpu() for tensor in inputs),
            self.layout,
            out=expected,
            backend="eager"
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_exact_decode_prefill_and_capacity_padding_bytes(self):
        for rows in (0, 1, 8, 33):
            inputs = self.inputs(rows)
            output = self.layout.allocate(rows + 5, device="cuda")
            self.assertIs(pack_dispatch(*inputs, self.layout, out=output), output)
            self.check(inputs, output)

    def test_graph_replay_changes_values_routes_masks_and_padding(self):
        inputs = self.inputs(17)
        x, weights, ids, valid = inputs
        output = self.layout.allocate(24, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                pack_dispatch(*inputs, self.layout, out=output)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pack_dispatch(*inputs, self.layout, out=output)
        for scale in (0, 0.37, 1.5):
            x.normal_()
            weights.fill_(scale)
            ids.add_(1).remainder_(384)
            valid.logical_not_()
            x[~valid] = float("nan")
            output.fill_(173)
            graph.replay()
            self.check(inputs, output)


if __name__ == "__main__":
    unittest.main()
