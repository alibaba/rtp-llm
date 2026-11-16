"""Inactive DeepEP routes must retain order and weight bits through replay."""

import unittest

import torch
from rtp_llm.device.device_type import is_ppu

from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_deepep_fp4 import pad_topk


@unittest.skipUnless(
    is_ppu(),
    "requires a PPU",
)
class TopkPaddingTest(unittest.TestCase):
    def test_hash_nonfinite_rows_fall_back_to_slot_routes(self):
        from rtp_llm.models_py.kernels.ppu.moe.ppu_hash_gate import hash_gate_with_padding

        scores = torch.tensor(
            [[-float("inf"), 1, 2, 3], [float("inf"), 1, 2, 3],
             [float("nan"), 1, 2, 3], [0, 1, 2, 3]], device="cuda"
        )
        table = torch.tensor([[3, 0]], device="cuda")
        ids = torch.zeros(4, dtype=torch.int64, device="cuda")
        scale, topk = 2.5, 2
        active = torch.nn.functional.softplus(scores).sqrt()
        # PR1434 semantics: a row with any non-finite router logit or activated
        # score is replaced wholesale by slot experts 0..topk-1 with uniform
        # weights, so a broken row never trusts the hash table.
        finite = torch.isfinite(scores).all(dim=-1) & torch.isfinite(active).all(dim=-1)
        slots = torch.arange(topk, device="cuda").expand(4, topk)
        routed = torch.where(finite.unsqueeze(-1), table[ids], slots)
        selected = active.gather(1, routed)
        expected_w = selected / (selected.sum(dim=-1, keepdim=True) + 1e-12) * scale
        expected_w = torch.where(
            finite.unsqueeze(-1),
            expected_w,
            torch.full_like(expected_w, scale / topk),
        )
        for mask in (torch.ones(4, dtype=torch.bool, device="cuda"),
                     torch.tensor([True, False, True, False], device="cuda")):
            keep = mask.unsqueeze(-1)
            indices = torch.where(keep, routed, torch.full_like(routed, -1))
            weights = torch.where(keep, expected_w, torch.zeros_like(expected_w))
            actual_weights, actual_indices = hash_gate_with_padding(
                scores, ids, table, scale, active_token_mask=mask
            )
            self.assertTrue(torch.equal(actual_indices, indices))
            torch.testing.assert_close(actual_weights, weights, rtol=0, atol=1e-6)

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

    def test_mask_changes_in_graph_preserve_active_route_bits(self):
        for width in (1, 2, 4, 6, 8, 16):
            rows = 17
            indices = torch.randint(-1, 256, (rows, width), device="cuda")
            weights = torch.randn((rows, width), device="cuda")
            weights[:, 0] = -0.0
            if width > 1:
                weights[:, 1] = 2**-140
            mask = torch.ones(rows, dtype=torch.bool, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    pad_topk(indices, weights, mask)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = pad_topk(indices, weights, mask)
            torch.cuda.current_stream().wait_stream(stream)
            for active in (17, 3, 0, 9, 17):
                mask.copy_(torch.arange(rows, device="cuda") < active)
                output[0].fill_(999)
                output[1].fill_(123)
                graph.replay()
                expected_indices = indices.clone()
                expected_weights = weights.clone()
                expected_indices[active:] = -1
                expected_weights[active:] = 0
                self.assert_routes(expected_indices, expected_weights, output)
            with self.assertRaisesRegex(ValueError, "device bool"):
                pad_topk(indices, weights, mask.to(torch.int32))
            with self.assertRaisesRegex(ValueError, "device bool"):
                pad_topk(indices, weights, mask[:-1])

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
