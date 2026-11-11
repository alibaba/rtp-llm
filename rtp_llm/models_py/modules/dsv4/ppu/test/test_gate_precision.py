"""Stable biased routing must preserve small softplus scores on PPU."""

import unittest
import torch
from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.kernels.ppu.moe.ppu_gate_padding import (
    gate_with_padding,
    stable_sqrtsoftplus_gate,
)


@unittest.skipUnless(is_ppu(), "requires a PPU")
class GatePrecisionTest(unittest.TestCase):
    @torch.inference_mode()
    def test_six_routes_match_native_division(self):
        torch.manual_seed(1587)
        bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.1
        for rows in (1, 5, 6, 16, 36):
            scores = torch.randn((rows, 256), device="cuda", dtype=torch.float32)
            for scale in (1.5, 2.5):
                weights, indices = stable_sqrtsoftplus_gate(scores, bias, 6, scale)
                active = torch.nn.functional.softplus(scores).sqrt()
                expected_indices = (active + bias).topk(6, dim=-1).indices
                selected = active.gather(1, expected_indices)
                expected = selected / (selected.sum(-1, keepdim=True) + 1e-12) * scale
                self.assertTrue(torch.equal(indices, expected_indices))
                # A loose tolerance hides changes after weighted SwiGLU quantization.
                self.assertTrue(torch.equal(weights, expected), (rows, scale))

    @torch.inference_mode()
    def test_small_scores_and_graph_replay(self):
        torch.manual_seed(1587)
        for rows in (1, 5, 6, 36):
            scores = torch.linspace(-100, 20, 256, device="cuda").repeat(rows, 1)
            # Counterbalance the exact activation so small contributions decide top-k.
            bias = -torch.nn.functional.softplus(scores[0]).sqrt()
            bias += torch.linspace(0, 0.001, 256, device="cuda").flip(0)
            mask = torch.ones(rows, device="cuda", dtype=torch.bool)
            for topk in (1, 6, 16, 32):
                gate_with_padding(scores, bias, topk)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    weights, indices = gate_with_padding(
                        scores, bias, topk, 1.5, active_token_mask=mask
                    )
                for delta in (0.0, -0.25, 0.5):
                    scores.add_(delta)
                    mask[:] = True
                    if rows > 1:
                        mask[-1] = False
                    weights.fill_(float("nan"))
                    indices.fill_(-99)
                    graph.replay()
                    active = torch.nn.functional.softplus(scores).sqrt()
                    expected_indices = (active + bias).topk(topk, dim=-1).indices
                    expected_weights = active.gather(1, expected_indices)
                    expected_weights = (
                        expected_weights
                        / (expected_weights.sum(-1, keepdim=True) + 1e-12)
                        * 1.5
                    )
                    self.assertTrue(
                        torch.equal(
                            indices[mask, :topk].sort(-1).values,
                            expected_indices[mask].sort(-1).values,
                        )
                    )
                    ew = torch.zeros_like(scores).scatter_(
                        1, expected_indices, expected_weights
                    )
                    aw = torch.zeros_like(scores[mask]).scatter_(
                        1, indices[mask, :topk], weights[mask, :topk]
                    )
                    torch.testing.assert_close(aw, ew[mask], atol=2e-7, rtol=2e-6)
                    self.assertTrue((indices[~mask] == -1).all())
                    self.assertTrue((weights[~mask] == 0).all())
                    self.assertTrue((indices[mask, topk:] == -1).all())
                    self.assertTrue((weights[mask, topk:] == 0).all())
                graph.reset()
                w, i = stable_sqrtsoftplus_gate(scores, bias, topk, 1.5)
                self.assertEqual(w.shape, (rows, topk))
                self.assertTrue(w.is_contiguous() and i.is_contiguous())

    @torch.inference_mode()
    def test_nonfinite_and_empty(self):
        bias = torch.zeros(256, device="cuda")
        for value in (float("nan"), float("inf"), -float("inf")):
            scores = torch.zeros((2, 256), device="cuda")
            scores[0, 3] = value
            w, i = stable_sqrtsoftplus_gate(scores, bias, 6, 1.5)
            torch.testing.assert_close(
                w[0], torch.full((6,), 0.25, device="cuda"), rtol=0, atol=0
            )
            self.assertTrue(torch.equal(i[0], torch.arange(6, device="cuda")))
        w, i = stable_sqrtsoftplus_gate(torch.empty((0, 256), device="cuda"), bias, 6)
        self.assertEqual(w.shape, (0, 6))
        self.assertEqual(i.shape, (0, 6))


if __name__ == "__main__":
    unittest.main()
