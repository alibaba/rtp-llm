"""Qualify the scoped delayed-mHC post diagnostic, including writeback/graphs."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import (
    mhc_post_reference,
    reference_mhc_post,
)
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit


def unit(dim):
    return DelayedHCUnit(
        torch.zeros(24, 4 * dim, device="cuda", dtype=torch.float32),
        torch.zeros(24, device="cuda", dtype=torch.float32),
        torch.ones(3, device="cuda", dtype=torch.float32),
        dim=dim,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        norm_eps=1e-6,
    )


class MHCPostPrecisionTest(unittest.TestCase):
    @torch.inference_mode()
    def test_scalar_oracle_orientation_writeback_and_source_immutability(self):
        dim, owner = 32, unit(32)
        # Powers of two make every intermediate exact; unequal coefficients
        # distinguish source and destination stream axes, including cancellation.
        residual_values = [-8.0, 4.0, 2.0, -1.0]
        post_values = [0.5, 0.25, 1.0, 2.0]
        comb_values = [
            [1.0, 0.25, 0.5, 2.0],
            [0.5, 1.0, 2.0, 0.25],
            [2.0, 0.5, 0.25, 1.0],
            [0.25, 2.0, 1.0, 0.5],
        ]
        expected = torch.tensor(
            [
                4.0 * post_values[destination]
                + sum(
                    comb_values[source][destination] * residual_values[source]
                    for source in range(4)
                )
                for destination in range(4)
            ],
            dtype=torch.bfloat16,
        )
        for leading in ((1,), (5,), (1, 1), (2, 3)):
            with self.subTest(leading=leading):
                x = torch.full(
                    (*leading, dim), 4.0, device="cuda", dtype=torch.bfloat16
                )
                residual = (
                    torch.tensor(residual_values, device="cuda", dtype=torch.bfloat16)
                    .reshape(4, 1)
                    .expand(*leading, 4, dim)
                    .contiguous()
                )
                post = torch.tensor(post_values, device="cuda").reshape(4, 1)
                post = post.expand(*leading, 4, 1).contiguous()
                comb = torch.tensor(comb_values, device="cuda")
                comb = comb.expand(*leading, 4, 4).contiguous()
                sources = [value.clone() for value in (x, post, comb)]
                with mhc_post_reference(True):
                    output = owner.post(x, residual, post, comb)
                self.assertEqual(output.data_ptr(), residual.data_ptr())
                target = expected.reshape(4, 1).expand(*leading, 4, dim)
                self.assertTrue(torch.equal(output.cpu(), target))
                for before, after in zip(sources, (x, post, comb)):
                    self.assertTrue(torch.equal(before, after))

    def test_scoped_disabled_nested_and_exception_restoration(self):
        original = DelayedHCUnit._post_impl
        with mhc_post_reference(False):
            self.assertIs(DelayedHCUnit._post_impl, original)
        with self.assertRaisesRegex(RuntimeError, "scope failure"):
            with mhc_post_reference(True):
                self.assertIs(DelayedHCUnit._post_impl, reference_mhc_post)
                with mhc_post_reference(True), mhc_post_reference(False):
                    self.assertIs(DelayedHCUnit._post_impl, reference_mhc_post)
                self.assertIs(DelayedHCUnit._post_impl, reference_mhc_post)
                raise RuntimeError("scope failure")
        self.assertIs(DelayedHCUnit._post_impl, original)

    @torch.inference_mode()
    def test_changed_input_graph_replay_preserves_writeback(self):
        torch.manual_seed(419)
        dim, owner = 5120, unit(5120)
        for leading in ((1,), (5,), (2, 3)):
            with self.subTest(leading=leading), mhc_post_reference(True):

                def values():
                    return (
                        torch.randn(*leading, dim, device="cuda", dtype=torch.bfloat16),
                        torch.randn(
                            *leading, 4, dim, device="cuda", dtype=torch.bfloat16
                        ),
                        torch.randn(*leading, 4, 1, device="cuda", dtype=torch.float32),
                        torch.randn(*leading, 4, 4, device="cuda", dtype=torch.float32),
                    )

                static = values()
                owner.post(*static)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = owner.post(*static)
                self.assertEqual(captured.data_ptr(), static[1].data_ptr())
                for _ in range(3):
                    changed = values()
                    eager = owner.post(
                        changed[0], changed[1].clone(), changed[2], changed[3]
                    ).clone()
                    for destination, source in zip(static, changed):
                        destination.copy_(source)
                    graph.replay()
                    self.assertTrue(torch.equal(captured, eager))


if __name__ == "__main__":
    unittest.main()
