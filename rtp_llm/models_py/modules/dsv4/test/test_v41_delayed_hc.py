"""Upstream e4a9162f6f delayed-mHC tests plus local interface regressions."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCHead, DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.hc.delayed_reference import (
    DelayedHCUnit as ReferenceHCUnit,
    identity_pre_mix,
)


class DelayedHCTest(unittest.TestCase):
    @staticmethod
    def unit(pre_bias):
        base = torch.zeros(24)
        base[:4] = torch.tensor(pre_bias)
        return DelayedHCUnit(
            torch.zeros(24, 12),
            base,
            torch.ones(3),
            dim=3,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )

    def test_entry_uses_identity_and_next_sublayer_uses_previous_mix(self):
        first = self.unit([-3.0, -1.0, 1.0, 3.0])
        second = self.unit([9.0, 9.0, 9.0, 9.0])
        second.set_previous(first)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        first_input, post, comb = first.pre(x)
        torch.testing.assert_close(first_input, x[:, 0])
        residual = first.post(torch.ones(2, 3), x, post, comb)
        second_input, _, _ = second.pre(residual)
        expected = (residual * first.pre_mix_out.unsqueeze(-1)).sum(-2)
        torch.testing.assert_close(second_input, expected)
        own_mix_readout = (residual * second.pre_mix_out.unsqueeze(-1)).sum(-2)
        self.assertFalse(torch.allclose(second_input, own_mix_readout))

    def test_head_consumes_final_ffn_mix_for_batched_verify(self):
        final = self.unit([-2.0, 0.0, 2.0, 4.0])
        head = DelayedHCHead(final)
        x = torch.arange(72, dtype=torch.float32).reshape(2, 3, 4, 3)
        _, post, comb = final.pre(x)
        residual = final.post(torch.ones(2, 3, 3), x, post, comb)
        expected = (residual * final.pre_mix_out.unsqueeze(-1)).sum(-2)
        torch.testing.assert_close(head.head(residual), expected)
        self.assertEqual(tuple(head.head(residual).shape), (2, 3, 3))
        self.assertEqual(len(list(head.parameters())), 0)

    def test_new_request_replaces_previous_token_layout(self):
        first, second = self.unit([0.0] * 4), self.unit([2.0] * 4)
        second.set_previous(first)
        for tokens in (7, 1, 3):
            x = torch.ones(tokens, 4, 3)
            first.pre(x)
            y, _, _ = second.pre(x)
            torch.testing.assert_close(y, torch.full((tokens, 3), 2.000004))

    def test_loader_scale_column_is_normalized_before_execution(self):
        expected = self.unit([-3.0, -1.0, 1.0, 3.0])
        loaded = DelayedHCUnit(
            expected.fn,
            expected.base,
            expected.scale.unsqueeze(-1),
            dim=3,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        self.assertEqual(tuple(loaded.scale.shape), (3,))
        self.assertEqual(loaded.scale.data_ptr(), expected.scale.data_ptr())
        x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        for actual, reference in zip(loaded.pre(x), expected.pre(x)):
            torch.testing.assert_close(actual, reference)

    def test_predecessor_must_run_and_remain_alive(self):
        first, second = self.unit([0.0] * 4), self.unit([1.0] * 4)
        second.set_previous(first)
        x = torch.ones(2, 4, 3)
        with self.assertRaisesRegex(RuntimeError, "predecessor has not run"):
            second.pre(x)
        first.pre(x)
        second.pre(x)
        del first
        with self.assertRaisesRegex(RuntimeError, "predecessor has not run"):
            second.pre(x)

    def test_upstream_chain_matches_explicit_reference(self):
        torch.manual_seed(41)
        for shape in ((5, 4, 3), (2, 3, 4, 3), (0, 4, 3)):
            with self.subTest(shape=shape):
                actual = torch.randn(shape)
                expected = actual.clone()
                previous = None
                previous_mix = identity_pre_mix(expected)
                units = []
                for index in range(4):
                    unit = self.unit([index - 2.0, -1.0, 1.0, index + 2.0])
                    unit.fn.copy_(torch.randn_like(unit.fn) * 0.01)
                    reference = ReferenceHCUnit(
                        unit.fn,
                        unit.base,
                        unit.scale,
                        dim=3,
                        hc_mult=4,
                        hc_sinkhorn_iters=20,
                        norm_eps=1e-6,
                        hc_eps=1e-6,
                    )
                    unit.set_previous(previous)
                    y, post, comb = unit.pre(actual)
                    ref_y, mixes = reference.pre(expected, previous_mix)
                    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-6)
                    actual = unit.post(y * 0.25, actual, post, comb)
                    expected = reference.post(ref_y * 0.25, expected, mixes)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    previous, previous_mix = unit, mixes.next_pre_mix
                    units.append(unit)
                reduced = (expected * previous_mix.unsqueeze(-1)).sum(-2)
                torch.testing.assert_close(
                    DelayedHCHead(units[-1]).head(actual),
                    reduced,
                    rtol=1e-5,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
