"""Actual delayed-HC stage-entry readout and per-forward imported state."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.hc.pp_state import imported_delayed_pre_mix


def fixture(*, device="cpu", dim=8, rows=3):
    torch.manual_seed(41021)
    streams = 4
    width = streams * (streams + 2)
    unit = DelayedHCUnit(
        dim=dim,
        hc_mult=streams,
        norm_eps=1e-6,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        fn=torch.randn(width, streams * dim, device=device),
        base=torch.randn(width, device=device),
        scale=torch.tensor([0.2, 0.3, 0.4], device=device),
    )
    residual = torch.randn(rows, streams, dim, device=device).bfloat16()
    pre_mix = torch.rand(rows, streams, device=device)
    return unit, residual, pre_mix


class ImportedDelayedStateTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "SM120 delayed-HC bridge qualification requires its own GPU lease",
    )
    @torch.inference_mode()
    def test_sm120_real_unit_matches_local_link_at_v41_width(self):
        for rows in (1, 5, 257):
            unit, residual, _ = fixture(device="cuda", dim=5120, rows=rows)
            previous, earlier, _ = fixture(device="cuda", dim=5120, rows=rows)
            previous.base[:4].add_(0.5)
            earlier.mul_(0.375).add_(0.125)
            previous.pre(earlier)
            unit.set_previous(previous)
            expected = tuple(value.clone() for value in unit.pre(residual))
            current = unit.pre_mix_out.clone()
            self.assertFalse(torch.equal(previous.pre_mix_out, current))
            unit.set_previous(None)
            with imported_delayed_pre_mix(unit, residual, previous.pre_mix_out):
                previous.pre_mix_out.zero_()
                actual = unit.pre(residual)
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(
                        left.view(torch.uint8), right.view(torch.uint8), rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    current.view(torch.uint8),
                    unit.pre_mix_out.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
            self.assertIsNone(unit._previous_ref)

    def test_real_unit_uses_previous_mix_and_preserves_current_coefficients(self):
        unit, residual, pre_mix = fixture()
        identity, post, comb = unit._pre_impl(residual)
        current = unit.pre_mix_out.clone()
        expected = (residual.float() * pre_mix.unsqueeze(-1)).sum(1).bfloat16()
        with imported_delayed_pre_mix(unit, residual, pre_mix):
            value, imported_post, imported_comb = unit._pre_impl(residual)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            torch.testing.assert_close(imported_post, post, rtol=0, atol=0)
            torch.testing.assert_close(imported_comb, comb, rtol=0, atol=0)
            torch.testing.assert_close(unit.pre_mix_out, current, rtol=0, atol=0)
        self.assertIsNone(unit._previous_ref)
        torch.testing.assert_close(
            unit._pre_impl(residual)[0], identity, rtol=0, atol=0
        )

    def test_receive_buffer_reuse_cannot_change_bound_forward(self):
        unit, residual, pre_mix = fixture()
        expected = (residual.float() * pre_mix.unsqueeze(-1)).sum(1).bfloat16()
        with imported_delayed_pre_mix(unit, residual, pre_mix):
            pre_mix.zero_()
            torch.testing.assert_close(
                unit._pre_impl(residual)[0], expected, rtol=0, atol=0
            )
        with imported_delayed_pre_mix(unit, residual, pre_mix):
            torch.testing.assert_close(
                unit._pre_impl(residual)[0], torch.zeros_like(expected), rtol=0, atol=0
            )
        self.assertIsNone(unit._previous_ref)

    def test_exception_and_nested_binding_do_not_leave_imported_state(self):
        unit, residual, pre_mix = fixture()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with imported_delayed_pre_mix(unit, residual, pre_mix):
                with self.assertRaisesRegex(ValueError, "local predecessor"):
                    with imported_delayed_pre_mix(unit, residual, pre_mix):
                        self.fail("nested binding was admitted")
                unit._pre_impl(residual)
                raise RuntimeError("injected")
        self.assertIsNone(unit._previous_ref)
        torch.testing.assert_close(
            unit._pre_impl(residual)[0], residual[:, 0], rtol=0, atol=0
        )

    def test_existing_local_predecessor_is_preserved(self):
        unit, residual, pre_mix = fixture()
        previous, _, _ = fixture()
        previous._pre_impl(residual)
        unit.set_previous(previous)
        with self.assertRaisesRegex(ValueError, "local predecessor"):
            with imported_delayed_pre_mix(unit, residual, pre_mix):
                self.fail("local predecessor was replaced")
        self.assertIs(unit._previous_ref(), previous)

    def test_bad_payload_is_rejected_before_binding(self):
        unit, residual, pre_mix = fixture()
        invalid_mix = [
            pre_mix.bfloat16(),
            pre_mix[:, :3],
            pre_mix.t(),
            torch.full_like(pre_mix, float("nan")),
            torch.full_like(pre_mix, float("inf")),
        ]
        for candidate in invalid_mix:
            with self.assertRaises(ValueError):
                with imported_delayed_pre_mix(unit, residual, candidate):
                    self.fail("invalid pre-mix was admitted")
            self.assertIsNone(unit._previous_ref)
        for candidate in (
            residual.float(),
            residual.transpose(0, 1),
            residual[:, :, ::2],
        ):
            with self.assertRaises(ValueError):
                with imported_delayed_pre_mix(unit, candidate, pre_mix):
                    self.fail("invalid residual was admitted")
            self.assertIsNone(unit._previous_ref)


if __name__ == "__main__":
    unittest.main()
