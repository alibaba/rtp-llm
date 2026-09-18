import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import (
    Qwen4ExpGatedResidual,
    inject_into_residual,
)

_HC = 4
_HIDDEN = 16
_LOWRANK = 6
_EPS = 1e-6


def _reference(hyper_input, gamma_raw, down, up, inject, *, hc, eps):
    """Independent transcription of transformers Qwen4ExpTextGatedResidual.

    Mirrors modular_qwen4_exp.py: Qwen4ExpTextRMSNorm(group_size=hidden) does
    ``_norm(x.float()) * (1.0 + weight.float())``, then the low-rank gate.
    """
    hidden = hyper_input.shape[-1] // hc
    x = hyper_input.float().reshape(*hyper_input.shape[:-1], hc, hidden)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = (x.flatten(-2) * (1.0 + gamma_raw.float())).type_as(hyper_input)

    w = F.silu(F.linear(normed, down) / hc)
    w = torch.sigmoid(F.linear(w, up))
    w = w.unflatten(-1, (hc, hidden))
    mixed = (w * normed.unflatten(-1, (hc, hidden))).mean(dim=-2)
    if inject is None:
        return mixed, None
    return mixed, 2 * torch.sigmoid(F.linear(normed, inject) / hc)


class Qwen4ExpGatedResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hc_hidden = _HC * _HIDDEN
        # Checkpoint gamma is zero-centred; the module adds the +1 in fp32.
        self.gamma_raw = torch.randn(self.hc_hidden) * 0.1
        self.down = torch.randn(_LOWRANK, self.hc_hidden) * 0.05
        self.up = torch.randn(self.hc_hidden, _LOWRANK) * 0.05
        self.inject = torch.randn(_HC, self.hc_hidden) * 0.05

    def _build(self, inject):
        return Qwen4ExpGatedResidual(
            self.gamma_raw,
            self.down,
            self.up,
            inject,
            hc_mult=_HC,
            norm_eps=_EPS,
        )

    def test_matches_upstream_reference_flat(self):
        x = torch.randn(7, self.hc_hidden)

        mixed, hyper_input, inject_weights = self._build(self.inject)(x)
        ref_mixed, ref_inject = _reference(
            x, self.gamma_raw, self.down, self.up, self.inject, hc=_HC, eps=_EPS
        )

        self.assertEqual(mixed.shape, (7, _HIDDEN))
        self.assertEqual(inject_weights.shape, (7, _HC))
        self.assertIs(hyper_input, x)
        torch.testing.assert_close(mixed, ref_mixed)
        torch.testing.assert_close(inject_weights, ref_inject)

    def test_matches_upstream_reference_batched(self):
        x = torch.randn(2, 5, self.hc_hidden)

        mixed, _, inject_weights = self._build(self.inject)(x)
        ref_mixed, ref_inject = _reference(
            x, self.gamma_raw, self.down, self.up, self.inject, hc=_HC, eps=_EPS
        )

        self.assertEqual(mixed.shape, (2, 5, _HIDDEN))
        torch.testing.assert_close(mixed, ref_mixed)
        torch.testing.assert_close(inject_weights, ref_inject)

    def test_global_mixer_has_no_write_gate(self):
        x = torch.randn(3, self.hc_hidden)

        mixed, _, inject_weights = self._build(None)(x)
        ref_mixed, ref_inject = _reference(
            x, self.gamma_raw, self.down, self.up, None, hc=_HC, eps=_EPS
        )

        self.assertIsNone(inject_weights)
        self.assertIsNone(ref_inject)
        torch.testing.assert_close(mixed, ref_mixed)

    def test_norm_is_per_branch_not_whole_stream(self):
        # Scaling one branch must not change the other branches' normed values,
        # which is what group_size=hidden buys us over a plain 10240-wide RMS.
        unit = self._build(self.inject)
        x = torch.randn(1, self.hc_hidden)
        scaled = x.clone().unflatten(-1, (_HC, _HIDDEN))
        scaled[..., 0, :] *= 8.0
        scaled = scaled.flatten(-2)

        base = unit._norm(x).unflatten(-1, (_HC, _HIDDEN))
        after = unit._norm(scaled).unflatten(-1, (_HC, _HIDDEN))

        torch.testing.assert_close(base[..., 1:, :], after[..., 1:, :])
        torch.testing.assert_close(base[..., 0, :], after[..., 0, :])

    def test_inject_writes_gated_output_into_every_branch(self):
        hyper_input = torch.randn(4, self.hc_hidden)
        sublayer_out = torch.randn(4, _HIDDEN)
        inject_weights = torch.rand(4, _HC)

        out = inject_into_residual(hyper_input, sublayer_out, inject_weights)

        self.assertEqual(out.shape, hyper_input.shape)
        delta = (out - hyper_input).unflatten(-1, (_HC, _HIDDEN))
        for branch in range(_HC):
            torch.testing.assert_close(
                delta[:, branch, :],
                sublayer_out * inject_weights[:, branch : branch + 1],
            )

    def test_rejects_wrong_stream_width(self):
        unit = self._build(self.inject)

        with self.assertRaisesRegex(ValueError, "expected 64 gated residual"):
            unit(torch.randn(2, self.hc_hidden + 1))

    def test_rejects_mismatched_lowrank(self):
        with self.assertRaisesRegex(ValueError, "mix_down rank .* != mix_up rank"):
            Qwen4ExpGatedResidual(
                self.gamma_raw,
                self.down,
                torch.randn(self.hc_hidden, _LOWRANK + 1),
                self.inject,
                hc_mult=_HC,
                norm_eps=_EPS,
            )

    def test_rejects_mismatched_inject_shape(self):
        with self.assertRaisesRegex(ValueError, "inject weight"):
            Qwen4ExpGatedResidual(
                self.gamma_raw,
                self.down,
                self.up,
                torch.randn(_HC + 1, self.hc_hidden),
                hc_mult=_HC,
                norm_eps=_EPS,
            )


if __name__ == "__main__":
    unittest.main()
