import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import (
    Qwen4ExpGatedResidual,
    inject_into_residual,
)

_HC = 4
_HIDDEN = 16
_LOWRANK = 6
_EPS = 1e-6


def _make_unit(seed, with_inject=True):
    torch.manual_seed(seed)
    hc_hidden = _HC * _HIDDEN
    inject = torch.randn(_HC, hc_hidden) * 0.05 if with_inject else None
    return Qwen4ExpGatedResidual(
        torch.randn(hc_hidden) * 0.1,
        torch.randn(_LOWRANK, hc_hidden) * 0.05,
        torch.randn(hc_hidden, _LOWRANK) * 0.05,
        inject,
        hc_mult=_HC,
        norm_eps=_EPS,
    )


def _fake_attn(x):
    """Stand-in sublayer: any deterministic [.., hidden] -> [.., hidden]."""
    return torch.tanh(x) * 2.0


def _fake_mlp(x):
    return torch.sigmoid(x) - 0.5


class ResidualStreamFlowTest(unittest.TestCase):
    """Pins the layer/model residual algebra transcribed from upstream.

    Upstream Qwen4ExpTextDecoderLayer.forward is:
        h, hyper, w = attn_hyper_connection(hyper)
        h = attn(h)
        hyper = hyper + (h.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
        ... same for mlp ...
    and Qwen4ExpTextModel.forward brackets the loop with
        hidden = hidden.repeat(1, 1, hc_count)   ...   hyper_connection_mixer(hidden)
    """

    def test_embedding_tiling_puts_a_copy_in_every_branch(self):
        embed = torch.randn(5, _HIDDEN)

        hyper = embed.repeat(*(1,) * (embed.dim() - 1), _HC)

        self.assertEqual(hyper.shape, (5, _HC * _HIDDEN))
        for branch in range(_HC):
            torch.testing.assert_close(
                hyper[:, branch * _HIDDEN : (branch + 1) * _HIDDEN], embed
            )

    def test_one_sublayer_round_trip_matches_reference(self):
        unit = _make_unit(0)
        embed = torch.randn(3, _HIDDEN)
        hyper = embed.repeat(1, _HC)
        x, hyper_in, w = unit(hyper)
        out = inject_into_residual(hyper_in, _fake_attn(x), w)

        ref = hyper + (_fake_attn(x).unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)
        self.assertEqual(out.shape, hyper.shape)
        torch.testing.assert_close(out, ref)

    def test_two_units_and_mixer_compose_into_a_layer(self):
        attn_hc = _make_unit(0)
        mlp_hc = _make_unit(1)
        mixer = _make_unit(2, with_inject=False)
        embed = torch.randn(3, _HIDDEN)
        hyper = embed.repeat(1, _HC)
        x, hyper, w = attn_hc(hyper)
        hyper = inject_into_residual(hyper, _fake_attn(x), w)
        x, hyper, w = mlp_hc(hyper)
        hyper = inject_into_residual(hyper, _fake_mlp(x), w)
        out, _, none = mixer(hyper)

        self.assertIsNone(none)
        self.assertEqual(out.shape, (3, _HIDDEN))
        self.assertTrue(torch.isfinite(out).all())

    def test_residual_stream_is_not_left_normalized(self):
        # `hyper_input` returned by the unit must be the raw stream, not the
        # normalized one -- writing into a normalized stream would drop the
        # accumulated residual.
        unit = _make_unit(0)
        hyper = torch.randn(3, _HC * _HIDDEN) * 5.0

        _, hyper_out, _ = unit(hyper)

        torch.testing.assert_close(hyper_out, hyper)

    def test_stream_width_survives_many_layers(self):
        units = [_make_unit(i) for i in range(6)]
        hyper = torch.randn(2, _HIDDEN).repeat(1, _HC)

        for unit in units:
            x, hyper, w = unit(hyper)
            hyper = inject_into_residual(hyper, x * 0.5, w)

        self.assertEqual(hyper.shape, (2, _HC * _HIDDEN))
        self.assertTrue(torch.isfinite(hyper).all())


if __name__ == "__main__":
    unittest.main()
