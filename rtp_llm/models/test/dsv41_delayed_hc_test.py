"""Unfused delayed mHC and split-stage readout against checkpoint equations."""

import dataclasses
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source

# Load the real shared helper and its light dependencies without native package init.
with patch.dict(sys.modules):
    load_codec_source("rtp_llm.models_py.modules.dsv4._profiler", "_profiler.py")
    load_codec_source("rtp_llm.models_py.modules.dsv4.hc.base", "hc/base.py")
    load_codec_source(
        "rtp_llm.models_py.modules.dsv4.hc.fallback_impl", "hc/fallback_impl.py"
    )
    hc = load_codec_source("_dsv41_delayed_hc_test_source", "hc/delayed_reference.py")


def reference_mixes(x, unit):
    flat = x.flatten(-2).float()
    linear = (flat.unsqueeze(-2) * unit.fn).sum(-1)
    projection = linear / torch.sqrt(
        (flat * flat).sum(-1, keepdim=True) / flat.shape[-1] + unit.norm_eps
    )
    raw_pre = projection[..., :4] * unit.scale[0] + unit.base[:4]
    raw_post = projection[..., 4:8] * unit.scale[1] + unit.base[4:8]
    raw_comb = (projection[..., 8:] * unit.scale[2] + unit.base[8:]).reshape(
        *x.shape[:-2], 4, 4
    )
    pre = 1 / (1 + torch.exp(-raw_pre)) + unit.hc_eps
    post = 2 / (1 + torch.exp(-raw_post))
    scores = torch.exp(raw_comb - raw_comb.max(-1, keepdim=True).values)
    comb = scores / scores.sum(-1, keepdim=True) + unit.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + unit.hc_eps)
    for _ in range(unit.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + unit.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + unit.hc_eps)
    return pre, post, comb


def unit(dim=8):
    return hc.DelayedHCUnit(
        torch.randn(24, 4 * dim) * 0.1,
        torch.randn(24) * 0.1,
        torch.tensor([0.5, 0.6, 0.7]),
        dim=dim,
    )


class V41DelayedHCTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(414)

    def test_initial_readout_selects_lane_zero(self):
        for shape in ((3, 4, 8), (2, 3, 4, 8), (0, 4, 8)):
            x = torch.randn(shape, dtype=torch.bfloat16)
            initial = hc.identity_pre_mix(x)
            self.assertEqual(initial.dtype, torch.float32)
            torch.testing.assert_close(
                hc.collapse_delayed(x, initial), x[..., 0, :], rtol=0, atol=0
            )

    def test_current_projection_does_not_replace_predecessor_readout(self):
        current = unit()
        x = torch.randn(3, 4, 8)
        previous = torch.zeros(3, 4)
        previous[:, 2] = 1
        collapsed, mixes = current.pre(x, previous)
        torch.testing.assert_close(collapsed, x[:, 2], rtol=0, atol=0)
        self.assertFalse(
            torch.allclose(collapsed, hc.collapse_delayed(x, mixes.next_pre_mix))
        )
        self.assertFalse(hasattr(current, "pre_mix_out"))

    def test_mixes_and_post_follow_independent_checkpoint_equations(self):
        for shape in ((3, 4, 8), (2, 3, 4, 8)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                current = unit()
                x = torch.randn(shape, dtype=dtype)
                previous = torch.rand(shape[:-1])
                collapsed, actual = current.pre(x, previous)
                torch.testing.assert_close(
                    collapsed,
                    (x.float() * previous[..., None]).sum(-2).to(dtype),
                    rtol=0,
                    atol=0,
                )
                pre, post, comb = reference_mixes(x, current)
                for result, expected in (
                    (actual.next_pre_mix, pre),
                    (actual.post, post),
                    (actual.residual_mix, comb),
                ):
                    torch.testing.assert_close(result, expected, rtol=2e-6, atol=2e-6)
                output = torch.tanh(collapsed)
                expected = torch.stack(
                    [
                        actual.post[..., dest, None] * output.float()
                        + sum(
                            actual.residual_mix[..., src, dest, None]
                            * x[..., src, :].float()
                            for src in range(4)
                        )
                        for dest in range(4)
                    ],
                    dim=-2,
                ).to(dtype)
                torch.testing.assert_close(
                    current.post(output, x, actual), expected, rtol=0, atol=0
                )

    def test_sinkhorn_preserves_orientation_and_normalization(self):
        current = unit()
        x = torch.randn(4, 4, 8)
        _, mixes = current.pre(x, hc.identity_pre_mix(x))
        torch.testing.assert_close(
            mixes.residual_mix.sum(-1), torch.ones(4, 4), rtol=0, atol=5e-6
        )
        torch.testing.assert_close(
            mixes.residual_mix.sum(-2), torch.ones(4, 4), rtol=0, atol=5e-6
        )
        permutation = torch.eye(4)[[1, 2, 3, 0]].expand(4, -1, -1)
        permuted = dataclasses.replace(
            mixes, post=torch.zeros_like(mixes.post), residual_mix=permutation
        )
        result = current.post(torch.zeros(4, 8), x, permuted)
        torch.testing.assert_close(result, x[:, [3, 0, 1, 2]], rtol=0, atol=0)

    def test_split_attention_ffn_chain_and_final_readout(self):
        units = [unit() for _ in range(8)]
        x = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        initial = hc.identity_pre_mix(x)

        def run(hidden, previous, sequence):
            for current in sequence:
                collapsed, mixes = current.pre(hidden, previous)
                hidden = current.post(torch.tanh(collapsed), hidden, mixes)
                previous = mixes.next_pre_mix
            return hidden, previous

        full_hidden, full_mix = run(x, initial, units)
        # Boundary after a layer's FFN: two HC units per transformer layer.
        boundary_hidden, boundary_mix = run(x, initial, units[:4])
        split_hidden, split_mix = run(
            boundary_hidden.clone(), boundary_mix.clone(), units[4:]
        )
        torch.testing.assert_close(split_hidden, full_hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            hc.collapse_delayed(split_hidden, split_mix),
            hc.collapse_delayed(full_hidden, full_mix),
            rtol=0,
            atol=0,
        )
        wrong_hidden, wrong_mix = run(
            boundary_hidden, hc.identity_pre_mix(boundary_hidden), units[4:]
        )
        self.assertFalse(
            torch.equal(
                hc.collapse_delayed(wrong_hidden, wrong_mix),
                hc.collapse_delayed(full_hidden, full_mix),
            )
        )

    def test_interleaved_forwards_do_not_overwrite_coefficients(self):
        current = unit()
        first = torch.randn(2, 4, 8)
        second = torch.randn(5, 4, 8)
        y1, mix1 = current.pre(first, hc.identity_pre_mix(first))
        saved = mix1.next_pre_mix.clone()
        current.pre(second, hc.identity_pre_mix(second))
        torch.testing.assert_close(mix1.next_pre_mix, saved, rtol=0, atol=0)
        self.assertEqual(current.post(y1, first, mix1).shape, first.shape)

    def test_named_tensor_scale_shapes_normalize_once(self):
        original = unit()
        normalized = hc.DelayedHCUnit(
            original.fn, original.base[:, None], original.scale[:, None], dim=8
        )
        self.assertEqual(normalized.scale.shape, (3,))
        self.assertEqual(normalized.base.shape, (24,))

    def test_invalid_weights_residual_or_predecessor_fail(self):
        current = unit()
        x = torch.randn(2, 4, 8)
        for previous in (
            torch.ones(2, 4, dtype=torch.bfloat16),
            torch.ones(3, 4),
            torch.ones(2, 1),
        ):
            with self.assertRaises(ValueError):
                current.pre(x, previous)
        for bad in (
            torch.ones(2, 3, 8),
            torch.ones(2, 4, 9),
            torch.ones(2, 4, 8, dtype=torch.int32),
        ):
            with self.assertRaises(ValueError):
                current.pre(bad, torch.ones(2, 4))
        for changes in (
            dict(fn=current.fn.bfloat16()),
            dict(scale=torch.ones(4)),
            dict(base=torch.ones(23)),
            dict(norm_eps=0),
            dict(hc_eps=float("nan")),
            dict(hc_sinkhorn_iters=0),
        ):
            kwargs = dict(fn=current.fn, base=current.base, scale=current.scale, dim=8)
            with self.assertRaises(ValueError):
                hc.DelayedHCUnit(**{**kwargs, **changes})


if __name__ == "__main__":
    unittest.main()
