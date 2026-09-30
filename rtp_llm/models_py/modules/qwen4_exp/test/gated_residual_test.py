import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import (
    Qwen4ExpGatedResidual,
    grouped_rms_norm,
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_merged_down_epilogue_preserves_bf16_rounding_and_graph_replay(self):
        from rtp_llm.models_py.modules.qwen4_exp.gated_residual_down_triton import (
            down_inject_epilogue,
        )

        source = torch.randn(17, 336, device="cuda", dtype=torch.bfloat16) * 4

        def reference():
            return (
                F.silu(source[:, :320] / 4),
                2 * torch.sigmoid(source[:, 320:324] / 4),
            )

        expected = reference()
        actual = down_inject_epilogue(source, 320, 4)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = down_inject_epilogue(source, 320, 4)
        source.copy_(
            torch.linspace(-128, 128, source.numel(), device="cuda").reshape_as(source)
        )
        graph.replay()
        for got, want in zip(actual, reference()):
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_merged_projection_matches_original_at_real_width(self):
        torch.manual_seed(43)
        width, rank = 4 * 2560, 320
        gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
        down = torch.randn(rank, width, device="cuda", dtype=torch.bfloat16) * 0.02
        up = torch.randn(width, rank, device="cuda", dtype=torch.bfloat16) * 0.02
        inject = torch.randn(4, width, device="cuda", dtype=torch.bfloat16) * 0.02
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_MERGED_DOWN_INJECT": "0"}):
            old = Qwen4ExpGatedResidual(
                gamma, down, up, inject, hc_mult=4, norm_eps=_EPS
            )
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_MERGED_DOWN_INJECT": "1"}):
            new = Qwen4ExpGatedResidual(
                gamma, down, up, inject, hc_mult=4, norm_eps=_EPS
            )
        self.assertEqual(new.merged_down_inject.shape, (336, width))
        for rows in (1, 8, 17, 33, 128):
            with self.subTest(rows=rows):
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                expected = old(x)
                actual = new(x)
                self.assertIs(actual[1], x)
                for got, want in ((actual[0], expected[0]), (actual[2], expected[2])):
                    torch.testing.assert_close(got, want, atol=8e-3, rtol=1e-2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = new(x)
        x.normal_()
        graph.replay()
        expected = old(x)
        for got, want in ((actual[0], expected[0]), (actual[2], expected[2])):
            torch.testing.assert_close(got, want, atol=8e-3, rtol=1e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_group_norm_matches_torch_and_replays(self):
        width = 4 * 2560
        gamma = (torch.randn(width, device="cuda") * 0.1).bfloat16()
        source = torch.randn(4, width, device="cuda", dtype=torch.bfloat16)

        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_GROUP_NORM": "0"}):
            reference = grouped_rms_norm(source, gamma, 2560, _EPS)
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_GROUP_NORM": "1"}):
            actual = grouped_rms_norm(source, gamma, 2560, _EPS)
            torch.testing.assert_close(
                actual.float(), reference.float(), atol=0.016, rtol=0.01
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = grouped_rms_norm(source, gamma, 2560, _EPS)
            source.copy_(torch.randn_like(source))
            with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_GROUP_NORM": "0"}):
                reference = grouped_rms_norm(source, gamma, 2560, _EPS)
            graph.replay()
            torch.testing.assert_close(
                replayed.float(), reference.float(), atol=0.016, rtol=0.01
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_mix_reduce_matches_torch_and_replays(self):
        from rtp_llm.models_py.modules.qwen4_exp.gated_residual_mix_triton import (
            fused_mix_reduce,
        )

        width = 4 * 2560
        for rows in (1, 8, 17):
            with self.subTest(rows=rows):
                logits = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                normed = torch.randn_like(logits)
                reference = (
                    torch.sigmoid(logits).reshape(rows, 4, 2560)
                    * normed.reshape(rows, 4, 2560)
                ).mean(1)
                actual = fused_mix_reduce(logits, normed, 4)
                torch.testing.assert_close(
                    actual.float(), reference.float(), atol=0.016, rtol=0.01
                )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = fused_mix_reduce(logits, normed, 4)
        logits.copy_(torch.randn_like(logits))
        normed.copy_(torch.randn_like(normed))
        reference = (
            torch.sigmoid(logits).reshape(17, 4, 2560) * normed.reshape(17, 4, 2560)
        ).mean(1)
        graph.replay()
        torch.testing.assert_close(
            replayed.float(), reference.float(), atol=0.016, rtol=0.01
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_inject_matches_torch_and_replays(self):
        for rows in (1, 8, 17):
            with self.subTest(rows=rows):
                hyper = torch.randn(rows, 4 * 2560, device="cuda", dtype=torch.bfloat16)
                sublayer = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                weights = torch.rand(rows, 4, device="cuda", dtype=torch.bfloat16)
                with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_INJECT": "0"}):
                    reference = inject_into_residual(hyper, sublayer, weights)
                with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_INJECT": "1"}):
                    actual = inject_into_residual(hyper, sublayer, weights)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)

        graph = torch.cuda.CUDAGraph()
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_INJECT": "1"}):
            with torch.cuda.graph(graph):
                replayed = inject_into_residual(hyper, sublayer, weights)
        hyper.copy_(torch.randn_like(hyper))
        sublayer.copy_(torch.randn_like(sublayer))
        weights.copy_(torch.rand_like(weights))
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_INJECT": "0"}):
            reference = inject_into_residual(hyper, sublayer, weights)
        graph.replay()
        torch.testing.assert_close(replayed, reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
