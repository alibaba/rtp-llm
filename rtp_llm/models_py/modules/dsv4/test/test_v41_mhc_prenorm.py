"""V4.1 split-K mHC prenorm: TF32 contract, delayed mixes and graph replay."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import (
    DelayedHCUnit,
    _tile_ops,
    collapse_delayed,
)
from rtp_llm.models_py.modules.dsv4.hc.v41_prenorm import (
    is_supported,
    prenorm,
    prepare_tf32_weight,
)

_EPS = 1e-6


def _make_unit(fn, base, scale):
    return DelayedHCUnit(
        fn,
        base,
        scale,
        dim=5120,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=_EPS,
        hc_eps=_EPS,
    )


def _old_mixes(residual, fn):
    return _tile_ops().mhc_pre_norm_fn(residual, fn, None, _EPS, n_splits=1)


def _split_mixes(mixes, base, scale):
    # The existing coefficient kernel requires [batch, tokens, 24].
    ops = _tile_ops()
    pre, post, comb = ops.mhc_pre_split_mixes(
        mixes.reshape(1, -1, 24), scale, base, 4, 2.0, _EPS
    )
    return pre, post, ops.sinkhorn_normalize(comb, repeat=20, eps=_EPS)


class V41MHCPreNormCPUContractTest(unittest.TestCase):
    def test_tf32_tie_rule_and_weight_cache_invalidation(self):
        # RTP adds half a TF32 ULP before truncation, including negative ties;
        # this differs from round-to-nearest-even at the exact halfway value.
        bits = torch.tensor(
            [0x3F800FFF, 0x3F801000, -1082126337, -1082126336, 0, -2147483648],
            dtype=torch.int32,
        )
        weight = bits.view(torch.float32).reshape(2, 3)
        expected = torch.tensor(
            [0x3F800000, 0x3F802000, -1082130432, -1082122240, 0, -2147483648],
            dtype=torch.int32,
        ).reshape(2, 3)
        prepared = prepare_tf32_weight(weight)
        torch.testing.assert_close(prepared.view(torch.int32), expected, rtol=0, atol=0)
        self.assertIs(prepare_tf32_weight(weight), prepared)
        weight.add_(0.125)
        changed = prepare_tf32_weight(weight)
        self.assertIsNot(changed, prepared)
        torch.testing.assert_close(
            changed.view(torch.int32),
            (weight.view(torch.int32) + 0x1000) & -8192,
            rtol=0,
            atol=0,
        )
        replacement = weight.clone()
        replaced = prepare_tf32_weight(replacement)
        self.assertNotEqual(replaced.data_ptr(), changed.data_ptr())
        torch.testing.assert_close(replaced, changed, rtol=0, atol=0)

    def test_inference_weight_cache_and_replacement(self):
        # Loaded inference tensors have no version counter and are immutable.
        # Replacement is supported; in-place changes are intentionally outside
        # this contract, just as in the existing TileLang weight cache.
        with torch.inference_mode():
            weight = torch.tensor([[0.1, -0.3, 1.5]], dtype=torch.float32)
            replacement = weight.clone().mul_(0.5)
        with self.assertRaises(RuntimeError):
            _ = weight._version
        prepared = prepare_tf32_weight(weight)
        self.assertIs(prepare_tf32_weight(weight), prepared)
        replaced = prepare_tf32_weight(replacement)
        self.assertIs(prepare_tf32_weight(replacement), replaced)
        self.assertNotEqual(prepared.data_ptr(), replaced.data_ptr())
        for raw, result in ((weight, prepared), (replacement, replaced)):
            torch.testing.assert_close(
                result.view(torch.int32),
                (raw.view(torch.int32) + 0x1000) & -8192,
                rtol=0,
                atol=0,
            )

    def test_cpu_gate_returns_none_without_loading_gemm(self):
        residual = torch.empty(6, 4, 5120, dtype=torch.bfloat16)
        weight = torch.empty(24, 20480)
        with patch(
            "rtp_llm.models_py.modules.dsv4.hc.v41_prenorm._has_prenorm_gemm"
        ) as available:
            self.assertFalse(is_supported(residual, weight))
            self.assertIsNone(prenorm(residual, weight, _EPS))
            available.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41MHCPreNormCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("V4.1 prenorm fast path requires SM100")

    def setUp(self):
        self.env = patch.dict(os.environ, {"DSV41_FUSED_MHC_PRENORM": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(41)

    @staticmethod
    def _inputs(tokens):
        return (
            torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16),
            torch.randn(24, 20480, device="cuda") * 0.003,
            torch.randn(24, device="cuda") * 0.1,
            torch.tensor([0.2, 0.4, 0.3], device="cuda"),
        )

    def _assert_mixes_close(self, actual, expected):
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        # Split-K changes FP32 accumulation order. Bound both every element
        # and aggregate error; do not relax the existing full-HC graph tests.
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)
        rms = torch.mean((actual.double() - expected.double()).square()).sqrt()
        scale = torch.mean(expected.double().square()).sqrt()
        self.assertLessEqual(rms.item(), 1e-6 + 1e-4 * scale.item())

    @torch.no_grad()
    def _check_mixes_and_coefficients(self, residual, fn, base, scale):
        expected = _old_mixes(residual, fn)
        actual = prenorm(residual, fn, _EPS)
        self.assertIsNotNone(actual)
        self._assert_mixes_close(actual, expected)
        for got, ref in zip(
            _split_mixes(actual, base, scale), _split_mixes(expected, base, scale)
        ):
            torch.testing.assert_close(got, ref, rtol=2e-4, atol=1e-5)
        return actual

    @torch.no_grad()
    def test_token_boundaries_and_3d_4d_layouts(self):
        for tokens in (1, 6, 24, 31, 64):
            residual, fn, base, scale = self._inputs(tokens)
            batch = 4 if tokens in (24, 64) else 1
            for layout in (residual, residual.reshape(batch, -1, 4, 5120)):
                with self.subTest(tokens=tokens, shape=tuple(layout.shape)):
                    self.assertTrue(is_supported(layout, fn))
                    actual = self._check_mixes_and_coefficients(layout, fn, base, scale)
                    self.assertEqual(actual.shape, (*layout.shape[:-2], 24))

    @torch.no_grad()
    def test_zero_epsilon_and_large_amplitude(self):
        residual, fn, base, scale = self._inputs(6)
        for amplitude in (0.0, 1e-5, 1.0, 1024.0):
            with self.subTest(amplitude=amplitude):
                source = (residual.float() * amplitude).bfloat16()
                actual = self._check_mixes_and_coefficients(source, fn, base, scale)
                if amplitude == 0:
                    torch.testing.assert_close(
                        actual, torch.zeros_like(actual), rtol=0, atol=0
                    )

    @torch.no_grad()
    def test_delayed_entry_and_previous_mix_semantics(self):
        residual, fn, base, scale = self._inputs(6)
        residual = residual.reshape(1, 6, 4, 5120)
        observations = []
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"DSV41_FUSED_MHC_PRENORM": enabled}):
                first = _make_unit(fn, base, scale)
                second = _make_unit(fn * -0.7, base + 0.4, scale)
                second.set_previous(first)
                first_y, first_post, first_comb = first.pre(residual)
                torch.testing.assert_close(first_y, residual[..., 0, :], rtol=0, atol=0)
                second_y, second_post, second_comb = second.pre(residual)
                expected_y = collapse_delayed(residual, first.pre_mix_out)
                torch.testing.assert_close(second_y, expected_y, rtol=0, atol=0)
                own_y = collapse_delayed(residual, second.pre_mix_out)
                self.assertFalse(torch.equal(second_y, own_y))
                observations.append(
                    (
                        first.pre_mix_out,
                        first_post,
                        first_comb,
                        second.pre_mix_out,
                        second_post,
                        second_comb,
                        second_y,
                    )
                )
        for actual, expected in zip(observations[1][:-1], observations[0][:-1]):
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-5)
        # Keep the pre-existing full delayed-HC graph tolerance unchanged.
        torch.testing.assert_close(
            observations[1][-1], observations[0][-1], rtol=0.02, atol=0.03125
        )

    @torch.no_grad()
    def test_graph_replay_changes_inputs(self):
        for tokens in (1, 6, 24):
            with self.subTest(tokens=tokens):
                residual, fn, base, scale = self._inputs(tokens)
                with torch.inference_mode():
                    fn = fn.clone()
                with self.assertRaises(RuntimeError):
                    _ = fn._version
                for _ in range(3):
                    mixes = prenorm(residual, fn, _EPS)
                    _split_mixes(mixes, base, scale)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = prenorm(residual, fn, _EPS)
                    coefficients = _split_mixes(output, base, scale)
                graph.replay()
                initial = output.clone()
                for amplitude in (-0.75, 2.0):
                    residual.copy_(torch.randn_like(residual) * amplitude)
                    graph.replay()
                    expected = _old_mixes(residual, fn)
                    self._assert_mixes_close(output, expected)
                    self.assertFalse(torch.equal(output, initial))
                    torch.testing.assert_close(
                        output, prenorm(residual, fn, _EPS), rtol=0, atol=0
                    )
                    for actual, reference in zip(
                        coefficients, _split_mixes(expected, base, scale)
                    ):
                        torch.testing.assert_close(
                            actual, reference, rtol=2e-4, atol=1e-5
                        )

    @torch.no_grad()
    def test_weight_inplace_update_and_replacement_reach_gemm(self):
        residual, fn, base, scale = self._inputs(6)
        before = prenorm(residual, fn, _EPS)
        prepared = prepare_tf32_weight(fn)
        fn.mul_(-0.5)
        changed = self._check_mixes_and_coefficients(residual, fn, base, scale)
        self.assertIsNot(prepare_tf32_weight(fn), prepared)
        self.assertFalse(torch.equal(before, changed))
        replacement = fn.clone().add_(0.001)
        replaced = self._check_mixes_and_coefficients(
            residual, replacement, base, scale
        )
        self.assertFalse(torch.equal(changed, replaced))

    @torch.no_grad()
    def test_gate_env_layout_dtype_device_and_large_tokens(self):
        residual, fn, _, _ = self._inputs(6)
        self.assertTrue(is_supported(residual, fn))
        cases = [
            (residual.float(), fn),
            (residual.cpu(), fn),
            (residual, fn.cpu()),
            (residual, fn.bfloat16()),
            (residual[::2], fn),
            (residual, fn.T.contiguous().T),
            (residual, fn[:23]),
            (residual[:, :3], fn),
            (residual.unsqueeze(0).unsqueeze(0), fn),
            (torch.empty(65, 4, 5120, device="cuda", dtype=torch.bfloat16), fn),
        ]
        for source, weight in cases:
            with self.subTest(
                shape=tuple(source.shape), dtype=source.dtype, device=source.device
            ):
                self.assertFalse(is_supported(source, weight))
                self.assertIsNone(prenorm(source, weight, _EPS))
        with patch.dict(os.environ, {"DSV41_FUSED_MHC_PRENORM": "0"}):
            self.assertFalse(is_supported(residual, fn))
            self.assertIsNone(prenorm(residual, fn, _EPS))
        with patch(
            "rtp_llm.models_py.modules.dsv4.hc.v41_prenorm._has_prenorm_gemm",
            return_value=False,
        ):
            self.assertIsNone(prenorm(residual, fn, _EPS))
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            self.assertFalse(is_supported(residual, fn))
        with torch.enable_grad():
            self.assertFalse(is_supported(residual, fn.clone().requires_grad_()))
        empty = prenorm(residual[:0], fn, _EPS)
        self.assertEqual(tuple(empty.shape), (0, 24))
        self.assertEqual(empty.dtype, torch.float32)

    @torch.no_grad()
    def test_gemm_failure_is_not_silently_fallback(self):
        residual, fn, _, _ = self._inputs(1)
        with patch(
            "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.tf32_hc_prenorm_gemm",
            side_effect=RuntimeError("injected prenorm GEMM failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected prenorm GEMM failure"):
                prenorm(residual, fn, _EPS)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_CHECKPOINT"),
        "set DSV41_TEST_CHECKPOINT for real weights",
    )
    @torch.no_grad()
    def test_real_checkpoint_attention_and_ffn_weights(self):
        from safetensors import safe_open

        checkpoint = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for prefix in ("layers.0.hc_attn", "layers.39.hc_ffn"):
            with self.subTest(prefix=prefix):
                weights = []
                for suffix in ("fn", "base", "scale"):
                    name = f"{prefix}_{suffix}"
                    with safe_open(
                        str(checkpoint / index[name]), framework="pt", device="cpu"
                    ) as source:
                        weights.append(source.get_tensor(name).float().cuda())
                fn, base, scale = weights
                for tokens in (1, 6, 24):
                    residual = self._inputs(tokens)[0].reshape(1, tokens, 4, 5120)
                    old = _old_mixes(residual, fn)
                    actual = prenorm(residual, fn, _EPS)
                    self.assertIsNotNone(actual)
                    # Build the reference independently of the production
                    # weight-preparation function. Long-K FP32 summation in
                    # the old kernel has larger cancellation error than DG:
                    # per-element comparison to that result is not an oracle.
                    rounded = ((fn.view(torch.int32) + 0x1000) & -8192).view(
                        torch.float32
                    )
                    x64 = residual.reshape(tokens, 20480).double()
                    oracle = (
                        (x64 @ rounded.double().T)
                        * torch.rsqrt(x64.square().mean(-1, keepdim=True) + _EPS)
                    ).reshape_as(actual)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    reference_rms = oracle.square().mean().sqrt().item()
                    error = actual.double() - oracle
                    self.assertLessEqual(
                        error.square().mean().sqrt().item(),
                        1e-7 + 1e-5 * reference_rms,
                    )
                    self.assertLessEqual(
                        error.abs().max().item(), 1e-7 + 2e-5 * reference_rms
                    )
                    old_error = actual.double() - old.double()
                    self.assertLessEqual(
                        old_error.square().mean().sqrt().item(),
                        1e-6 + 1e-4 * reference_rms,
                    )
                    for got, ref in zip(
                        _split_mixes(actual, base.reshape(24), scale.reshape(3)),
                        _split_mixes(old, base.reshape(24), scale.reshape(3)),
                    ):
                        torch.testing.assert_close(got, ref, rtol=2e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
