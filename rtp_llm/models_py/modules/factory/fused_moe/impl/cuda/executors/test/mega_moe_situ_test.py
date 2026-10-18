import math
import unittest

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    situ_kwargs,
)


def _modern_situ_kernel(
    gate, up, *, activation_beta=None, activation_linear_beta=None
):
    """CPU model of DeepGEMM's activation boundary, excluding GEMMs/rounding.

    The scale/None contract is from DeepGEMM d029add40cdc7afce6e7ed22c81ee32b835deea9:
    csrc/apis/mega.hpp and
    deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh.
    This fixture validates parameter binding, not CUDA kernel execution.
    """
    gate_scale = 1.0 if activation_beta is None else activation_beta
    if gate_scale <= 0:
        raise ValueError("gate scale must be positive")
    if activation_linear_beta is not None and activation_linear_beta <= 0:
        raise ValueError("up scale must be positive or None")
    activated_gate = (
        gate_scale * math.tanh(gate / gate_scale) / (1.0 + math.exp(-gate))
    )
    activated_up = up
    if activation_linear_beta is not None:
        activated_up = activation_linear_beta * math.tanh(
            up / activation_linear_beta
        )
    return activated_gate * activated_up


class MegaMoeSiTUTest(unittest.TestCase):
    def test_binding_preserves_independent_gate_and_up_saturation(self):
        # Analytic oracle: tanh(log(2))=3/5, sigmoid(log(4))=4/5,
        # tanh(log(3))=4/5. With gate beta=2 and up beta=5 the
        # positive product is (2*3/5*4/5)*(5*4/5)=96/25.
        # The asymmetric scales detect swapped bindings or sigmoid slopes.
        kwargs = situ_kwargs(_modern_situ_kernel, 2.0, 5.0)
        cases = (
            (math.log(4), 5 * math.log(3), 96 / 25),
            (-math.log(4), 5 * math.log(3), -24 / 25),
            (math.log(4), -5 * math.log(3), -96 / 25),
            (0.0, 5 * math.log(3), 0.0),
        )
        for gate, up, expected in cases:
            with self.subTest(gate=gate, up=up):
                self.assertAlmostEqual(
                    _modern_situ_kernel(gate, up, **kwargs), expected, places=12
                )

    def test_none_keeps_up_unsaturated_instead_of_using_zero_sentinel(self):
        kwargs = situ_kwargs(_modern_situ_kernel, 2.0, None)
        self.assertIsNone(kwargs["activation_linear_beta"])
        # The gate is 24/25; up remains the original 5*log(3).
        actual = _modern_situ_kernel(math.log(4), 5 * math.log(3), **kwargs)
        self.assertAlmostEqual(actual, (24 / 5) * math.log(3), places=12)
        self.assertGreater(actual, 96 / 25)
        with self.assertRaisesRegex(ValueError, "up scale"):
            _modern_situ_kernel(
                math.log(4),
                5 * math.log(3),
                activation_beta=2.0,
                activation_linear_beta=0.0,
            )

    def test_modern_api_wins_when_alpha_is_also_present(self):
        def kernel(
            *,
            activation_alpha=1.0,
            activation_beta=None,
            activation_linear_beta=None,
        ):
            self.assertEqual(activation_alpha, 1.0)
            return _modern_situ_kernel(
                math.log(4),
                5 * math.log(3),
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
            )

        self.assertAlmostEqual(kernel(**situ_kwargs(kernel, 2.0, 5.0)), 96 / 25)

    def test_unverified_or_incomplete_apis_are_rejected(self):
        def old_alpha_beta(*, activation_alpha=None, activation_beta=None):
            raise AssertionError("unsupported kernel must not execute")

        def old_situ_names(*, situ_beta=None, situ_linear_beta=None):
            raise AssertionError("unsupported kernel must not execute")

        def missing_linear(*, activation_beta=None, **kwargs):
            raise AssertionError("unsupported kernel must not execute")

        def generic_kwargs(**kwargs):
            raise AssertionError("unsupported kernel must not execute")

        def positional_only(activation_beta, activation_linear_beta, /):
            raise AssertionError("unsupported kernel must not execute")

        for kernel in (
            old_alpha_beta,
            old_situ_names,
            missing_linear,
            generic_kwargs,
            positional_only,
        ):
            with self.subTest(kernel=kernel.__name__):
                with self.assertRaisesRegex(RuntimeError, "kernel API is not supported"):
                    situ_kwargs(kernel, 2.0, 5.0)

    def test_invalid_gate_scale_is_rejected(self):
        for value in (None, 0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "gate beta"):
                    situ_kwargs(_modern_situ_kernel, value, 5.0)

    def test_invalid_up_scale_is_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "up beta"):
                    situ_kwargs(_modern_situ_kernel, 2.0, value)


if __name__ == "__main__":
    unittest.main()
