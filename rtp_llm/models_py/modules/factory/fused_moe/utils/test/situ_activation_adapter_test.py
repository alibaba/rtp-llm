"""Check coefficient semantics across the supported SiTU Python interfaces."""

import math
import unittest

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.activation import (
    situ_kwargs,
)


def activation_beta_kernel(activation_beta=None, activation_linear_beta=None):
    return activation_beta, activation_linear_beta


def situ_beta_kernel(situ_beta=None, situ_linear_beta=None):
    return situ_beta, situ_linear_beta


def alpha_beta_kernel(activation_alpha=None, activation_beta=0.0):
    return activation_alpha, activation_beta


class SituActivationAdapterTest(unittest.TestCase):
    def test_gate_and_up_scales_keep_their_roles(self):
        # Distinct coefficients catch a swap between the gate and up branches.
        # SiTU uses beta*tanh(gate/beta)*sigmoid(gate), times
        # linear_beta*tanh(up/linear_beta) when the up branch is saturated.
        gate, up, gate_beta, up_beta = 8.0, 50.0, 4.0, 25.0
        expected = (
            4.0 * math.tanh(gate / 4.0) / (1.0 + math.exp(-gate))
            * 25.0 * math.tanh(up / 25.0)
        )
        for kernel in (activation_beta_kernel, situ_beta_kernel, alpha_beta_kernel):
            with self.subTest(interface=kernel.__name__):
                beta, linear_beta = kernel(**situ_kwargs(kernel, gate_beta, up_beta))
                self.assertEqual((beta, linear_beta), (4.0, 25.0))
                actual = (
                    beta * math.tanh(gate / beta) / (1.0 + math.exp(-gate))
                    * linear_beta * math.tanh(up / linear_beta)
                )
                self.assertAlmostEqual(actual, expected)

    def test_optional_up_saturation(self):
        self.assertEqual(
            situ_kwargs(activation_beta_kernel, 4.0, None),
            {"activation_beta": 4.0, "activation_linear_beta": None},
        )
        self.assertEqual(
            situ_kwargs(alpha_beta_kernel, 4.0, None),
            {"activation_alpha": 4.0, "activation_beta": 0.0},
        )
        with self.assertRaisesRegex(RuntimeError, "saturated up branch"):
            situ_kwargs(situ_beta_kernel, 4.0, None)

    def test_single_coefficient_api_is_rejected(self):
        def kernel(activation_alpha=None):
            pass

        with self.assertRaisesRegex(RuntimeError, "both SiTU coefficients"):
            situ_kwargs(kernel, 4.0, 25.0)

    def test_invalid_coefficients(self):
        for value in (None, 0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(gate=value), self.assertRaises(ValueError):
                situ_kwargs(activation_beta_kernel, value, 25.0)
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(up=value), self.assertRaises(ValueError):
                situ_kwargs(activation_beta_kernel, 4.0, value)


if __name__ == "__main__":
    unittest.main()
