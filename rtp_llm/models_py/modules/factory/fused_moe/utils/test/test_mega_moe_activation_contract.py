"""SiTU parameter and matched workspace ABI contracts; no GPU required."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "mega_moe" / "activation.py"
SPEC = importlib.util.spec_from_file_location("mega_moe_activation_contract", SOURCE)
ACTIVATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTIVATION)


class MegaMoeActivationContractTest(unittest.TestCase):
    def test_gate_and_up_scales_preserve_distinct_meanings(self):
        def kernel(gate, up, *, activation_beta, activation_linear_beta):
            g = activation_beta * math.tanh(gate / activation_beta)
            g /= 1.0 + math.exp(-gate)
            u = up if activation_linear_beta is None else (
                activation_linear_beta * math.tanh(up / activation_linear_beta)
            )
            return g * u

        for up_beta in (None, 25.0):
            kwargs = ACTIVATION.situ_kwargs(kernel, 4.0, up_beta)
            for gate, up in ((0.0, 0.0), (-8.0, 50.0), (12.0, -30.0)):
                expected = 4.0 * math.tanh(gate / 4.0) / (1 + math.exp(-gate))
                expected *= up if up_beta is None else 25.0 * math.tanh(up / 25.0)
                self.assertEqual(kernel(gate, up, **kwargs), expected)

    def test_incomplete_activation_signature_is_rejected(self):
        def unverified(*, activation_alpha):
            pass
        with self.assertRaisesRegex(RuntimeError, "verified SiTU"):
            ACTIVATION.situ_kwargs(unverified, 4.0, None)

    def test_verified_native_zero_sentinel_keeps_unsaturated_up(self):
        def native(gate, up, *, activation_alpha, activation_beta):
            bounded_up = up if activation_beta == 0 else (
                activation_beta * math.tanh(up / activation_beta)
            )
            return activation_alpha * math.tanh(gate / activation_alpha) * bounded_up
        for up_beta in (None, 25.0):
            kwargs = ACTIVATION.situ_kwargs(native, 4.0, up_beta)
            expected = 4.0 * math.tanh(3.0 / 4.0)
            expected *= 30.0 if up_beta is None else 25.0 * math.tanh(30.0 / 25.0)
            self.assertEqual(native(3.0, 30.0, **kwargs), expected)

    def test_legacy_situ_names_keep_the_saturated_up_contract(self):
        def legacy(*, situ_beta, situ_linear_beta):
            pass
        self.assertEqual(ACTIVATION.situ_kwargs(legacy, 4.0, 25.0),
                         {"situ_beta": 4.0, "situ_linear_beta": 25.0})
        with self.assertRaisesRegex(RuntimeError, "saturated up"):
            ACTIVATION.situ_kwargs(legacy, 4.0, None)

    def test_invalid_betas_are_rejected(self):
        def kernel(*, activation_beta, activation_linear_beta):
            pass
        for gate, up in ((0, 25), (-1, 25), (float("nan"), 25),
                         (4, 0), (4, -1), (4, float("inf"))):
            with self.subTest(gate=gate, up=up), self.assertRaises(ValueError):
                ACTIVATION.situ_kwargs(kernel, gate, up)

    def test_kernel_transform_and_workspace_use_the_same_abi(self):
        backend = SimpleNamespace()
        names = ("fp8_fp4_mega_moe", "transform_weights_for_mega_moe",
                 "get_symm_buffer_for_mega_moe")
        for name in names:
            setattr(backend, name, lambda: "generic")
            setattr(backend, name + "_k3", lambda: "situ")
        for activation, suffix in (("swiglu", ""), ("situ", "_k3")):
            self.assertEqual(ACTIVATION.mega_moe_functions(backend, activation),
                             tuple(getattr(backend, name + suffix) for name in names))

    def test_incomplete_specialized_abi_is_not_mixed_with_generic_layout(self):
        backend = SimpleNamespace(fp8_fp4_mega_moe_k3=lambda: None,
                                  transform_weights_for_mega_moe=lambda: None,
                                  get_symm_buffer_for_mega_moe=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "matching MegaMoE"):
            ACTIVATION.mega_moe_functions(backend, "situ")


if __name__ == "__main__":
    unittest.main()
