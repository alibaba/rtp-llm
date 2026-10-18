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
        with self.assertRaisesRegex(RuntimeError, "kernel API is not supported"):
            ACTIVATION.situ_kwargs(unverified, 4.0, None)

    def test_old_alpha_beta_api_is_rejected(self):
        def native(*, activation_alpha, activation_beta):
            raise AssertionError("unsupported kernel must not execute")
        for up_beta in (None, 25.0):
            with self.subTest(up_beta=up_beta), self.assertRaisesRegex(
                RuntimeError, "kernel API is not supported"
            ):
                ACTIVATION.situ_kwargs(native, 4.0, up_beta)

    def test_legacy_situ_names_are_rejected(self):
        def legacy(*, situ_beta, situ_linear_beta):
            raise AssertionError("unsupported kernel must not execute")
        for up_beta in (None, 25.0):
            with self.subTest(up_beta=up_beta), self.assertRaisesRegex(
                RuntimeError, "kernel API is not supported"
            ):
                ACTIVATION.situ_kwargs(legacy, 4.0, up_beta)

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
