"""Default CUDA dispatch and isolation of explicitly bound PPU operators."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "dsv4_provider", Path(__file__).parents[1] / "models/dsv4/impl/platform_provider.py"
)
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)
C = provider.Dsv4ProviderCapability


class PlatformProviderTest(unittest.TestCase):
    def test_default_factories_preserve_arguments_and_identity(self):
        sentinel = object()
        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        for name in (
            "build_dsv4_fp8_linear", "build_dsv4_wo_a_fp8_linear",
            "build_dsv4_decode_metadata", "run_dsv4_fp8_mqa_logits",
        ):
            self.assertIs(getattr(provider, name)(original, 3, flag=7), sentinel)
            self.assertEqual(calls[-1], ((3,), {"flag": 7}))
        self.assertIs(provider.build_dsv4_bf16_fp32_linear(original), original)

    def test_bound_operations_do_not_change_other_models(self):
        first = SimpleNamespace(capabilities={C.BF16_FP32_LINEAR})
        second = SimpleNamespace(capabilities={C.BF16_FP32_LINEAR})
        first.run_bf16_fp32_linear = lambda default, x: ("first", x)
        second.run_bf16_fp32_linear = lambda default, x: ("second", x)
        default = lambda x: ("cuda", x)
        a = provider.build_dsv4_bf16_fp32_linear(default, platform_provider=first)
        b = provider.build_dsv4_bf16_fp32_linear(default, platform_provider=second)
        self.assertEqual(a(1), ("first", 1))
        self.assertEqual(b(2), ("second", 2))
        self.assertIs(provider.build_dsv4_bf16_fp32_linear(default), default)
        self.assertEqual(a(3), ("first", 3))

    def test_falsey_provider_is_not_replaced(self):
        class FalseyProvider:
            capabilities = {C.FP8_LINEAR}

            def __bool__(self):
                return False

            def build_fp8_linear(self, default, *args, **kwargs):
                return self, args, kwargs

        selected = FalseyProvider()
        result = provider.build_dsv4_fp8_linear(
            None, "weight", platform_provider=selected, scale=3
        )
        self.assertEqual(result, (selected, ("weight",), {"scale": 3}))

    def test_layout_and_required_operation_contracts(self):
        default = provider.resolve_dsv4_platform_provider({C.BLOCK, C.TRANSFORMER})
        self.assertEqual(provider.resolve_dsv4_attention_layout(default).value, "flat")
        padded = SimpleNamespace(capabilities={C.ATTENTION}, attention_layout="padded")
        self.assertEqual(provider.resolve_dsv4_attention_layout(padded).value, "padded")
        with self.assertRaises(RuntimeError):
            provider.run_dsv4_hc_prenorm("input")


if __name__ == "__main__":
    unittest.main()
