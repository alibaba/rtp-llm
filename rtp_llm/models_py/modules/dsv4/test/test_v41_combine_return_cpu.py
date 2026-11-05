"""CPU tests for the R25/F1 fused combine+return flag and guard fallbacks.

Hermetic namespace-stub loading; the kernel itself is GPU-only and is
gated by r25_f1_bitwise.py + the parity sha instead.
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

fused = importlib.import_module(
    "rtp_llm.models_py.modules.dsv4._v41_combine_return_triton"
)
ret = importlib.import_module("rtp_llm.models_py.modules.dsv4.v41_return")


class FlagTest(unittest.TestCase):
    def test_value_based_default_on(self):
        # R27: adopted as default (escape = explicit "0").
        env = dict(os.environ)
        env.pop(fused.COMBINE_RETURN_FUSED_FLAG, None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(fused.combine_return_fused_enabled())
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("", False),
        ):
            with mock.patch.dict(
                os.environ, {fused.COMBINE_RETURN_FUSED_FLAG: value}, clear=False
            ):
                self.assertEqual(fused.combine_return_fused_enabled(), expected, value)


class GuardFallbackTest(unittest.TestCase):
    def test_cpu_inputs_fall_back(self):
        down = torch.zeros((4, 5120), dtype=torch.bfloat16)
        rm = torch.zeros((4, 6), dtype=torch.int64)
        layout = ret.V41ReturnLayout(5120, "mxfp8")
        self.assertIsNone(
            fused.combine_return_fused(down, rm, layout, _validated_map=True)
        )

    def test_fp32_layout_falls_back(self):
        down = torch.zeros((4, 5120), dtype=torch.bfloat16)
        rm = torch.zeros((4, 6), dtype=torch.int64)
        layout = ret.V41ReturnLayout(5120, "fp32")
        self.assertIsNone(
            fused.combine_return_fused(down, rm, layout, _validated_map=True)
        )

    def test_bad_topk_falls_back(self):
        down = torch.zeros((4, 5120), dtype=torch.bfloat16)
        rm = torch.zeros((4, 33), dtype=torch.int64)
        layout = ret.V41ReturnLayout(5120, "mxfp8")
        self.assertIsNone(
            fused.combine_return_fused(down, rm, layout, _validated_map=True)
        )

    def test_layout_payload_bytes(self):
        layout = ret.V41ReturnLayout(5120, "mxfp8")
        self.assertEqual(layout.payload_bytes, 5120 + 160)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
