"""CPU tests for R30/A DSV41_MHC_MEGA_FUSED (flag matrix).

Hermetic namespace-stub loading; the kernel is GPU-only (gated on the
node by r30_a_bitwise.py + the parity sha).
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[5]
for suffix in ["", ".models_py", ".models_py.modules", ".models_py.modules.dsv4"]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

mega = importlib.import_module("rtp_llm.models_py.modules.dsv4._v41_mhc_mega_fused")


class MegaFusedFlagTest(unittest.TestCase):
    def setUp(self):
        mega._ANNOUNCED[0] = False

    def test_absent_is_off(self):
        env = dict(os.environ)
        env.pop("DSV41_MHC_MEGA_FUSED", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(mega.mhc_mega_fused_enabled())

    def test_value_based(self):
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("", False),
            ("2", False),
        ):
            mega._ANNOUNCED[0] = False
            with mock.patch.dict(
                os.environ, {"DSV41_MHC_MEGA_FUSED": value}, clear=False
            ):
                self.assertEqual(mega.mhc_mega_fused_enabled(), expected, value)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
