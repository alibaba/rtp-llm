"""CPU tests for R30/B DSV41_STAGING_PREWARM (flag matrix).

Hermetic namespace-stub loading; the prewarm itself is CUDA-only (its
engagement is marked by [STAGINGPREWARM] in the rank logs; parity sha
gates the rest).
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

pw = importlib.import_module("rtp_llm.models_py.modules.dsv4._v41_staging_prewarm")


class StagingPrewarmFlagTest(unittest.TestCase):
    def test_absent_is_on(self):
        # R30: adopted as default (escape = explicit "0").
        env = dict(os.environ)
        env.pop("DSV41_STAGING_PREWARM", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(pw.staging_prewarm_enabled())

    def test_value_based(self):
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("", False),
        ):
            with mock.patch.dict(
                os.environ, {"DSV41_STAGING_PREWARM": value}, clear=False
            ):
                self.assertEqual(pw.staging_prewarm_enabled(), expected, value)

    def test_cpu_noop(self):
        pw._DONE.clear()
        with mock.patch.dict(os.environ, {"DSV41_STAGING_PREWARM": "1"}, clear=False):
            pw.prewarm_device(torch.device("cpu"))  # must not raise
        pw._DONE.clear()


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
