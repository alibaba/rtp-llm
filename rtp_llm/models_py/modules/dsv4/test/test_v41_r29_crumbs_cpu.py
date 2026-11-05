"""CPU tests for the R29/B crumb flags.

Hermetic namespace-stub loading. The kernels are GPU-only; the flag
matrices are pure python (import-cached modules loaded fresh per case).
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
    ".models_py.modules.dsv4.fp8",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)


def _load(name, rel, env):
    with mock.patch.dict(os.environ, env, clear=True):
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class InvRopeFoldFlagTest(unittest.TestCase):
    REL = "rtp_llm/models_py/modules/dsv4/fp8/_v41_prefill_attention_v2.py"

    def test_default_off(self):
        m = _load("spv2_t1", self.REL, {})
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(m.inv_rope_fold_enabled())

    def test_value_based(self):
        m = _load("spv2_t2", self.REL, {})
        for value, expected in (("1", True), ("0", False), ("true", False)):
            # The flag is read at call time (announce-once); patch the env
            # around the CALL, not the load.
            m._INV_ROPE_FOLD_ANNOUNCED[0] = False
            with mock.patch.dict(
                os.environ, {"DSV41_INV_ROPE_FOLD": value}, clear=False
            ):
                self.assertEqual(m.inv_rope_fold_enabled(), expected, value)


class AttnNormQuantFuseFlagTest(unittest.TestCase):
    REL = "rtp_llm/models_py/modules/dsv4/fp8/_v41_norm_quant.py"

    def test_default_off(self):
        m = _load("nq_t1", self.REL, {})
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(m.attn_norm_quant_fuse_enabled())

    def test_value_based(self):
        m = _load("nq_t2", self.REL, {})
        for value, expected in (("1", True), ("0", False), ("true", False)):
            m._ATTN_NORM_QUANT_FUSE_ANNOUNCED[0] = False
            with mock.patch.dict(
                os.environ, {"DSV41_ATTN_NORM_QUANT_FUSE": value}, clear=False
            ):
                self.assertEqual(m.attn_norm_quant_fuse_enabled(), expected, value)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
