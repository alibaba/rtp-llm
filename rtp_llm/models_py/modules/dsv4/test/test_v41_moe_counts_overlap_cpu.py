"""CPU tests for R27/B DSV41_MOE_COUNTS_OVERLAP (flag + CPU fallback).

Hermetic namespace-stub loading as in test_v41_deferred_checks_cpu. The
pinned/event read itself is CUDA-only; on CPU tensors the contiguous
branch must keep the legacy synchronous path (covered indirectly by the
existing v41 CPU suites — here we pin the flag semantics and the
helper's import surface).
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

moe = importlib.import_module("rtp_llm.models_py.modules.dsv4.v41_moe")


class CountsOverlapFlagTest(unittest.TestCase):
    def test_absent_is_on(self):
        # R28: adopted as default (escape = explicit "0").
        env = dict(os.environ)
        env.pop("DSV41_MOE_COUNTS_OVERLAP", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(moe._moe_counts_overlap_enabled())

    def test_value_based(self):
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("", False),
            ("2", False),
        ):
            with mock.patch.dict(
                os.environ, {"DSV41_MOE_COUNTS_OVERLAP": value}, clear=False
            ):
                self.assertEqual(moe._moe_counts_overlap_enabled(), expected, value)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
