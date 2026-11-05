"""CPU tests for R28/C DSV41_CHUNK_BOUNDARY_OVERLAP (_v41_chunk_boundary.py).

Hermetic namespace-stub loading; the module is pure python (os/sys only).
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
for suffix in ["", ".models_py", ".models_py.modules", ".models_py.modules.dsv4"]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

cb = importlib.import_module("rtp_llm.models_py.modules.dsv4._v41_chunk_boundary")


class ChunkBoundaryFlagTest(unittest.TestCase):
    def test_absent_is_off(self):
        env = dict(os.environ)
        env.pop("DSV41_CHUNK_BOUNDARY_OVERLAP", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(cb.chunk_boundary_overlap_enabled())

    def test_value_based(self):
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("", False),
            ("2", False),
        ):
            with mock.patch.dict(
                os.environ, {"DSV41_CHUNK_BOUNDARY_OVERLAP": value}, clear=False
            ):
                self.assertEqual(cb.chunk_boundary_overlap_enabled(), expected, value)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
