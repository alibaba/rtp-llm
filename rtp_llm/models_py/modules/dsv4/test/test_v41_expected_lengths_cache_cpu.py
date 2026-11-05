"""CPU tests for the G2 per-request chunk-lengths upload cache.

Hermetic namespace-stub loading as in moe_cache_test_support.
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
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

expected = importlib.import_module(
    "rtp_llm.models_py.modules.dsv4._v41_expected_lengths"
)


class ExpectedLengthsCacheTest(unittest.TestCase):
    def cp(self, values=(1024, 1024, 512, 512)):
        return SimpleNamespace(chunk_lengths_per_req=list(values))

    def test_default_off_returns_fresh_tensor(self):
        env = dict(os.environ)
        env.pop("DSV41_EXPECTED_LENGTHS_CACHE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            c = self.cp()
            il = torch.tensor([256] * 4, dtype=torch.int32)
            a = expected.expected_chunk_lengths(c, il)
            b = expected.expected_chunk_lengths(c, il)
        self.assertTrue(torch.equal(a, b))
        self.assertIsNot(a, b)
        self.assertFalse(hasattr(c, "_v41_expected_lengths_cache"))

    def test_cache_hit_and_value_match(self):
        with mock.patch.dict(os.environ, {"DSV41_EXPECTED_LENGTHS_CACHE": "1"}):
            c = self.cp()
            il = torch.tensor([256] * 4, dtype=torch.int32)
            a = expected.expected_chunk_lengths(c, il)
            b = expected.expected_chunk_lengths(c, il)
        self.assertIs(a, b)
        self.assertEqual(a.dtype, il.dtype)
        self.assertEqual(a.tolist(), list(c.chunk_lengths_per_req))

    def test_cache_invalidates_on_value_change(self):
        with mock.patch.dict(os.environ, {"DSV41_EXPECTED_LENGTHS_CACHE": "1"}):
            c = self.cp()
            il = torch.tensor([256] * 4, dtype=torch.int32)
            a = expected.expected_chunk_lengths(c, il)
            c.chunk_lengths_per_req = [1, 2, 3, 4]
            b = expected.expected_chunk_lengths(c, il)
        self.assertIsNot(a, b)
        self.assertEqual(b.tolist(), [1, 2, 3, 4])

    def test_cache_invalidates_on_dtype_change(self):
        with mock.patch.dict(os.environ, {"DSV41_EXPECTED_LENGTHS_CACHE": "1"}):
            c = self.cp()
            a = expected.expected_chunk_lengths(
                c, torch.tensor([256] * 4, dtype=torch.int32)
            )
            b = expected.expected_chunk_lengths(
                c, torch.tensor([256] * 4, dtype=torch.int64)
            )
        self.assertIsNot(a, b)
        self.assertEqual(a.dtype, torch.int32)
        self.assertEqual(b.dtype, torch.int64)

    def test_frozen_context_falls_back(self):
        class Frozen:
            __slots__ = ("chunk_lengths_per_req",)

            def __init__(self):
                self.chunk_lengths_per_req = [8, 8]

        with mock.patch.dict(os.environ, {"DSV41_EXPECTED_LENGTHS_CACHE": "1"}):
            c = Frozen()
            il = torch.tensor([4, 4], dtype=torch.int32)
            a = expected.expected_chunk_lengths(c, il)
            b = expected.expected_chunk_lengths(c, il)
        self.assertEqual(a.tolist(), [8, 8])
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
