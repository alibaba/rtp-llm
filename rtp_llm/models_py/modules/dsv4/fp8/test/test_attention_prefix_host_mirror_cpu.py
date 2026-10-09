"""CPU tests for the attention.py host-mirror prefix scalars.

``attention.py`` imports deep_gemm at module level, so the two REAL helper
functions are extracted by AST and driven with CPU tensors + a host-sync ban:
with the CP host mirror present and domain-matching, no ``.item()``/``.any()``
device reduction may run; otherwise the legacy path is taken unchanged.
"""

from __future__ import annotations

import ast
import pathlib
import unittest
from runpy import run_path
from unittest.mock import patch

import torch

ATTN_PATH = pathlib.Path(__file__).resolve().parents[1] / "attention.py"
ATTN_SRC = ATTN_PATH.read_text()
NAMES = {"_any_prefix_continuation", "_first_prefix_int"}


def _extract():
    tree = ast.parse(ATTN_SRC)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {n.name for n in nodes} == NAMES, "helper missing from attention.py"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    env = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple,
    }
    exec(compile(mod, str(ATTN_PATH), "exec"), env)
    return env


ENV = _extract()
any_prefix_continuation = ENV["_any_prefix_continuation"]
first_prefix_int = ENV["_first_prefix_int"]


class _BanHostSync:
    def __enter__(self):
        def banned(*a, **k):
            raise AssertionError("host synchronizing call on the hot path")

        self._patchers = [
            patch.object(torch.Tensor, "item", banned),
            patch.object(torch.Tensor, "cpu", banned),
        ]
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._patchers:
            p.stop()
        return False


class AnyPrefixContinuation(unittest.TestCase):
    def test_host_mirror_skips_device_reduction(self):
        device_arg = torch.tensor([0, 0, 5, 0], dtype=torch.int32)  # stands in for CUDA
        with _BanHostSync():
            self.assertTrue(any_prefix_continuation(device_arg, (0, 0, 5, 0)))
            self.assertFalse(any_prefix_continuation(device_arg, (0, 0, 0, 0)))

    def test_values_match_legacy_reduction(self):
        for values in ([0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 9], [3, 2, 1, 7]):
            device_arg = torch.tensor(values, dtype=torch.int32)
            legacy = bool((device_arg > 0).any().item())
            self.assertEqual(any_prefix_continuation(device_arg, tuple(values)), legacy)

    def test_domain_mismatch_falls_back_to_legacy(self):
        # Shorter host mirror would answer False; the domain guard must route
        # to the legacy device reduction (True) instead of trusting it.
        device_arg = torch.tensor([0, 0, 0, 1], dtype=torch.int32)
        self.assertTrue(any_prefix_continuation(device_arg, (0, 0, 0)))
        self.assertTrue(any_prefix_continuation(device_arg, (0, 0)))

    def test_none_mirror_falls_back(self):
        device_arg = torch.tensor([0, 2], dtype=torch.int32)
        self.assertTrue(any_prefix_continuation(device_arg, None))


class FirstPrefixInt(unittest.TestCase):
    def test_host_mirror_skips_item(self):
        device_arg = torch.tensor([64, 64], dtype=torch.int32)
        with _BanHostSync():
            self.assertEqual(first_prefix_int(device_arg, (64, 64)), 64)

    def test_fallback_matches(self):
        device_arg = torch.tensor([12, 34], dtype=torch.int64)
        self.assertEqual(first_prefix_int(device_arg, None), 12)
        self.assertEqual(
            first_prefix_int(device_arg, (12,)), 12
        )  # len mismatch → legacy

    def test_empty_mirror_falls_back(self):
        device_arg = torch.tensor([3], dtype=torch.int32)
        self.assertEqual(first_prefix_int(device_arg, ()), 3)


class ProductionCallSites(unittest.TestCase):
    def test_call_sites_use_helpers(self):
        self.assertIn("any_cont = _any_prefix_continuation(", ATTN_SRC)
        self.assertIn("sp_int=_first_prefix_int(", ATTN_SRC)
        # The old unconditional syncs are gone from those two sites.
        self.assertNotIn("any_cont = bool((prefix_lengths > 0).any().item())", ATTN_SRC)
        self.assertNotIn("sp_int=int(prefix_lengths[0].item())", ATTN_SRC)


def setUpModule():
    run_path(
        str(pathlib.Path(__file__).resolve().parents[2] / "test" / "cpu_test_utils.py")
    )["isolate_cpu_test_module"]()


if __name__ == "__main__":
    unittest.main(verbosity=2)
