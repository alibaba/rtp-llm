"""CPU tests for the Batch-3 A helpers: flag parsing, device row_map
equivalence vs the reference Python loop, wire/lookup caches, deferred
check CPU semantics."""

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

lean = importlib.import_module("rtp_llm.models_py.modules.dsv4._v41_pp_lean")


def reference_row_map(request_ids, positions, valid_rows, indices):
    return torch.tensor(
        [
            [indices[r], p, 1] if v else [-1, -1, 0]
            for r, p, v in zip(
                request_ids.tolist(), positions.tolist(), valid_rows.tolist()
            )
        ],
        dtype=torch.int64,
    ).reshape(-1, 3)


class FlagTest(unittest.TestCase):
    def _env_without(self, *keys):
        env = dict(os.environ)
        for k in keys:
            env.pop(k, None)
        return env

    def test_defaults_on(self):
        # Batch-4a decision 1: both PP-lean paths default ON (byte-exact).
        with mock.patch.dict(
            os.environ,
            self._env_without("DSV41_PP_ASYNC_PREFLIGHT", "DSV41_PP_DEVICE_ROW_MAP"),
            clear=True,
        ):
            self.assertTrue(lean.pp_async_preflight_enabled())
            self.assertTrue(lean.pp_device_rowmap_enabled())

    def test_one_enables(self):
        with mock.patch.dict(
            os.environ,
            {"DSV41_PP_ASYNC_PREFLIGHT": "1", "DSV41_PP_DEVICE_ROW_MAP": "1"},
        ):
            self.assertTrue(lean.pp_async_preflight_enabled())
            self.assertTrue(lean.pp_device_rowmap_enabled())

    def test_escape_hatch_and_other_values_disable(self):
        for value in ("0", "true", ""):
            with mock.patch.dict(
                os.environ,
                {"DSV41_PP_ASYNC_PREFLIGHT": value, "DSV41_PP_DEVICE_ROW_MAP": value},
            ):
                self.assertFalse(lean.pp_async_preflight_enabled())
                self.assertFalse(lean.pp_device_rowmap_enabled())


class RowMapTest(unittest.TestCase):
    def _check(self, request_ids, positions, valid_rows, indices):
        got = lean.build_row_map(request_ids, positions, valid_rows, indices)
        want = reference_row_map(request_ids, positions, valid_rows, indices)
        self.assertEqual(got.dtype, torch.int64)
        self.assertTrue(torch.equal(got, want), f"{got} != {want}")

    def test_all_valid_single_request(self):
        n = 16
        self._check(
            torch.zeros(n, dtype=torch.int64),
            torch.arange(n, dtype=torch.int64),
            torch.ones(n, dtype=torch.bool),
            {0: 0},
        )

    def test_padded_rows(self):
        n = 12
        valid = torch.ones(n, dtype=torch.bool)
        valid[7:] = False
        self._check(
            torch.zeros(n, dtype=torch.int64),
            torch.arange(n, dtype=torch.int64),
            valid,
            {0: 0},
        )

    def test_multi_request(self):
        request_ids = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int64)
        positions = torch.tensor([3, 4, 5, 6, 7], dtype=torch.int64)
        valid = torch.tensor([True, True, False, True, True])
        self._check(request_ids, positions, valid, {0: 2, 1: 0, 2: 1})

    def test_invalid_rows_may_carry_unknown_ids(self):
        # the reference never indexes invalid rows; the builder clamps and
        # discards them via where
        request_ids = torch.tensor([0, 999], dtype=torch.int64)
        positions = torch.tensor([1, 2], dtype=torch.int64)
        valid = torch.tensor([True, False])
        self._check(request_ids, positions, valid, {0: 0})

    def test_valid_unknown_id_raises_on_cpu(self):
        with self.assertRaises(ValueError):
            lean.build_row_map(
                torch.tensor([7], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([True]),
                {0: 0},
            )

    def test_inputs_not_mutated(self):
        request_ids = torch.tensor([0, 1], dtype=torch.int64)
        positions = torch.tensor([5, 6], dtype=torch.int64)
        valid = torch.tensor([True, False])
        lean.build_row_map(
            request_ids.clone(), positions.clone(), valid.clone(), {0: 0, 1: 1}
        )
        self.assertTrue(torch.equal(request_ids, torch.tensor([0, 1])))


class CacheTest(unittest.TestCase):
    def test_wire_cache_identity(self):
        a = lean.expected_wire_tensor("t_a", (1, 2, 3), torch.device("cpu"))
        b = lean.expected_wire_tensor("t_a", (1, 2, 3), torch.device("cpu"))
        c = lean.expected_wire_tensor("t_a", (1, 2, 4), torch.device("cpu"))
        self.assertIs(a, b)
        self.assertIsNot(a, c)
        self.assertEqual(a.dtype, torch.int32)

    def test_row_lookup_cache(self):
        l1 = lean._row_lookup({0: 1}, torch.device("cpu"))
        l2 = lean._row_lookup({0: 1}, torch.device("cpu"))
        l3 = lean._row_lookup({0: 2}, torch.device("cpu"))
        self.assertIs(l1, l2)
        self.assertIsNot(l1, l3)


class DeferredCheckCPUTest(unittest.TestCase):
    def test_cpu_true_passes(self):
        lean.deferred_check(torch.tensor(True), "msg")

    def test_cpu_false_raises(self):
        with self.assertRaises(ValueError):
            lean.deferred_check(torch.tensor(False), "msg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
