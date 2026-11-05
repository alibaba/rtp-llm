"""CPU equivalence tests for the G1 masked PP frame finiteness validation.

The masked reduction must reproduce ``torch.isfinite(values[active]).all()``
exactly (including the empty-selection edge), without boolean-mask indexing.
Loads v41_pp hermetically (namespace stubs) as in moe_cache_test_support.
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

v41_pp = importlib.import_module("rtp_llm.models_py.modules.dsv4.v41_pp")


class MaskedAllFiniteTest(unittest.TestCase):
    def _reference(self, values, active):
        return torch.isfinite(values[active]).all()

    def test_equivalence_fuzz(self):
        torch.manual_seed(7)
        for trial in range(200):
            rows = int(torch.randint(1, 10, (1,)))
            for shape in ((rows, 4, 8), (rows, 4)):
                values = torch.randn(*shape)
                flat = values.flatten()
                k = int(torch.randint(0, 4, (1,)))
                for idx in torch.randperm(flat.numel())[:k]:
                    flat[idx] = float("nan") if trial % 2 else float("inf")
                active = torch.rand(rows) < 0.6
                got = bool(v41_pp._masked_all_finite(values, active))
                want = bool(self._reference(values, active))
                self.assertEqual(got, want, f"trial={trial} shape={shape}")

    def test_all_inactive_is_true(self):
        values = torch.full((3, 4), float("nan"))
        inactive = torch.zeros(3, dtype=torch.bool)
        self.assertTrue(bool(v41_pp._masked_all_finite(values, inactive)))
        self.assertEqual(
            bool(v41_pp._masked_all_finite(values, inactive)),
            bool(self._reference(values, inactive)),
        )

    def test_single_bad_active_row_is_false(self):
        values = torch.ones((4, 2, 3))
        values[2, 1, 0] = float("inf")
        active = torch.tensor([True, False, True, False])
        self.assertFalse(bool(v41_pp._masked_all_finite(values, active)))
        # same non-finite in an INACTIVE row stays valid
        active2 = torch.tensor([True, False, False, False])
        self.assertTrue(bool(v41_pp._masked_all_finite(values, active2)))

    def test_flag_wiring(self):
        env = dict(os.environ)
        env.pop("DSV41_PP_MASKED_VALIDATE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(v41_pp._pp_masked_validate_enabled())
        with mock.patch.dict(os.environ, {"DSV41_PP_MASKED_VALIDATE": "1"}):
            self.assertTrue(v41_pp._pp_masked_validate_enabled())


if __name__ == "__main__":
    unittest.main()
