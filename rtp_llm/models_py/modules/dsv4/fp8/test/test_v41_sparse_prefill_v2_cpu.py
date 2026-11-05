"""CPU tests for the R6 sparse-prefill V2 flag plumbing and validation."""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[6]
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

v2 = importlib.import_module(
    "rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_v2"
)


class SparsePrefillV2ModeTest(unittest.TestCase):
    def test_default_h32(self):
        # R44 adoption: absent env selects h32 (R41: 3/3 pairs, 64/64 tokens,
        # op rel_l2 ~1.2e-5 vs legacy); qt64/s2/0 remain the escape hatches.
        env = dict(os.environ)
        env.pop("DSV41_SPARSE_PREFILL_V2", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(v2.sparse_prefill_v2_mode(), "h32")

    def test_escape_hatch_qt64(self):
        # R24 default, kept as an escape hatch.
        with mock.patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_V2": "qt64"}):
            self.assertEqual(v2.sparse_prefill_v2_mode(), "qt64")

    def test_escape_hatch_s2(self):
        # R21 byte-exact pipelined launch, kept as an escape hatch.
        with mock.patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_V2": "s2"}):
            self.assertEqual(v2.sparse_prefill_v2_mode(), "s2")

    def test_escape_hatch_zero(self):
        with mock.patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_V2": "0"}):
            self.assertEqual(v2.sparse_prefill_v2_mode(), "0")

    def test_known_modes(self):
        for mode in ("s2", "qt", "qt64", "h32"):
            with mock.patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_V2": mode}):
                self.assertEqual(v2.sparse_prefill_v2_mode(), mode)

    def test_invalid_mode_falls_back(self):
        with mock.patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_V2": "turbo"}):
            self.assertEqual(v2.sparse_prefill_v2_mode(), "0")

    def test_cpu_input_rejected_by_validation(self):
        q = torch.zeros(2, 4, 512, dtype=torch.bfloat16)
        kv = torch.zeros(8, 1, 512, dtype=torch.bfloat16)
        idx = torch.zeros(2, 1, 32, dtype=torch.int32)
        lens = torch.full((2,), 32, dtype=torch.int32)
        sinks = torch.zeros(4)
        with self.assertRaises(ValueError):
            v2.sparse_prefill_attention_v2(q, kv, idx, lens, sinks, "qt")


if __name__ == "__main__":
    unittest.main()
