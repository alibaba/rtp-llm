"""CPU tests for the R5-a tensor-core router helpers (flag parsing, dispatch,
verify-mode parity counters). Hermetic namespace-stub loading."""

import importlib
import io
import contextlib
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

rtc = importlib.import_module("rtp_llm.models_py.modules.dsv4._v41_router_tc")


class RouterTcModeTest(unittest.TestCase):
    def test_default_off(self):
        env = dict(os.environ)
        env.pop("DSV41_ROUTER_TENSORCORE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(rtc.router_tc_mode(), "0")
            self.assertFalse(rtc.router_verify_enabled())

    def test_modes_and_invalid(self):
        for mode in ("tf32", "bf16", "0"):
            with mock.patch.dict(os.environ, {"DSV41_ROUTER_TENSORCORE": mode}):
                self.assertEqual(rtc.router_tc_mode(), mode)
        with mock.patch.dict(os.environ, {"DSV41_ROUTER_TENSORCORE": "garbage"}):
            self.assertEqual(rtc.router_tc_mode(), "0")
        with mock.patch.dict(os.environ, {"DSV41_ROUTER_TC_VERIFY": "1"}):
            self.assertTrue(rtc.router_verify_enabled())


class RouterLinearTest(unittest.TestCase):
    def test_cpu_modes_match_fp32(self):
        torch.manual_seed(0)
        clean = torch.randn(16, 32)
        weight = torch.randn(8, 32)
        ref = torch.nn.functional.linear(clean, weight)
        for mode in ("0", "tf32", "bf16"):
            got = rtc.router_linear(clean, weight, mode)
            # CPU has no TF32/BF16 tensor-core branch: identical FP32 result
            self.assertTrue(torch.equal(got, ref), mode)

    def test_tf32_flag_untouched_on_cpu(self):
        before = torch.backends.cuda.matmul.allow_tf32
        rtc.router_linear(torch.randn(4, 8), torch.randn(2, 8), "tf32")
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, before)


class VerifyParityTest(unittest.TestCase):
    def setUp(self):
        rtc.VERIFY_STATE.update(
            calls=0, live_rows=0, flip_rows=0, max_logit_absdiff=0.0,
            worst_flip_margin=None,
        )
        self.bias = torch.zeros(8)
        self.valid = torch.ones(3, dtype=torch.bool)

    def _run(self, ref, tc, valid=None):
        err = io.StringIO()
        with contextlib.redirect_stdout(err):
            rtc.verify_router_parity(
                ref, tc, 1.0, self.bias, 2, valid if valid is not None else self.valid
            )
        return err.getvalue()

    def test_identical_no_flips(self):
        ref = torch.tensor([[1.0, 0.5, 0.1, 0, 0, 0, 0, 0]] * 3)
        rtc.VERIFY_STATE["calls"] = 20  # next call hits the %21 print cadence
        out = self._run(ref, ref.clone())
        self.assertEqual(rtc.VERIFY_STATE["flip_rows"], 0)
        self.assertEqual(rtc.VERIFY_STATE["calls"], 21)
        self.assertEqual(rtc.VERIFY_STATE["live_rows"], 3)
        self.assertIn("flip_rows=0", out)

    def test_no_print_off_cadence(self):
        ref = torch.tensor([[1.0, 0.5, 0.1, 0, 0, 0, 0, 0]] * 3)
        out = self._run(ref, ref.clone())
        self.assertEqual(out, "")

    def test_forced_flip_counted_with_margin(self):
        ref = torch.tensor(
            [
                [1.0, 0.6, 0.5, 0, 0, 0, 0, 0],  # boundary e1>e2 -> flips
                [1.0, 0.9, 0.1, 0, 0, 0, 0, 0],
                [1.0, 0.9, 0.1, 0, 0, 0, 0, 0],
            ]
        )
        tc = ref.clone()
        tc[0, 1] = 0.4  # expert1 drops below expert2 on row 0 only
        out = self._run(ref, tc)
        self.assertEqual(rtc.VERIFY_STATE["flip_rows"], 1)
        self.assertIsNotNone(rtc.VERIFY_STATE["worst_flip_margin"])
        self.assertLess(rtc.VERIFY_STATE["worst_flip_margin"], 0.1)
        self.assertIn("last_flips=1", out)

    def test_dead_row_flips_ignored(self):
        ref = torch.tensor(
            [
                [1.0, 0.5, 0.1, 0, 0, 0, 0, 0],
                [1.0, 0.9, 0.1, 0, 0, 0, 0, 0],
                [1.0, 0.9, 0.1, 0, 0, 0, 0, 0],
            ]
        )
        tc = ref.clone()
        tc[0, 2] = 0.95  # flip on row 0
        valid = torch.tensor([False, True, True])
        self._run(ref, tc, valid=valid)
        self.assertEqual(rtc.VERIFY_STATE["flip_rows"], 0)
        self.assertEqual(rtc.VERIFY_STATE["live_rows"], 2)

    def test_max_absdiff_tracked(self):
        ref = torch.zeros(3, 8)
        tc = torch.full((3, 8), 0.25)
        self._run(ref, tc)
        self.assertAlmostEqual(rtc.VERIFY_STATE["max_logit_absdiff"], 0.25, places=6)

    def test_valid_rows_none_counts_all_rows(self):
        ref = torch.tensor(
            [
                [1.0, 0.6, 0.5, 0, 0, 0, 0, 0],
                [1.0, 0.9, 0.1, 0, 0, 0, 0, 0],
            ]
        )
        tc = ref.clone()
        tc[0, 1] = 0.4  # flip row 0
        err = io.StringIO()
        with contextlib.redirect_stdout(err):
            rtc.verify_router_parity(ref, tc, 1.0, self.bias, 2, None)
        self.assertEqual(rtc.VERIFY_STATE["flip_rows"], 1)
        self.assertEqual(rtc.VERIFY_STATE["live_rows"], 2)


if __name__ == "__main__":
    unittest.main()
