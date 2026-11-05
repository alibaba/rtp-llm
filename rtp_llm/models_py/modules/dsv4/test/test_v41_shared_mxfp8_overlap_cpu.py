"""CPU tests for the R10-b MXFP8 shared-expert overlap flag: executor
selection, sequential fallback on CPU/empty inputs. Stream behavior itself
is GPU-side and byte-exactness gated by the parity sha."""

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
    ".models_py.modules.dsv4.moe",
    ".models_py.utils",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

# The real rtp_llm.models_py.utils.arch pulls rtp_llm.device -> rtp_llm.ops
# -> libth_transformer_config (.so); stub the single symbol shared_expert
# needs so the loader stays hermetic on CPU-only shells.
_arch = types.ModuleType("rtp_llm.models_py.utils.arch")
_arch.is_sm120 = lambda *a, **k: False
sys.modules["rtp_llm.models_py.utils.arch"] = _arch

se = importlib.import_module("rtp_llm.models_py.modules.dsv4.moe.shared_expert")


def executor(native_mxfp8):
    return se.get_shared_expert_executor(
        max_tokens_per_rank=1024, dim=64, inter_dim=32, native_mxfp8=native_mxfp8
    )


class SelectionTest(unittest.TestCase):
    def test_mxfp8_default_overlap(self):
        # R11 decision 2: aux-stream overlap is the default (byte-exact).
        env = dict(os.environ)
        env.pop("DSV41_SHARED_MXFP8_OVERLAP", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(executor(True).name, "mxfp8_overlap")

    def test_mxfp8_flag_selects_overlap(self):
        with mock.patch.dict(os.environ, {"DSV41_SHARED_MXFP8_OVERLAP": "1"}):
            self.assertEqual(executor(True).name, "mxfp8_overlap")

    def test_mxfp8_zero_restores_sequential(self):
        with mock.patch.dict(os.environ, {"DSV41_SHARED_MXFP8_OVERLAP": "0"}):
            self.assertEqual(executor(True).name, "mxfp8")

    def test_non_mxfp8_unaffected_by_flag(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV41_SHARED_MXFP8_OVERLAP": "1",
                "DSV4_SHARED_EXPERT_MODE": "sequential",
            },
        ):
            self.assertEqual(executor(False).name, "sequential")


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self.exe = se.MXFP8OverlapSharedExpertExecutor()
        self.calls = []

    def shared(self, x):
        self.calls.append(x)
        return x.float() * 2.0

    def test_cpu_input_runs_sequential_body(self):
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        self.exe.start(self.shared, x)
        out = self.exe.finish()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(torch.equal(out, x.float() * 2.0))

    def test_empty_input_short_circuits(self):
        x = torch.empty(0, 8, dtype=torch.bfloat16)
        self.exe.start(self.shared, x)
        out = self.exe.finish()
        self.assertEqual(self.calls, [])
        self.assertEqual(tuple(out.shape), (0, 8))
        self.assertEqual(out.dtype, torch.float32)

    def test_can_overlap_false_on_cpu(self):
        x = torch.randn(4, 8)
        self.assertFalse(self.exe._can_overlap(x))

    def test_finish_resets_state(self):
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        self.exe.start(self.shared, x)
        self.exe.finish()
        self.assertIsNone(self.exe._out)
        self.assertIsNone(self.exe._active_stream)


if __name__ == "__main__":
    unittest.main(verbosity=2)
