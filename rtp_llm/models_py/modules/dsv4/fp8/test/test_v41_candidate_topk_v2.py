"""R51: unit test for the SM120 fused candidate TopK (_v41_candidate_topk_v2).
CPU part: flag/shape gating via mocks. GPU part (only with DSV41_CANDIDATE_TOPK_V2=1
and a CUDA device): warmup compiles sm_120 and _semantic_matches must hold.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


def load():
    path = Path(__file__).resolve().parents[1] / "_v41_candidate_topk_v2.py"
    spec = importlib.util.spec_from_file_location("candidate_topk_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


op = load()


class V2GateTest(unittest.TestCase):
    def test_flag_default_off(self):
        env = dict(os.environ)
        env.pop("DSV41_CANDIDATE_TOPK_V2", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(op._enabled())
        with mock.patch.dict(os.environ, {"DSV41_CANDIDATE_TOPK_V2": "1"}):
            self.assertTrue(op._enabled())

    def test_shape_gate(self):
        self.assertTrue(op._shape_supported(256, 4096, 2048))
        self.assertFalse(op._shape_supported(256, 4096, 1024))  # wrong k
        self.assertFalse(op._shape_supported(256, 2048, 2048))  # width too small
        self.assertFalse(op._shape_supported(9000, 4096, 2048))  # too many rows


class V2GpuTest(unittest.TestCase):
    def test_sm120_warmup_and_semantics(self):
        if os.environ.get("DSV41_CANDIDATE_TOPK_V2", "0") != "1":
            self.skipTest("v2 flag off")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            self.skipTest("needs SM120")
        ok = op.warmup(torch.device("cuda", 0))
        self.assertTrue(ok, "v2 warmup failed on SM120")
        # real-shape select vs torch.topk
        torch.manual_seed(0)
        rows, width = 256, 8192
        scores = torch.randn(rows, width, dtype=torch.float32, device="cuda")
        out = torch.empty((rows, 2048), dtype=torch.int32, device="cuda")
        got = op.try_select(scores, out, k=2048)
        self.assertIsNotNone(got, "try_select returned None (fell back)")
        ref = torch.topk(scores, 2048, dim=-1, sorted=False).indices
        ref_set = torch.where(
            torch.topk(scores, 2048, dim=-1, sorted=False).values > -torch.inf, ref, -1
        ).int()
        self.assertTrue(
            torch.equal(got.sort(dim=-1).values, ref_set.sort(dim=-1).values),
            "v2 top-k set differs from torch.topk",
        )


if __name__ == "__main__":
    unittest.main()
