"""Qualify upstream candidate publication and SM100 fallback on real SM120."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_candidate_topk as native
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as candidates
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_candidate_topk,
    test_v41_prefill_candidates,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class SM120CandidateFallback(unittest.TestCase):
    def test_native_opt_in_keeps_torch_fallback_without_compile_or_launch(self):
        logits, visible = test_v41_prefill_candidates.make_case(17, 32769)
        output = torch.full((17, 2048), 123, dtype=torch.int32, device="cuda")
        expected = test_v41_prefill_candidates.reference_select(
            logits, visible, 8, 2048
        )
        with patch.dict(os.environ, {"DSV41_CANDIDATE_NATIVE_TOPK": "1"}), patch.object(
            native, "_load", side_effect=AssertionError("SM100 JIT on SM120")
        ) as build, patch.object(
            native, "_execute", side_effect=AssertionError("SM100 kernel on SM120")
        ) as launch:
            self.assertFalse(native.warmup(logits.device))
            self.assertTrue(native.is_supported(logits[:, :4099], output))
            self.assertIsNone(native.try_select(logits[:, :4099], output))
            self.assertTrue(torch.all(output == 123).item())
            actual, flags = candidates.select_candidates(
                logits, visible, 8, 2048, out=output, build_bitmap=False
            )
            self.assertIs(actual, output)
            self.assertIsNone(flags)
            torch.testing.assert_close(
                actual.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
            )
            build.assert_not_called()
            launch.assert_not_called()


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Candidate qualification requires a real SM120 GPU")
    for name in ("topk_v3", "dsv41_topk_v3_finite"):
        if not hasattr(rtp_llm_ops, name):
            raise RuntimeError(f"Candidate qualification requires native {name}")
    tests.addTests(loader.loadTestsFromModule(test_v41_prefill_candidates))
    # SM100 native execution has a separate donor suite and is not an SM120 path.
    tests.addTests(
        loader.loadTestsFromTestCase(test_v41_candidate_topk.HostContractTest)
    )
    return tests


if __name__ == "__main__":
    unittest.main()
