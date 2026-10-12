"""Run upstream sparse-selection correctness suites on required SM120 hardware."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import indexer
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_dsv4_top_k_per_row_prefill,
    test_topk_v3,
    test_v41_prefill_index_plan,
    test_v41_prefill_topk,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class SM120PrefillRouting(unittest.TestCase):
    def test_indexer_uses_row_prefill_with_fast_topk_requested(self):
        logits, starts, ends = test_dsv4_top_k_per_row_prefill._make(
            4, 4099, [0, 513, 2048, 4099], seed=41
        )
        logits[1] = -torch.inf
        for k in (512, 1024):
            output = torch.empty((4, k), dtype=torch.int32, device="cuda")
            with patch.dict(
                os.environ,
                {
                    "DSV4_PREFILL_FAST_TOPK": "1",
                    "DSV4_INDEXER_TOPK_BACKEND": "auto",
                    "DSV4_PREFILL_TOPK_FORCE_RADIX": "1",
                },
            ), patch.object(
                rtp_llm_ops,
                "fast_topk_v2_variable",
                side_effect=AssertionError("SM120 must not launch legacy fast TopK"),
            ) as legacy:
                indexer._run_prefill_topk(logits, starts, ends, output, k, 1)
                legacy.assert_not_called()
            test_dsv4_top_k_per_row_prefill._assert_equiv(
                output, logits, starts, ends, k, tag=f"SM120 indexer k={k}"
            )


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            "Sparse-selection SM120 qualification requires a real SM120 GPU"
        )
    for name in ("topk_v3", "dsv41_topk_v3_finite", "dsv4_top_k_per_row_prefill"):
        if not hasattr(rtp_llm_ops, name):
            raise RuntimeError(f"Sparse-selection qualification requires native {name}")
    if test_v41_prefill_index_plan.fused.triton is None:
        raise RuntimeError("Sparse-selection qualification requires Triton")
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(function)
        for module in (test_topk_v3, test_dsv4_top_k_per_row_prefill)
        for name, function in vars(module).items()
        if name.startswith("test_") and callable(function)
    )
    suite.addTests(loader.loadTestsFromModule(test_v41_prefill_index_plan))
    suite.addTests(loader.loadTestsFromModule(test_v41_prefill_topk))
    suite.addTests(tests)
    return suite


if __name__ == "__main__":
    unittest.main()
