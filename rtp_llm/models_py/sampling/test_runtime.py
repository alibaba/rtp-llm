import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.sampling import adapter, runtime


class SamplingRuntimeTest(unittest.TestCase):
    def test_warmup_preserves_global_rng(self) -> None:
        before = torch.cuda.get_rng_state().clone()
        self.assertEqual(runtime.warmup(8193, False), "")
        self.assertEqual(runtime.warmup(8193, True), "")
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))

    def test_dependency_failure_can_select_legacy(self) -> None:
        with patch.object(
            adapter, "sample", side_effect=ImportError("missing dependency")
        ):
            self.assertIn("ImportError", runtime.warmup(8193, False))
        with patch.object(
            adapter,
            "rejection_sample_logits",
            side_effect=ImportError("missing dependency"),
        ):
            self.assertIn("ImportError", runtime.warmup(8193, True))

    def test_execution_failure_is_not_hidden(self) -> None:
        with patch.object(
            adapter, "sample", side_effect=RuntimeError("CUDA execution failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA execution failed"):
                runtime.warmup(8193, False)


if __name__ == "__main__":
    unittest.main()
