"""SM120 cases split out of test_indexer_topk_v3_routing.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import indexer


class IndexerTopkV3RoutingTestSm120(unittest.TestCase):
    """Sibling of ``IndexerTopkV3RoutingTest`` carrying the SM120 cases."""

    def test_topk_v3_preserves_kernel_selection_and_trailing_padding(self) -> None:
        logits = torch.empty((1, 1024), dtype=torch.float32)
        lengths = torch.tensor([3], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)
        selected = torch.full_like(output, -1)
        selected[0, :4] = torch.tensor([2, -1, 0, 1], dtype=torch.int32)
        workspace = torch.empty(indexer._TOPK_V3_WORKSPACE_SIZE, dtype=torch.uint8)

        with (
            mock.patch.object(
                indexer,
                "rtp_llm_ops",
                SimpleNamespace(topk_v3=lambda *args: output.copy_(selected)),
            ),
            mock.patch.object(indexer, "_TOPK_V3_OK", True),
            mock.patch.object(indexer, "_get_topk_workspace", return_value=workspace),
        ):
            self.assertTrue(indexer._run_topk_v3(logits, lengths, output, 512, 1024))
            torch.testing.assert_close(output, selected, rtol=0, atol=0)

    def test_fast_prefill_topk_excludes_sm120(self) -> None:
        logits = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
        for sm120 in (False, True):
            with (
                self.subTest(sm120=sm120),
                mock.patch.object(indexer, "_FAST_PREFILL_TOPK_OK", True),
                mock.patch.object(indexer, "is_sm120", return_value=sm120),
            ):
                self.assertEqual(
                    indexer._fp8_prefill_fast_topk_enabled(logits), not sm120
                )
        with mock.patch.object(indexer, "_FAST_PREFILL_TOPK_OK", False):
            self.assertFalse(indexer._fp8_prefill_fast_topk_enabled(logits))


if __name__ == "__main__":
    import unittest

    unittest.main()
