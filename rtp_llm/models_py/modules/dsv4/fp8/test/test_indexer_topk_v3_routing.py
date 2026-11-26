import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4 import indexer_topk
from rtp_llm.models_py.modules.dsv4.fp8 import indexer


class IndexerTopkV3RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        indexer._topk_v3_workspace_cache.clear()

    def test_flash_and_pro_shared_indexer_routes_to_topk_v3(self) -> None:
        logits = torch.empty((3, 2048), dtype=torch.float32)
        lengths = torch.tensor([2048, 1024, 512], dtype=torch.int32)
        output = torch.empty((3, 512), dtype=torch.int32)
        workspace = torch.empty(indexer._TOPK_V3_WORKSPACE_SIZE, dtype=torch.uint8)
        topk_v3 = mock.Mock()

        with (
            mock.patch.object(indexer, "rtp_llm_ops", SimpleNamespace(topk_v3=topk_v3)),
            mock.patch.object(indexer, "_TOPK_V3_OK", True),
            mock.patch.object(indexer, "_get_topk_workspace", return_value=workspace),
        ):
            self.assertTrue(
                indexer._run_topk_v3(logits, lengths, output, 512, 2048)
            )

        topk_v3.assert_called_once_with(
            logits, lengths, output, workspace, 512, 2048
        )

    def test_unsupported_k_uses_existing_fallback(self) -> None:
        topk_v3 = mock.Mock()
        fake_ops = SimpleNamespace(topk_v3=topk_v3)
        tensor = torch.empty((1, 256), dtype=torch.float32)
        lengths = torch.tensor([256], dtype=torch.int32)
        output = torch.empty((1, 256), dtype=torch.int32)

        with (
            mock.patch.object(indexer, "rtp_llm_ops", fake_ops),
            mock.patch.object(indexer, "_TOPK_V3_OK", True),
        ):
            self.assertFalse(
                indexer._run_topk_v3(tensor, lengths, output, 256, 256)
            )

        topk_v3.assert_not_called()


class IndexerTopkV3RoutingTestSm120(unittest.TestCase):
    """SM120 topk_v3 preservation and fast prefill topk exclusion cases."""

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
                mock.patch.object(indexer_topk, "is_sm120", return_value=sm120),
            ):
                self.assertEqual(
                    indexer._fp8_prefill_fast_topk_enabled(logits), not sm120
                )
        with mock.patch.object(indexer, "_FAST_PREFILL_TOPK_OK", False):
            self.assertFalse(indexer._fp8_prefill_fast_topk_enabled(logits))


if __name__ == "__main__":
    unittest.main()
