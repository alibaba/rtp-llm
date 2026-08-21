import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

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
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("DSV4_TOPK_V3", None)
            self.assertTrue(
                indexer._run_topk_v3(logits, lengths, output, 512, 2048)
            )

        topk_v3.assert_called_once_with(
            logits, lengths, output, workspace, 512, 2048
        )

    def test_env_can_disable_topk_v3_for_debugging(self) -> None:
        topk_v3 = mock.Mock()
        fake_ops = SimpleNamespace(topk_v3=topk_v3)
        tensor = torch.empty((1, 512), dtype=torch.float32)
        lengths = torch.tensor([512], dtype=torch.int32)
        output = torch.empty((1, 512), dtype=torch.int32)

        with (
            mock.patch.object(indexer, "rtp_llm_ops", fake_ops),
            mock.patch.object(indexer, "_TOPK_V3_OK", True),
            mock.patch.dict(os.environ, {"DSV4_TOPK_V3": "0"}),
        ):
            self.assertFalse(
                indexer._run_topk_v3(tensor, lengths, output, 512, 512)
            )

        topk_v3.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
