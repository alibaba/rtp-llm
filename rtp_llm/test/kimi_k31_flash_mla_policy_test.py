"""Shared upstream L2 policy and K3.1-only initialization, without a GPU."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.kimi_k3.dsa_model_adapter import KimiK31ModelAdapter
from rtp_llm.models_py.utils.flash_mla_policy import configure_flash_mla_l2_persist


class Owner(KimiK31ModelAdapter, torch.nn.Module):
    def __init__(self, dsa):
        super().__init__()
        self.config = SimpleNamespace(
            k3_runtime_config=SimpleNamespace(dsa=dsa), gen_num_per_cycle=3
        )
        self.parallelism_config = SimpleNamespace(tp_size=8)
        self._max_generate_batch_size = 64
        self.layers = []
        self.weight = SimpleNamespace(weights={})


class FlashMlaPolicyTest(unittest.TestCase):
    def setUp(self):
        configure_flash_mla_l2_persist.cache_clear()
        self.addCleanup(configure_flash_mla_l2_persist.cache_clear)

    def test_default_and_explicit_disable(self):
        for setting in (None, "1"):
            configure_flash_mla_l2_persist.cache_clear()
            with patch.dict(os.environ, {"K31_UNRELATED": "preserve"}):
                os.environ.pop("FLASH_MLA_NO_L2_PERSIST", None)
                if setting is not None:
                    os.environ["FLASH_MLA_NO_L2_PERSIST"] = setting
                configure_flash_mla_l2_persist()
                configure_flash_mla_l2_persist()
                self.assertEqual(os.environ["FLASH_MLA_NO_L2_PERSIST"], "1")
                self.assertEqual(os.environ["K31_UNRELATED"], "preserve")

    def test_explicit_enable_is_not_reset_by_second_model(self):
        with patch.dict(os.environ, {"FLASH_MLA_NO_L2_PERSIST": "0"}):
            configure_flash_mla_l2_persist()
            self.assertNotIn("FLASH_MLA_NO_L2_PERSIST", os.environ)
            configure_flash_mla_l2_persist()
            self.assertNotIn("FLASH_MLA_NO_L2_PERSIST", os.environ)

    def test_ordinary_k3_does_not_change_policy(self):
        with patch.dict(os.environ):
            os.environ.pop("FLASH_MLA_NO_L2_PERSIST", None)
            Owner(None)._setup_dsa()
            self.assertNotIn("FLASH_MLA_NO_L2_PERSIST", os.environ)

    def test_dsa_binds_policy_before_workspace_and_keeps_mtp_capacity(self):
        workspace = SimpleNamespace(bind_value_weights=Mock())

        def create(*args):
            self.assertEqual(os.environ["FLASH_MLA_NO_L2_PERSIST"], "1")
            return workspace

        with patch.dict(os.environ):
            os.environ.pop("FLASH_MLA_NO_L2_PERSIST", None)
            with patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
                "dsa_kv_workspace.DsaWorkspaceOwner.from_model", side_effect=create
            ):
                Owner(object())._setup_dsa()
            self.assertEqual(workspace.graph_max_requests, 64)
            self.assertEqual(workspace.graph_max_queries, 4)
            workspace.bind_value_weights.assert_called_once()


if __name__ == "__main__":
    unittest.main()
