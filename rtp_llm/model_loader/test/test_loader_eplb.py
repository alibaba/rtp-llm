import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.model_loader.loader import ModelLoader


class ModelLoaderEplbTest(unittest.TestCase):
    def _make_loader(self, enable_eplb):
        loader = object.__new__(ModelLoader)
        loader.model_config = SimpleNamespace(
            phy2log_path=None,
            num_layers=2,
            expert_num=4,
            ckpt_path="/unused/checkpoint",
            compute_dtype=None,
        )
        loader._weights_info = SimpleNamespace(
            expert_num_=4,
            phy_exp_num_=4,
            ep_size=1,
            num_nodes=1,
            enable_eplb_=enable_eplb,
        )
        return loader

    @patch("rtp_llm.model_loader.loader.LoadConfig.create_redundant_expert")
    @patch("rtp_llm.utils.database.CkptDatabase")
    def test_disabled_eplb_does_not_construct_checkpoint_database(
        self, database_cls, create_redundant_expert
    ):
        create_redundant_expert.return_value = [[0, 1, 2, 3], [0, 1, 2, 3]]
        loader = self._make_loader(False)

        phy2log = loader._create_expert_placement()

        self.assertEqual(phy2log, create_redundant_expert.return_value)
        database_cls.assert_not_called()

    @patch("rtp_llm.model_loader.loader.LoadConfig.create_redundant_expert")
    @patch("rtp_llm.utils.database.CkptDatabase")
    def test_enabled_eplb_uses_executor_weights_without_checkpoint_database(
        self, database_cls, create_redundant_expert
    ):
        create_redundant_expert.return_value = [[0, 1, 2, 3], [0, 1, 2, 3]]
        loader = self._make_loader(True)

        phy2log = loader._create_expert_placement()

        database_cls.assert_not_called()
        self.assertEqual(phy2log, create_redundant_expert.return_value)


if __name__ == "__main__":
    unittest.main()
