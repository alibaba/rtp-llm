import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.maxpander_config import MaxPanderConfig
from rtp_llm.eplb.maxpander_plan import build_circulant_plan
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.utils.model_weight import W


class MaxPanderLoadConfigTest(unittest.TestCase):
    def test_zero_redundancy_still_builds_prefill_mapping_metadata(self):
        for role in ("PREFILL", "DECODE"):
            loader = ModelLoader.__new__(ModelLoader)
            config = MaxPanderConfig(enable=True).resolve(role)
            placement = self.placement(config, 0)
            loader.model_config = SimpleNamespace(maxpander_config=config)
            loader._weights_info = SimpleNamespace(enable_eplb_=False)
            loader._load_config = SimpleNamespace(
                expert_num=256, phy_exp_num=256, num_layers=2, phy2log=placement
            )
            weights = SimpleNamespace(weights=[{}, {}])
            loader._init_eplb_weight(weights, "cpu")
            for layer in weights.weights:
                self.assertEqual(W.log2phy in layer, role == "PREFILL")
                if role == "PREFILL":
                    self.assertEqual(layer[W.logic_expert_cnt].tolist(), [1] * 256)
                    inverse = layer[W.log2phy][:, 0].tolist()
                    self.assertEqual(
                        [placement[0][physical] for physical in inverse],
                        list(range(256)),
                    )

    def placement(self, config, redundant=64):
        return LoadConfig.create_redundant_expert(
            layer_num=2,
            expert_num=256,
            phy_exp_num=256 + redundant,
            ep_size=2,
            num_nodes=1,
            maxpander_config=config,
        )

    def test_prefill_uses_resolved_seed_not_environment(self):
        config = MaxPanderConfig(enable=True, seed=17).resolve("PREFILL")
        with patch.dict(os.environ, {"MAXPANDER_ENABLE": "0", "MAXPANDER_SEED": "99"}):
            for redundant in (0, 64, 128, 256):
                expected = build_circulant_plan(256, 256 + redundant, 2, seed=17)[
                    0
                ].tolist()
                self.assertEqual(
                    self.placement(config, redundant), [expected, expected]
                )

    def test_decode_and_disabled_keep_identity_even_if_environment_enabled(self):
        for config in (
            MaxPanderConfig(enable=True).resolve("DECODE"),
            MaxPanderConfig().resolve("PREFILL"),
        ):
            with patch.dict(os.environ, {"MAXPANDER_ENABLE": "1"}):
                self.assertEqual(self.placement(config, 0), [list(range(256))] * 2)


if __name__ == "__main__":
    unittest.main()
