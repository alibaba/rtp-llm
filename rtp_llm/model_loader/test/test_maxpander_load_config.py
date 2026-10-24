import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.config.maxpander_config import MaxPanderConfig
from rtp_llm.eplb.maxpander_plan import build_circulant_plan
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.utils.model_weight import W


class MaxPanderLoadConfigTest(unittest.TestCase):
    def test_ft_style_rejected_before_loading_or_placement(self):
        for expert_num, redundant in ((256, 0), (256, 64), (384, 0), (384, 96)):
            for seed in (0, 17):
                with self.subTest(
                    expert_num=expert_num, redundant=redundant, seed=seed
                ):
                    model_config = SimpleNamespace(
                        maxpander_config=MaxPanderConfig(
                            enable=True, seed=seed
                        ).resolve("PREFILL"),
                        expert_num=expert_num,
                        eplb_config=SimpleNamespace(redundant_expert=redundant),
                    )
                    weights_info = Mock()
                    with patch.object(ModelLoader, "create_eplb") as placement:
                        with self.assertRaisesRegex(ValueError, "ft-style"):
                            ModelLoader(
                                model_config,
                                weights_info,
                                None,
                                SimpleNamespace(is_ft_style=True),
                            )
                    placement.assert_not_called()
                    weights_info.create_model_weight_info.assert_not_called()

    def test_original_and_inactive_ft_style_loading_are_unchanged(self):
        for role, enabled, ft_style in (
            ("PREFILL", True, False),
            ("PREFILL", False, True),
            ("DECODE", True, True),
            ("FRONTEND", True, True),
        ):
            with self.subTest(role=role, enabled=enabled, ft_style=ft_style):
                model_config = SimpleNamespace(
                    maxpander_config=MaxPanderConfig(enable=enabled).resolve(role),
                    task_type=None,
                    compute_dtype="fp16",
                )
                weights_info = Mock()
                database = SimpleNamespace(is_ft_style=ft_style)
                with patch.object(ModelLoader, "create_eplb", return_value=(None, [])):
                    with patch("rtp_llm.device.get_current_device"):
                        ModelLoader(model_config, weights_info, None, database)
                weights_info.create_model_weight_info.assert_called_once_with(database)
                weights_info.create_load_config.assert_called_once()

    def test_ft_style_export_rejected_before_materializing_or_writing_weights(self):
        loader = ModelLoader.__new__(ModelLoader)
        loader.model_config = SimpleNamespace(
            maxpander_config=MaxPanderConfig(enable=True).resolve("PREFILL")
        )
        with patch.object(loader, "_create_model_weights") as create_weights:
            with self.assertRaisesRegex(ValueError, "ft-style"):
                loader.dump_weight_as_ft_style("cpu", "/must-not-write")
        create_weights.assert_not_called()

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
