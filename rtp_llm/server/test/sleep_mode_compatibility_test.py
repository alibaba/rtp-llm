import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.config.sleep_mode_compatibility import (
    Level2SleepCompatibility,
    reject_dynamic_lora_mutation,
    reject_dynamic_weight_update,
    validate_level2_sleep_compatibility,
)


class Level2SleepCompatibilityTest(unittest.TestCase):
    def validate(self, **kwargs) -> None:
        validate_level2_sleep_compatibility(
            enable_sleep_mode=True,
            sleep_mode_level=2,
            compatibility=Level2SleepCompatibility(**kwargs),
        )

    def test_allowed_matrix(self):
        allowed = [
            {},
            {"lora_adapter_count": 1, "merge_lora": True},
            {"local_multimodal_vit": False},
            {"checkpoint_backed_propose_model": False},
            {"eplb_enabled": False, "redundant_expert": 0},
        ]
        for case in allowed:
            with self.subTest(case=case):
                self.validate(**case)

        unsafe = Level2SleepCompatibility(
            lora_adapter_count=2,
            local_multimodal_vit=True,
            checkpoint_backed_propose_model=True,
            eplb_enabled=True,
            redundant_expert=1,
        )
        validate_level2_sleep_compatibility(
            enable_sleep_mode=False,
            sleep_mode_level=2,
            compatibility=unsafe,
        )
        validate_level2_sleep_compatibility(
            enable_sleep_mode=True,
            sleep_mode_level=1,
            compatibility=unsafe,
        )

    def test_rejected_matrix(self):
        rejected = [
            {"lora_adapter_count": 1, "merge_lora": False},
            {"lora_adapter_count": 2, "merge_lora": True},
            {"local_multimodal_vit": True},
            {"checkpoint_backed_propose_model": True},
            {"eplb_enabled": True},
            {"redundant_expert": 1},
        ]
        for case in rejected:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.validate(**case)

    def test_diagnostics_aggregate_in_deterministic_order(self):
        with self.assertRaisesRegex(
            ValueError,
            "unmerged or multiple LoRA adapters .*; local multimodal ViT; "
            "checkpoint-backed propose/draft model; MoE EPLB; "
            r"redundant experts .*\. Use sleep mode level 1 instead\.",
        ):
            self.validate(
                lora_adapter_count=2,
                merge_lora=False,
                local_multimodal_vit=True,
                checkpoint_backed_propose_model=True,
                eplb_enabled=True,
                redundant_expert=3,
            )

    def test_dynamic_lora_gate_only_blocks_level_two(self):
        reject_dynamic_lora_mutation(enable_sleep_mode=False, sleep_mode_level=2)
        reject_dynamic_lora_mutation(enable_sleep_mode=True, sleep_mode_level=1)
        with self.assertRaisesRegex(ValueError, "runtime LoRA add/update/load"):
            reject_dynamic_lora_mutation(enable_sleep_mode=True, sleep_mode_level=2)

    def test_dynamic_weight_update_gate_only_blocks_level_two(self):
        reject_dynamic_weight_update(enable_sleep_mode=False, sleep_mode_level=2)
        reject_dynamic_weight_update(enable_sleep_mode=True, sleep_mode_level=1)
        with self.assertRaisesRegex(ValueError, "runtime weight update"):
            reject_dynamic_weight_update(enable_sleep_mode=True, sleep_mode_level=2)


class BackendValidationOrderingTest(unittest.TestCase):
    @patch("rtp_llm.server.backend_manager.ModelFactory.from_model_configs")
    @patch(
        "rtp_llm.server.backend_manager.validate_level2_sleep_compatibility",
        side_effect=ValueError("incompatible"),
    )
    @patch("rtp_llm.server.backend_manager.ModelFactory.create_model_config")
    @patch("rtp_llm.server.backend_manager.EngineConfig.create")
    def test_rejection_precedes_model_creation(
        self,
        create_engine_config,
        create_model_config,
        validate,
        create_model,
    ):
        from rtp_llm.server.backend_manager import BackendManager

        py_env_configs = MagicMock()
        py_env_configs.lora_config.merge_lora = False
        py_env_configs.vit_config.vit_separation = 0
        py_env_configs.eplb_config.enable_eplb.return_value = False
        py_env_configs.eplb_config.redundant_expert = 0

        runtime_config = SimpleNamespace(enable_sleep_mode=True, sleep_mode_level=2)
        create_engine_config.return_value = SimpleNamespace(
            runtime_config=runtime_config,
            sp_config=SimpleNamespace(checkpoint_path=None),
            kv_cache_config=MagicMock(),
            profiling_debug_logging_config=MagicMock(),
        )
        create_model_config.return_value = SimpleNamespace(
            lora_infos={},
            mm_model_config=SimpleNamespace(is_multimodal=False),
        )

        manager = BackendManager.__new__(BackendManager)
        manager.py_env_configs = py_env_configs
        manager._distributed_server = MagicMock()

        with self.assertRaisesRegex(ValueError, "incompatible"):
            manager.start()

        validate.assert_called_once()
        create_model.assert_not_called()


class RuntimeLoraMutationGateTest(unittest.TestCase):
    @patch("rtp_llm.lora.lora_manager.sleep_mode_level", return_value=2)
    @patch("rtp_llm.lora.lora_manager.is_enabled", return_value=True)
    def test_update_and_add_entries_reject_before_loading(
        self, _is_enabled, _sleep_mode_level
    ):
        from rtp_llm.lora.lora_manager import LoraManager

        manager = LoraManager.__new__(LoraManager)
        manager.lora_infos_ = {}
        manager.max_lora_model_size_ = -1
        manager.weights_loader_ = MagicMock()

        with self.assertRaisesRegex(ValueError, "runtime LoRA add/update/load"):
            manager.get_add_lora_map({"adapter": "/adapter"})
        with self.assertRaisesRegex(ValueError, "runtime LoRA add/update/load"):
            manager.add_lora("adapter", "/adapter")
        manager.weights_loader_.load_lora_weights.assert_not_called()

    def test_remove_remains_allowed(self):
        from rtp_llm.lora.lora_manager import LoraManager

        manager = LoraManager.__new__(LoraManager)
        manager.lora_infos_ = {"adapter": "/adapter"}
        manager.lora_cpp_wrapper_ = MagicMock()

        manager.remove_lora("adapter")

        self.assertEqual(manager.lora_infos_, {})
        manager.lora_cpp_wrapper_.remove_lora.assert_called_once_with("adapter")

    @patch("rtp_llm.model_loader.loader.sleep_mode_level", return_value=2)
    @patch("rtp_llm.model_loader.loader.is_enabled", return_value=True)
    def test_direct_loader_entry_rejects_before_database_mutation(
        self, _is_enabled, _sleep_mode_level
    ):
        from rtp_llm.model_loader.loader import ModelLoader

        loader = ModelLoader.__new__(ModelLoader)
        loader._load_config = MagicMock()

        with self.assertRaisesRegex(ValueError, "runtime LoRA add/update/load"):
            loader.load_lora_weights("adapter", "/adapter")
        loader._load_config.database.load_lora.assert_not_called()


if __name__ == "__main__":
    unittest.main()
