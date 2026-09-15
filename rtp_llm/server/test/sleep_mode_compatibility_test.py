import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.config.sleep_mode_compatibility import (
    Level2SleepCompatibility,
    SleepQuiesceCompatibility,
    reject_dynamic_lora_mutation,
    reject_dynamic_weight_update,
    reject_embedding_sleep,
    validate_level2_sleep_compatibility,
    validate_sleep_quiesce_compatibility,
)


class SleepQuiesceCompatibilityTest(unittest.TestCase):
    def test_supported_topologies(self):
        defaults = SleepQuiesceCompatibility()
        cases = [
            defaults,
            replace(defaults, world_size=2, tp_size=2),
            replace(defaults, world_size=2, tp_size=2, ep_size=2),
            replace(
                defaults, world_size=2, dp_size=2, ep_size=2, expert_num=64, moe_style=1
            ),
            replace(
                defaults,
                world_size=4,
                tp_size=2,
                dp_size=2,
                ep_size=4,
                expert_num=64,
                moe_style=2,
                moe_layer_index=(0,),
            ),
        ]
        for case in cases:
            with self.subTest(case=case):
                validate_sleep_quiesce_compatibility(
                    enable_sleep_mode=True, compatibility=case
                )

    def test_rejected_topologies_leave_sleep_disabled_behavior_unchanged(self):
        defaults = SleepQuiesceCompatibility()
        dp = replace(defaults, world_size=2, dp_size=2, ep_size=2)
        cases = [
            (replace(defaults, has_system_prompt=True), "resident system prompts"),
            (replace(defaults, ffn_disaggregate=True), "FFN disaggregate"),
            (replace(defaults, world_size=2), "parallel topology"),
            (replace(defaults, tp_size=0), "parallel topology"),
            (replace(defaults, ep_size=3), "parallel topology"),
            (dp, "active MoE layer"),
            (replace(dp, expert_num=64), "active MoE layer"),
            (replace(dp, moe_style=1), "active MoE layer"),
            (replace(dp, expert_num=64, moe_style=1, num_layers=0), "active MoE layer"),
            (replace(dp, expert_num=64, moe_style=1, ep_size=1), "EP=world_size"),
            (replace(dp, expert_num=64, moe_style=2), "active MoE layer"),
            (
                replace(dp, expert_num=64, moe_style=2, moe_layer_index=(-1, 1)),
                "active MoE layer",
            ),
        ]
        for case, diagnostic in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    validate_sleep_quiesce_compatibility(
                        enable_sleep_mode=True, compatibility=case
                    )
                validate_sleep_quiesce_compatibility(
                    enable_sleep_mode=False, compatibility=case
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
            # Now supported: draft weights are reloaded via chained WeightManager.
            {"checkpoint_backed_propose_model": True},
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
            "MoE EPLB; "
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

    def test_embedding_sleep_rejected_only_when_both(self):
        # Allowed: generate deployment with sleep, embedding without sleep.
        reject_embedding_sleep(enable_sleep_mode=True, is_embedding=False)
        reject_embedding_sleep(enable_sleep_mode=False, is_embedding=True)
        reject_embedding_sleep(enable_sleep_mode=False, is_embedding=False)
        # Rejected: embedding deployment with sleep enabled (no lifecycle controller).
        with self.assertRaisesRegex(ValueError, "embedding deployments"):
            reject_embedding_sleep(enable_sleep_mode=True, is_embedding=True)


class BackendValidationOrderingTest(unittest.TestCase):

    @staticmethod
    def make_dense_dp_manager(
        create_engine_config, create_model_config, enabled, level
    ):
        from rtp_llm.ops import TaskType
        from rtp_llm.server.backend_manager import BackendManager

        create_engine_config.return_value = SimpleNamespace(
            runtime_config=SimpleNamespace(
                enable_sleep_mode=enabled, sleep_mode_level=level
            ),
            parallelism_config=SimpleNamespace(
                world_size=2,
                tp_size=1,
                dp_size=2,
                ep_size=2,
                ffn_disaggregate_config=SimpleNamespace(enable_ffn_disaggregate=False),
            ),
            sp_config=SimpleNamespace(checkpoint_path=None),
            kv_cache_config=SimpleNamespace(
                multi_task_prompt_tokens={},
                multi_task_prompt="",
                multi_task_prompt_str="",
            ),
            profiling_debug_logging_config=MagicMock(),
        )
        create_model_config.return_value = SimpleNamespace(
            lora_infos={},
            mm_model_config=SimpleNamespace(is_multimodal=False),
            task_type=TaskType.LANGUAGE_MODEL,
            num_layers=2,
            expert_num=0,
            moe_style=0,
            moe_layer_index=[],
        )
        manager = BackendManager.__new__(BackendManager)
        manager.py_env_configs = MagicMock()
        manager.py_env_configs.eplb_config.enable_eplb.return_value = False
        manager.py_env_configs.eplb_config.redundant_expert = 0
        manager._distributed_server = MagicMock()
        return manager

    @patch("rtp_llm.server.backend_manager.log_gpu_mem")
    @patch("rtp_llm.server.backend_manager.ModelFactory.from_model_configs")
    @patch("rtp_llm.server.backend_manager.init_distributed_environment")
    @patch("rtp_llm.server.backend_manager.ModelFactory.create_model_config")
    @patch("rtp_llm.server.backend_manager.EngineConfig.create")
    def test_quiesce_rejection_precedes_nccl_and_weights_for_both_levels(
        self,
        create_engine_config,
        create_model_config,
        init_nccl,
        create_model,
        _log_mem,
    ):
        for level in (1, 2):
            with self.subTest(level=level):
                manager = self.make_dense_dp_manager(
                    create_engine_config, create_model_config, True, level
                )
                with self.assertRaisesRegex(ValueError, "active MoE layer"):
                    manager.start()
        init_nccl.assert_not_called()
        create_model.assert_not_called()

    @patch("rtp_llm.server.backend_manager.log_gpu_mem")
    @patch("rtp_llm.server.backend_manager.ModelFactory.from_model_configs")
    @patch("rtp_llm.server.backend_manager.init_distributed_environment")
    @patch("rtp_llm.server.backend_manager.ModelFactory.create_model_config")
    @patch("rtp_llm.server.backend_manager.EngineConfig.create")
    def test_prompt_sources_are_rejected_before_tokenization(
        self,
        create_engine_config,
        create_model_config,
        init_nccl,
        create_model,
        _log_mem,
    ):
        for field, value in (
            ("multi_task_prompt", "/unused/prompts.json"),
            ("multi_task_prompt_str", '[{"task_id": "x", "prompt": "hello"}]'),
            ("multi_task_prompt_tokens", {"x": [1, 2]}),
        ):
            with self.subTest(field=field):
                manager = self.make_dense_dp_manager(
                    create_engine_config, create_model_config, True, 2
                )
                config = create_engine_config.return_value
                config.parallelism_config.tp_size = 2
                config.parallelism_config.dp_size = 1
                setattr(config.kv_cache_config, field, value)
                with self.assertRaisesRegex(ValueError, "resident system prompts"):
                    manager.start()
        init_nccl.assert_not_called()
        create_model.assert_not_called()

    @patch("rtp_llm.server.backend_manager.log_gpu_mem")
    @patch("rtp_llm.server.backend_manager.validate_sleep_quiesce_compatibility")
    @patch(
        "rtp_llm.server.backend_manager.init_distributed_environment",
        side_effect=RuntimeError("reached existing NCCL initialization"),
    )
    @patch("rtp_llm.server.backend_manager.ModelFactory.create_model_config")
    @patch("rtp_llm.server.backend_manager.EngineConfig.create")
    def test_sleep_disabled_does_not_inspect_new_capabilities(
        self, create_engine_config, create_model_config, init_nccl, validate, _log_mem
    ):
        manager = self.make_dense_dp_manager(
            create_engine_config, create_model_config, False, 2
        )
        # Absence of every new model capability proves the disabled path neither
        # constructs the capability record nor changes the existing startup path.
        for name in ("num_layers", "expert_num", "moe_style", "moe_layer_index"):
            delattr(create_model_config.return_value, name)
        with self.assertRaisesRegex(
            RuntimeError, "reached existing NCCL initialization"
        ):
            manager.start()
        validate.assert_not_called()
        init_nccl.assert_called_once()

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
