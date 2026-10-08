import unittest

from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, configure_shared_indexer_kv_cache


class SharedIndexerKvTest(unittest.TestCase):
    def config(self, pattern: list[str], model_type: str = "hy_v4") -> ModelConfig:
        config = ModelConfig()
        config.model_type = model_type
        config.indexer_types = pattern
        config.num_layers = len(pattern)
        config.is_mtp = model_type.endswith("_mtp")
        config._is_glm52_architecture = model_type == "glm_5"
        config.attn_config.is_sparse = True
        return config

    def test_hy4_enables_compaction_by_default(self) -> None:
        config = self.config(["full", "full", "shared", "shared", "full", "shared"])

        configure_shared_indexer_kv_cache(config, ModelArgs())

        self.assertTrue(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [0, 1, 1, 1, 2, 2])

    def test_explicit_disable_clears_previous_layout(self) -> None:
        config = self.config(["full", "shared"])
        config.enable_glm52_shared_indexer_kv_cache = True
        config.glm52_indexer_kv_slot_mapping = [0, 0]
        args = ModelArgs()
        args.enable_shared_indexer_kv_cache = False

        configure_shared_indexer_kv_cache(config, args)

        self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [])

    def test_hy4_78_layers_use_21_physical_slots(self) -> None:
        full_layers = {0, 1, *range(5, 78, 4)}
        config = self.config(
            ["full" if layer in full_layers else "shared" for layer in range(78)]
        )

        configure_shared_indexer_kv_cache(config, ModelArgs())

        self.assertTrue(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(len(config.glm52_indexer_kv_slot_mapping), 78)
        self.assertEqual(max(config.glm52_indexer_kv_slot_mapping) + 1, 21)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping[:6], [0, 1, 1, 1, 1, 2])

    def test_explicit_enable_supports_hy4_and_glm52(self) -> None:
        for model_type in ("hy_v4", "glm_5"):
            with self.subTest(model_type=model_type):
                config = self.config(["full", "shared"], model_type)
                args = ModelArgs()
                args.enable_shared_indexer_kv_cache = True

                configure_shared_indexer_kv_cache(config, args)

                self.assertTrue(config.enable_glm52_shared_indexer_kv_cache)
                self.assertEqual(config.glm52_indexer_kv_slot_mapping, [0, 0])

    def test_glm52_default_keeps_legacy_layout(self) -> None:
        config = self.config(["full", "shared"], "glm_5")

        configure_shared_indexer_kv_cache(config, ModelArgs())

        self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [])

    def test_glm52_legacy_switch_still_enables_compaction(self) -> None:
        config = self.config(["full", "shared"], "glm_5")
        args = ModelArgs()
        args.enable_glm52_shared_indexer_kv_cache = True

        configure_shared_indexer_kv_cache(config, args)

        self.assertTrue(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [0, 0])

    def test_explicit_disable_overrides_legacy_switch(self) -> None:
        config = self.config(["full", "shared"], "glm_5")
        args = ModelArgs()
        args.enable_glm52_shared_indexer_kv_cache = True
        args.enable_shared_indexer_kv_cache = False

        configure_shared_indexer_kv_cache(config, args)

        self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)

    def test_mtp_default_keeps_independent_storage(self) -> None:
        for model_type in ("hy_v4_mtp", "glm_5_mtp"):
            with self.subTest(model_type=model_type):
                config = self.config(["full"], model_type)

                configure_shared_indexer_kv_cache(config, ModelArgs())

                self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)
                self.assertEqual(config.glm52_indexer_kv_slot_mapping, [])

    def test_all_full_hy4_layers_need_no_compaction(self) -> None:
        config = self.config(["full", "full"])

        configure_shared_indexer_kv_cache(config, ModelArgs())

        self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [])

    def test_shared_first_layer_fails_before_allocation(self) -> None:
        config = self.config(["shared", "full"])

        with self.assertRaisesRegex(ValueError, "layer 0"):
            configure_shared_indexer_kv_cache(config, ModelArgs())

    def test_dense_hy4_is_rejected(self) -> None:
        config = self.config(["full", "shared"])
        config.attn_config.is_sparse = False

        with self.assertRaisesRegex(ValueError, "sparse MLA"):
            configure_shared_indexer_kv_cache(config, ModelArgs())

    def test_explicit_enable_rejects_unsupported_models(self) -> None:
        for model_type in ("qwen_3", "hy_v4_mtp", "glm_5_mtp"):
            with self.subTest(model_type=model_type):
                config = self.config(["full"], model_type)
                args = ModelArgs()
                args.enable_shared_indexer_kv_cache = True

                with self.assertRaisesRegex(ValueError, "target model"):
                    configure_shared_indexer_kv_cache(config, args)


if __name__ == "__main__":
    unittest.main()
