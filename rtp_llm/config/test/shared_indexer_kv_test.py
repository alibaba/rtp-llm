import unittest
from types import SimpleNamespace

from rtp_llm.config.model_config import configure_shared_indexer_kv_cache


class SharedIndexerKvTest(unittest.TestCase):
    def config(self, pattern, model_type="hy_v4"):
        return SimpleNamespace(
            model_type=model_type,
            num_layers=len(pattern),
            indexer_types=pattern,
            is_mtp=model_type.endswith("_mtp"),
            attn_config=SimpleNamespace(is_sparse=True),
            _is_glm52_architecture=model_type == "glm_5",
        )

    def args(self, enabled=None, legacy=False):
        return SimpleNamespace(
            enable_shared_indexer_kv_cache=enabled,
            enable_glm52_shared_indexer_kv_cache=legacy,
        )

    def test_hy4_shared_layers_alias_latest_full_slot(self):
        config = self.config(["full", "full", "shared", "shared", "full", "shared"])
        configure_shared_indexer_kv_cache(config, self.args())
        self.assertTrue(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [0, 1, 1, 1, 2, 2])
        configure_shared_indexer_kv_cache(config, self.args(False))
        self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [])

    def test_mtp_and_all_full_layers_keep_independent_storage(self):
        for model_type in ("hy_v4", "hy_v4_mtp", "glm_5_mtp"):
            config = self.config(["full", "full"], model_type)
            configure_shared_indexer_kv_cache(config, self.args())
            self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)

    def test_glm_legacy_and_explicit_override(self):
        config = self.config(["full", "shared"], "glm_5")
        configure_shared_indexer_kv_cache(config, self.args())
        self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)
        configure_shared_indexer_kv_cache(config, self.args(legacy=True))
        self.assertEqual(config.glm52_indexer_kv_slot_mapping, [0, 0])
        configure_shared_indexer_kv_cache(config, self.args(False, legacy=True))
        self.assertFalse(config.enable_glm52_shared_indexer_kv_cache)

    def test_invalid_shared_prefix_fails_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "layer 0"):
            configure_shared_indexer_kv_cache(
                self.config(["shared", "full"]), self.args()
            )


if __name__ == "__main__":
    unittest.main()
