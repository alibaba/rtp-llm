from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_v2 import DeepSeekV2


class Glm5KVCacheTest(TestCase):
    def config(self, model_type="glm_5", is_mtp=False):
        config = ModelConfig()
        config.model_type = model_type
        config.num_layers = 6
        config.attn_config.use_mla = True
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = 128
        config.attn_config.tokens_per_block = 64
        config.index_topk_pattern = "FFSSFS"
        config.is_mtp = is_mtp
        return config

    def tags(self, config):
        DeepSeekV2._post_build_model_config(config)
        return [[desc.tag for desc in layer] for layer in config.kv_cache_spec_descs]

    def test_pd_topology_contains_only_written_indexer_keys(self):
        config = self.config()
        expected = [
            ["default", "indexer_kv"],
            ["default", "indexer_kv"],
            ["default"],
            ["default"],
            ["default", "indexer_kv"],
            ["default"],
        ]
        self.assertEqual(self.tags(config), expected)
        self.assertEqual(self.tags(config), expected)  # Post-build is idempotent.

    def test_mtp_owns_indexer_even_with_shared_target_pattern(self):
        self.assertEqual(
            self.tags(self.config(is_mtp=True)),
            [["default", "indexer_kv"]] * 6,
        )

    def test_generic_sparse_mla_topology_is_preserved(self):
        self.assertEqual(
            self.tags(self.config(model_type="deepseek_v3")),
            [["default", "indexer_kv"]] * 6,
        )


if __name__ == "__main__":
    main()
