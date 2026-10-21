import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.hybrid_kv_cache import (
    build_hybrid_kv_cache_spec_descs,
    configure_recurrent_cache_capacity,
)
from rtp_llm.ops import CacheCapacityPolicyDesc, HybridAttentionType, KVCacheSpecType


class RecurrentCacheCapacityTest(unittest.TestCase):
    def config(self, types=None):
        config = ModelConfig()
        config.gen_num_per_cycle = 4
        config.kv_cache_spec_descs = build_hybrid_kv_cache_spec_descs(
            types or [HybridAttentionType.LINEAR] * 3 + [HybridAttentionType.NONE],
            KVCacheSpecType.MHA,
        )
        return config

    def test_production_concurrency_reserves_only_recurrent_pools(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71)
        for descs in config.kv_cache_spec_descs:
            for desc in descs:
                if desc.cache_type == KVCacheSpecType.LINEAR:
                    self.assertEqual(desc.capacity.explicit_block_num, 639)
                    self.assertFalse(desc.capacity.reservable)
                    self.assertTrue(desc.capacity.charge_to_paged_budget)
                else:
                    self.assertIsNone(desc.capacity)

    def test_preserves_explicit_operator_capacity(self):
        config = self.config()
        descs = config.kv_cache_spec_descs
        capacity = CacheCapacityPolicyDesc()
        capacity.explicit_block_num = 123
        capacity.reservable = True
        capacity.charge_to_paged_budget = False
        descs[0][0].capacity = capacity
        config.kv_cache_spec_descs = descs
        configure_recurrent_cache_capacity(config, 71)
        actual = config.kv_cache_spec_descs[0][0].capacity
        self.assertEqual(actual.explicit_block_num, 123)
        self.assertTrue(actual.reservable)
        self.assertFalse(actual.charge_to_paged_budget)

    def test_full_attention_draft_keeps_token_budget(self):
        config = self.config([HybridAttentionType.NONE])
        configure_recurrent_cache_capacity(config, 71)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)

    def test_repeated_configuration_preserves_capacity(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71)
        configure_recurrent_cache_capacity(config, 1)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 639
        )

    def test_invalid_concurrency_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            configure_recurrent_cache_capacity(self.config(), 0)


if __name__ == "__main__":
    unittest.main()
