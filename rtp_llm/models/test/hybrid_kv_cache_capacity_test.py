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
        config.recurrent_cache_prefix_blocks_per_request = 2
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
                    self.assertEqual(desc.capacity.explicit_block_num, 640)
                    self.assertFalse(desc.capacity.reservable)
                    self.assertTrue(desc.capacity.charge_to_paged_budget)
                else:
                    self.assertIsNone(desc.capacity)

    def test_usable_pool_can_allocate_every_active_request_state(self):
        # Model BlockPool's real ID namespace: zero is never allocatable.
        # Include simultaneous retained tails, all four proposals, replacement
        # before cleanup, and the configured reusable-prefix allowance.
        required_roles = [
            "retained_tail_0",
            "retained_tail_1",
            "proposal_0",
            "proposal_1",
            "proposal_2",
            "proposal_3",
            "replacement",
        ]
        for concurrency, prefixes, expected_usable in (
            (1, 0, 7),
            (1, 2, 9),
            (71, 0, 497),
            (71, 2, 639),
        ):
            with self.subTest(concurrency=concurrency, prefixes=prefixes):
                config = self.config()
                config.recurrent_cache_prefix_blocks_per_request = prefixes
                configure_recurrent_cache_capacity(config, concurrency)
                roles = required_roles + [f"prefix_{i}" for i in range(prefixes)]
                for descs in config.kv_cache_spec_descs:
                    for desc in descs:
                        if desc.cache_type != KVCacheSpecType.LINEAR:
                            continue
                        free_ids = set(range(1, desc.capacity.explicit_block_num))
                        self.assertEqual(len(free_ids), expected_usable)
                        active_states = {}
                        for request in range(concurrency):
                            for role in roles:
                                self.assertTrue(free_ids, (request, role))
                                active_states[request, role] = free_ids.pop()
                        self.assertEqual(
                            len(set(active_states.values())), expected_usable
                        )
                        self.assertNotIn(0, active_states.values())
                        self.assertFalse(free_ids)

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

    def test_repeated_configuration_recomputes_its_own_capacity(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71)
        configure_recurrent_cache_capacity(config, 1)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 10
        )

    def test_default_policy_keeps_token_budget(self):
        config = self.config()
        config.recurrent_cache_prefix_blocks_per_request = None
        configure_recurrent_cache_capacity(config, 71)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)

    def test_prefix_allowance_can_grow_and_policy_can_be_disabled(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71)
        config.recurrent_cache_prefix_blocks_per_request = 10
        configure_recurrent_cache_capacity(config, 71)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 1208
        )
        config.recurrent_cache_prefix_blocks_per_request = None
        configure_recurrent_cache_capacity(config, 71)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)

    def test_operator_change_after_auto_configuration_is_preserved(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71)
        descs = config.kv_cache_spec_descs
        for layer in descs:
            for desc in layer:
                if desc.cache_type == KVCacheSpecType.LINEAR:
                    capacity = desc.capacity
                    capacity.explicit_block_num = 123
                    desc.capacity = capacity
        config.kv_cache_spec_descs = descs
        configure_recurrent_cache_capacity(config, 1)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 123
        )

    def test_explicit_zero_is_not_mistaken_for_automatic_policy(self):
        config = self.config()
        descs = config.kv_cache_spec_descs
        for layer in descs:
            for desc in layer:
                if desc.cache_type == KVCacheSpecType.LINEAR:
                    capacity = CacheCapacityPolicyDesc()
                    capacity.explicit_block_num = 0
                    desc.capacity = capacity
        config.kv_cache_spec_descs = descs
        configure_recurrent_cache_capacity(config, 71)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 0
        )

    def test_invalid_concurrency_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            configure_recurrent_cache_capacity(self.config(), 0)


if __name__ == "__main__":
    unittest.main()
