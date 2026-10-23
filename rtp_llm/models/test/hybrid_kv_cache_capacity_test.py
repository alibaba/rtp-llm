import unittest
from unittest.mock import patch

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
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        for descs in config.kv_cache_spec_descs:
            for desc in descs:
                if desc.cache_type == KVCacheSpecType.LINEAR:
                    self.assertEqual(desc.capacity.explicit_block_num, 995)
                    self.assertFalse(desc.capacity.reservable)
                    self.assertTrue(desc.capacity.charge_to_paged_budget)
                else:
                    self.assertIsNone(desc.capacity)

    def test_prefix_reuse_keeps_token_budget_and_restores_generated_policy(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71, reuse_cache=True)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        self.assertIsNotNone(config.kv_cache_spec_descs[0][0].capacity)
        configure_recurrent_cache_capacity(config, 71, reuse_cache=True)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)

    def test_no_speculation_budget(self):
        config = self.config()
        config.gen_num_per_cycle = 0
        config.recurrent_cache_prefix_blocks_per_request = 0
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 214
        )

    def test_model_load_passes_prefix_reuse_policy(self):
        from types import SimpleNamespace

        from rtp_llm.models.base_model import BaseModel
        from rtp_llm.models.qwen3_next.qwen3_next import Qwen3NextBase

        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                model = Qwen3NextBase.__new__(Qwen3NextBase)
                model.model_config = self.config()
                model.max_generate_batch_size = 71
                model.kv_cache_config = SimpleNamespace(reuse_cache=reuse)
                with patch.object(BaseModel, "load", return_value="loaded"):
                    self.assertEqual(model.load(), "loaded")
                capacity = model.model_config.kv_cache_spec_descs[0][0].capacity
                if reuse:
                    self.assertIsNone(capacity)
                else:
                    self.assertEqual(capacity.explicit_block_num, 995)

    def test_preserves_explicit_operator_capacity(self):
        config = self.config()
        descs = config.kv_cache_spec_descs
        capacity = CacheCapacityPolicyDesc()
        capacity.explicit_block_num = 123
        capacity.reservable = True
        capacity.charge_to_paged_budget = False
        descs[0][0].capacity = capacity
        config.kv_cache_spec_descs = descs
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        actual = config.kv_cache_spec_descs[0][0].capacity
        self.assertEqual(actual.explicit_block_num, 123)
        self.assertTrue(actual.reservable)
        self.assertFalse(actual.charge_to_paged_budget)

    def test_full_attention_draft_keeps_token_budget(self):
        config = self.config([HybridAttentionType.NONE])
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)

    def test_repeated_configuration_recomputes_its_own_capacity(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        configure_recurrent_cache_capacity(config, 1, reuse_cache=False)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 15
        )

    def test_default_policy_keeps_token_budget(self):
        config = self.config()
        config.recurrent_cache_prefix_blocks_per_request = None
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)

    def test_prefix_allowance_can_grow_and_policy_can_be_disabled(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        config.recurrent_cache_prefix_blocks_per_request = 10
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 1563
        )
        config.recurrent_cache_prefix_blocks_per_request = None
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        self.assertIsNone(config.kv_cache_spec_descs[0][0].capacity)

    def test_operator_change_after_auto_configuration_is_preserved(self):
        config = self.config()
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        descs = config.kv_cache_spec_descs
        for layer in descs:
            for desc in layer:
                if desc.cache_type == KVCacheSpecType.LINEAR:
                    capacity = desc.capacity
                    capacity.explicit_block_num = 123
                    desc.capacity = capacity
        config.kv_cache_spec_descs = descs
        configure_recurrent_cache_capacity(config, 1, reuse_cache=False)
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
        configure_recurrent_cache_capacity(config, 71, reuse_cache=False)
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].capacity.explicit_block_num, 0
        )

    def test_invalid_concurrency_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            configure_recurrent_cache_capacity(self.config(), 0, reuse_cache=False)

    def test_weight_converter_zero_concurrency_skips_capacity(self):
        from rtp_llm.models.base_model import BaseModel
        from rtp_llm.models.qwen3_next.qwen3_next import Qwen3NextBase

        model = Qwen3NextBase.__new__(Qwen3NextBase)
        model.model_config = self.config()
        model.max_generate_batch_size = 0
        with patch.object(BaseModel, "load", return_value="loaded") as load, patch(
            "rtp_llm.models.qwen3_next.qwen3_next.configure_recurrent_cache_capacity"
        ) as configure:
            self.assertEqual(model.load(), "loaded")
        load.assert_called_once_with()
        configure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
