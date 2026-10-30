import unittest

from rtp_llm.ops import AttentionConfigs, ModelConfig, RopeConfig, RopeStyle


class ModelConfigAttentionTest(unittest.TestCase):
    def test_need_rope_kv_cache_embedding_override_truth_table(self):
        cases = (
            (RopeStyle.No, False, False, False),
            (RopeStyle.No, False, True, False),
            (RopeStyle.No, True, False, False),
            (RopeStyle.No, True, True, True),
            (RopeStyle.Base, False, False, False),
            (RopeStyle.Base, False, True, True),
            (RopeStyle.Base, True, False, False),
            (RopeStyle.Base, True, True, True),
        )

        for rope_style, use_kvcache, initial_value, expected in cases:
            with self.subTest(
                rope_style=rope_style,
                use_kvcache=use_kvcache,
                initial_value=initial_value,
            ):
                rope_config = RopeConfig()
                rope_config.style = rope_style
                attention_config = AttentionConfigs()
                attention_config.rope_config = rope_config
                attention_config.need_rope_kv_cache = initial_value
                model_config = ModelConfig()
                model_config.attn_config = attention_config
                model_config.use_kvcache = use_kvcache

                derived_config = model_config.getAttentionConfigs(1)

                self.assertEqual(derived_config.need_rope_kv_cache, expected)
                self.assertEqual(
                    model_config.attn_config.need_rope_kv_cache,
                    initial_value,
                    "getAttentionConfigs must not mutate the source config",
                )


class ModelConfigLocalKvShardingTest(unittest.TestCase):
    def test_layout_override_round_trips_through_cpp_binding(self):
        config = ModelConfig()
        self.assertIsNone(config.local_kv_cache_sharded)
        for value in (True, False, None):
            config.local_kv_cache_sharded = value
            self.assertIs(config.local_kv_cache_sharded, value)

if __name__ == "__main__":
    unittest.main()
