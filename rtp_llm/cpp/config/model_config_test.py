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


class ModelConfigMtpHiddenSizeTest(unittest.TestCase):
    def test_hidden_width_product_boundaries(self):
        maximum = 2**63 - 1
        for hidden, multiplier in ((1024, 4), (maximum, 1), (maximum // 2, 2)):
            with self.subTest(hidden=hidden, multiplier=multiplier):
                config = ModelConfig()
                config.hidden_size = hidden
                config.hc_mult = multiplier
                self.assertEqual(config.getMtpInputHiddenSize(), hidden * multiplier)
        for hidden, multiplier, error in (
            (0, 1, "cannot resolve"),
            (-1, 1, "cannot resolve"),
            (1, 0, "cannot resolve"),
            (1, -1, "cannot resolve"),
            (maximum, 2, "overflows int64_t"),
        ):
            with self.subTest(hidden=hidden, multiplier=multiplier):
                config = ModelConfig()
                config.hidden_size = hidden
                config.hc_mult = multiplier
                with self.assertRaisesRegex(RuntimeError, error):
                    config.getMtpInputHiddenSize()


if __name__ == "__main__":
    unittest.main()
