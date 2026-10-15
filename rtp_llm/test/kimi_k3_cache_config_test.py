import unittest

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3ModelConfig, KimiK3Mtp
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType
from rtp_llm.utils.model_weight import W


class KimiK3MainCacheConfigTest(unittest.TestCase):
    def test_hybrid_schedule_builds_separate_main_cache_groups(self):
        config = KimiK3ModelConfig()
        config.num_layers = 4
        KimiK3._parse_hybrid_attention_config(
            {"linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4]}},
            config,
        )
        KimiK3._post_build_model_config(config)

        self.assertEqual(config.hybrid_attention_config.hybrid_attention_types, [
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ])
        self.assertEqual([descs[0].tag for descs in config.kv_cache_spec_descs], [
            "linear", "linear", "linear", "full"
        ])
        self.assertEqual([descs[0].cache_type for descs in config.kv_cache_spec_descs], [
            KVCacheSpecType.LINEAR,
            KVCacheSpecType.LINEAR,
            KVCacheSpecType.LINEAR,
            KVCacheSpecType.MLA,
        ])


class KimiK3MtpMediaTokenConfigTest(unittest.TestCase):
    @staticmethod
    def _checkpoint_config():
        return {
            "model_type": "kimi_k3",
            "text_config": {
                "model_type": "kimi_linear",
                "num_hidden_layers": 4,
                "num_nextn_predict_layers": 1,
                "hidden_size": 64,
                "vocab_size": 128,
                "max_position_embeddings": 128,
                "intermediate_size": 32,
                "num_attention_heads": 8,
                "q_lora_rank": 32,
                "kv_lora_rank": 32,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 8,
                "v_head_dim": 8,
                "num_experts": 16,
                "num_experts_per_token": 2,
                "moe_intermediate_size": 32,
                "routed_expert_hidden_size": 32,
                "num_shared_experts": 1,
                "hidden_act": "situ",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "attn_res_block_size": 4,
                "mla_use_nope": True,
                "mla_use_output_gate": True,
                "linear_attn_config": {
                    "num_heads": 8,
                    "head_dim": 8,
                    "gate_lower_bound": -5.0,
                    "kda_layers": [1, 2, 3],
                    "full_attn_layers": [4, 5],
                },
                "quantization_config": {
                    "format": "mxfp4-pack-quantized",
                    "config_groups": {
                        "group_0": {"weights": {"num_bits": 4, "group_size": 32}}
                    },
                },
            },
        }

    def test_media_token_is_retained_without_configuring_vit(self):
        for media_token_id in (0, 17, 127):
            with self.subTest(media_token_id=media_token_id):
                checkpoint = self._checkpoint_config()
                checkpoint["media_placeholder_token_id"] = media_token_id
                # Draft token restoration needs no vision processor config or files.
                checkpoint["vision_config"] = {"vt_hidden_size": 8}
                config = KimiK3Mtp._from_config_json(checkpoint)

                self.assertEqual(
                    config.mm_related_params.special_token_ids["image_token_index"],
                    media_token_id,
                )
                self.assertFalse(config.mm_model_config.is_multimodal)
                self.assertNotIn("vision_config", config.mm_related_params.config)
                self.assertTrue(config.is_mtp)

    def test_text_only_checkpoint_does_not_require_media_token(self):
        config = KimiK3Mtp._from_config_json(self._checkpoint_config())

        self.assertNotIn("image_token_index", config.mm_related_params.special_token_ids)
        self.assertFalse(config.mm_model_config.is_multimodal)
        self.assertTrue(config.is_mtp)

    def test_media_token_must_be_within_vocabulary(self):
        for media_token_id in (-1, 128):
            with self.subTest(media_token_id=media_token_id):
                checkpoint = self._checkpoint_config()
                checkpoint["media_placeholder_token_id"] = media_token_id
                with self.assertRaisesRegex(
                    ValueError, "media_placeholder_token_id.*within the vocabulary"
                ):
                    KimiK3Mtp._from_config_json(checkpoint)


class KimiK3MtpMediaTokenPairingTest(unittest.TestCase):
    @staticmethod
    def _pair(target_id=None, draft_id=None):
        configs = (KimiK3ModelConfig(), KimiK3ModelConfig())
        for config, media_id in zip(configs, (target_id, draft_id)):
            config.vocab_size = 128
            config.hidden_size = 64
            config.data_type = "bf16"
            config.enable_fp32_lm_head = False
            config.mm_model_config.is_multimodal = False
            if media_id is not None:
                config.mm_related_params.special_token_ids["image_token_index"] = media_id
        target = object.__new__(KimiK3)
        target.model_config = configs[0]
        target.model_config.mm_model_config.is_multimodal = target_id is not None
        return target, configs[1]

    def test_draft_inherits_target_media_token_without_enabling_vit(self):
        for media_id in (0, 17):
            with self.subTest(media_id=media_id):
                target, draft = self._pair(target_id=media_id)
                aliases = KimiK3Mtp.speculative_weight_alias_names(target, draft)
                self.assertEqual(aliases, (W.embedding, W.lm_head))
                self.assertEqual(
                    draft.mm_related_params.special_token_ids["image_token_index"],
                    media_id,
                )
                self.assertFalse(draft.mm_model_config.is_multimodal)

    def test_matching_media_tokens_and_text_only_pair_keep_weight_aliases(self):
        for media_id in (0, 17, None):
            with self.subTest(media_id=media_id):
                target, draft = self._pair(media_id, media_id)
                aliases = KimiK3Mtp.speculative_weight_alias_names(target, draft)
                self.assertEqual(aliases, (W.embedding, W.lm_head))
                self.assertFalse(draft.mm_model_config.is_multimodal)
                if media_id is None:
                    for config in (target.model_config, draft):
                        self.assertNotIn(
                            "image_token_index", config.mm_related_params.special_token_ids
                        )
                else:
                    self.assertEqual(
                        draft.mm_related_params.special_token_ids["image_token_index"],
                        media_id,
                    )

    def test_mismatched_media_tokens_fail_at_pairing(self):
        for target_id, draft_id in ((0, 17), (17, 0)):
            with self.subTest(target_id=target_id, draft_id=draft_id):
                target, draft = self._pair(target_id, draft_id)
                with self.assertRaisesRegex(ValueError, "media"):
                    KimiK3Mtp.speculative_weight_alias_names(target, draft)

    def test_target_media_token_must_be_in_vocabulary_before_inheritance(self):
        for media_id in (-1, 128):
            with self.subTest(media_id=media_id):
                target, draft = self._pair(target_id=media_id)
                with self.assertRaisesRegex(ValueError, "media"):
                    KimiK3Mtp.speculative_weight_alias_names(target, draft)
                self.assertNotIn(
                    "image_token_index", draft.mm_related_params.special_token_ids
                )


if __name__ == "__main__":
    unittest.main()
