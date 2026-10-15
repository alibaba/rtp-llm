import unittest
from dataclasses import replace
from unittest.mock import patch

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
    def setUp(self):
        backend = patch.dict("os.environ", {"KIMI_K3_KDA_PREFILL_BACKEND": "rtp"})
        backend.start()
        self.addCleanup(backend.stop)

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
                self.assertEqual(
                    config.k3_runtime_config.media_placeholder_token_id, media_token_id
                )
                self.assertFalse(config.mm_model_config.is_multimodal)
                self.assertNotIn("vision_config", config.mm_related_params.config)
                self.assertTrue(config.is_mtp)

    def test_text_only_checkpoint_does_not_require_media_token(self):
        config = KimiK3Mtp._from_config_json(self._checkpoint_config())

        self.assertNotIn(
            "image_token_index", config.mm_related_params.special_token_ids
        )
        self.assertIsNone(config.k3_runtime_config.media_placeholder_token_id)
        self.assertFalse(config.mm_model_config.is_multimodal)
        self.assertTrue(config.is_mtp)

    def test_media_token_must_be_an_integer_vocabulary_id(self):
        for media_token_id in (-1, 128, 1.5, True, False, "17"):
            with self.subTest(media_token_id=media_token_id):
                checkpoint = self._checkpoint_config()
                checkpoint["media_placeholder_token_id"] = media_token_id
                with self.assertRaisesRegex(ValueError, "media_placeholder_token_id"):
                    KimiK3Mtp._from_config_json(checkpoint)

    def test_outer_media_token_takes_precedence_over_text_config(self):
        checkpoint = self._checkpoint_config()
        checkpoint["media_placeholder_token_id"] = 17
        checkpoint["text_config"]["media_placeholder_token_id"] = 3
        config = KimiK3Mtp._from_config_json(checkpoint)
        self.assertEqual(config.k3_runtime_config.media_placeholder_token_id, 17)
        self.assertEqual(
            config.mm_related_params.special_token_ids["image_token_index"], 17
        )


class KimiK3MtpMediaTokenPairingTest(unittest.TestCase):
    def setUp(self):
        backend = patch.dict("os.environ", {"KIMI_K3_KDA_PREFILL_BACKEND": "rtp"})
        backend.start()
        self.addCleanup(backend.stop)

    @staticmethod
    def _pair(target_id=None, draft_id=None, *, runtime_id=None):
        target_config = KimiK3ModelConfig()
        draft = KimiK3Mtp._from_config_json(
            KimiK3MtpMediaTokenConfigTest._checkpoint_config()
        )
        for config, media_id in ((target_config, target_id), (draft, draft_id)):
            config.vocab_size = 128
            config.input_vocab_size = 128
            config.hidden_size = 64
            config.data_type = "bf16"
            config.enable_fp32_lm_head = False
            config.mm_model_config.is_multimodal = False
            if media_id is not None:
                config.mm_related_params.special_token_ids["image_token_index"] = (
                    media_id
                )
        draft.k3_runtime_config = replace(
            draft.k3_runtime_config, media_placeholder_token_id=runtime_id
        )
        target = object.__new__(KimiK3)
        target.model_config = target_config
        target.model_config.mm_model_config.is_multimodal = target_id is not None
        return target, draft

    def test_target_fills_missing_runtime_or_special_media_id_without_enabling_vit(
        self,
    ):
        for media_id in (0, 17):
            for special_id, runtime_id in (
                (None, None),
                (media_id, None),
                (None, media_id),
                (media_id, media_id),
            ):
                with self.subTest(
                    media_id=media_id, special_id=special_id, runtime_id=runtime_id
                ):
                    target, draft = self._pair(
                        media_id, special_id, runtime_id=runtime_id
                    )
                    aliases = KimiK3Mtp.speculative_weight_alias_names(target, draft)
                    self.assertEqual(aliases, (W.embedding, W.lm_head))
                    self.assertEqual(
                        draft.mm_related_params.special_token_ids["image_token_index"],
                        media_id,
                    )
                    self.assertEqual(
                        draft.k3_runtime_config.media_placeholder_token_id, media_id
                    )
                    self.assertFalse(draft.mm_model_config.is_multimodal)

    def test_explicit_none_special_id_can_inherit_target_media_token(self):
        target, draft = self._pair(target_id=17)
        draft.mm_related_params.special_token_ids["image_token_index"] = None
        aliases = KimiK3Mtp.speculative_weight_alias_names(target, draft)
        self.assertEqual(aliases, (W.embedding, W.lm_head))
        self.assertEqual(draft.k3_runtime_config.media_placeholder_token_id, 17)
        self.assertEqual(
            draft.mm_related_params.special_token_ids["image_token_index"], 17
        )

    def test_text_only_pair_keeps_weight_aliases_without_media_id_or_vit(self):
        target, draft = self._pair()
        aliases = KimiK3Mtp.speculative_weight_alias_names(target, draft)
        self.assertEqual(aliases, (W.embedding, W.lm_head))
        self.assertIsNone(draft.k3_runtime_config.media_placeholder_token_id)
        for config in (target.model_config, draft):
            self.assertNotIn(
                "image_token_index", config.mm_related_params.special_token_ids
            )
            self.assertFalse(config.mm_model_config.is_multimodal)

    def test_mismatched_runtime_or_special_id_fails_without_partial_inheritance(self):
        for target_id, other_id in ((0, 17), (17, 0)):
            for special_id, runtime_id in (
                (other_id, None),
                (None, other_id),
                (target_id, other_id),
                (other_id, target_id),
            ):
                with self.subTest(
                    target_id=target_id, special_id=special_id, runtime_id=runtime_id
                ):
                    target, draft = self._pair(
                        target_id, special_id, runtime_id=runtime_id
                    )
                    original_ids = dict(draft.mm_related_params.special_token_ids)
                    original_runtime = draft.k3_runtime_config
                    with self.assertRaisesRegex(ValueError, "media"):
                        KimiK3Mtp.speculative_weight_alias_names(target, draft)
                    self.assertEqual(
                        draft.mm_related_params.special_token_ids, original_ids
                    )
                    self.assertIs(draft.k3_runtime_config, original_runtime)

    def test_existing_draft_media_ids_must_be_integer_vocabulary_ids(self):
        for target_id, invalid_id in (
            (0, False),
            (1, True),
            (17, 17.0),
            (17, "17"),
            (17, -1),
            (17, 128),
        ):
            for source in ("special", "runtime"):
                with self.subTest(
                    target_id=target_id, invalid_id=invalid_id, source=source
                ):
                    target, draft = self._pair(
                        target_id,
                        invalid_id if source == "special" else target_id,
                        runtime_id=invalid_id if source == "runtime" else target_id,
                    )
                    original_ids = dict(draft.mm_related_params.special_token_ids)
                    original_runtime = draft.k3_runtime_config
                    with self.assertRaisesRegex(ValueError, "media"):
                        KimiK3Mtp.speculative_weight_alias_names(target, draft)
                    self.assertEqual(
                        draft.mm_related_params.special_token_ids, original_ids
                    )
                    self.assertIs(draft.k3_runtime_config, original_runtime)

    def test_target_media_id_is_validated_before_inheritance(self):
        for media_id in (-1, 128, 1.5, True, False, "17"):
            with self.subTest(media_id=media_id):
                target, draft = self._pair(target_id=media_id)
                original_runtime = draft.k3_runtime_config
                with self.assertRaisesRegex(ValueError, "media"):
                    KimiK3Mtp.speculative_weight_alias_names(target, draft)
                self.assertNotIn(
                    "image_token_index", draft.mm_related_params.special_token_ids
                )
                self.assertIs(draft.k3_runtime_config, original_runtime)


if __name__ == "__main__":
    unittest.main()
