from types import SimpleNamespace
from unittest import TestCase, main

from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.utils.model_weight import W


class Glm5WeightAliasTest(TestCase):
    def setUp(self):
        common = dict(
            ckpt_path="/tmp/glm5-full",
            vocab_size=256,
            hidden_size=128,
            data_type="bf16",
            enable_fp32_lm_head=True,
            normalize_lm_head_weight=False,
            logit_scale=1.0,
        )
        self.target_config = SimpleNamespace(model_type="glm_5", **common)
        self.draft_config = SimpleNamespace(
            model_type="glm_5_mtp", mtp_layer_offset=78, **common
        )
        self.target = SimpleNamespace(
            model_config=self.target_config,
            merge_lora=True,
            model_weights_loader=SimpleNamespace(
                _load_config=SimpleNamespace(merge_lora=False)
            ),
        )

    def aliases(self):
        return DeepSeekV2.speculative_weight_alias_names(self.target, self.draft_config)

    def test_full_checkpoint_uses_effective_lora_flag(self):
        self.assertEqual(self.aliases(), (W.embedding, W.lm_head))
        self.target.model_weights_loader._load_config.merge_lora = True
        self.assertEqual(self.aliases(), ())

    def test_missing_loader_fails_closed_for_lora(self):
        del self.target.model_weights_loader
        self.assertEqual(self.aliases(), ())
        self.target.merge_lora = False
        self.assertEqual(self.aliases(), (W.embedding, W.lm_head))

    def test_extracted_checkpoint_and_other_models_do_not_alias(self):
        self.draft_config.mtp_layer_offset = 0
        self.assertEqual(self.aliases(), ())
        self.draft_config.mtp_layer_offset = 78
        self.target_config.model_type = "deepseek_v3"
        self.assertEqual(self.aliases(), ())

    def test_path_shape_dtype_and_head_semantics_must_match(self):
        for name, value in (
            ("ckpt_path", "/tmp/other"),
            ("vocab_size", 257),
            ("hidden_size", 129),
            ("data_type", "fp16"),
            ("enable_fp32_lm_head", False),
            ("normalize_lm_head_weight", True),
            ("logit_scale", 0.5),
        ):
            with self.subTest(name=name):
                previous = getattr(self.draft_config, name)
                setattr(self.draft_config, name, value)
                self.assertEqual(self.aliases(), ())
                setattr(self.draft_config, name, previous)
        self.target_config.hidden_size = None
        self.draft_config.hidden_size = None
        self.assertEqual(self.aliases(), ())



if __name__ == "__main__":
    main()
