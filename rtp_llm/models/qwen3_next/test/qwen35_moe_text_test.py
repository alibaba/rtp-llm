"""Configuration-only checks; no full model allocation or checkpoint load."""

import unittest
from types import SimpleNamespace

from rtp_llm.model_factory_register import _model_factory
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe, Qwen35MoeText


class Qwen35MoeTextTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            ckpt_path="checkpoint",
            mm_model_config=SimpleNamespace(
                is_multimodal=True, mm_sep_tokens=[[1, 2]], mm_position_ids_style=2
            ),
            mm_related_params=SimpleNamespace(config={}),
        )

    def test_text_registration_omits_vision_without_discarding_mrope(self):
        config = self.config()
        Qwen35MoeText._parse_mm_config({}, config)
        self.assertFalse(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [])
        self.assertEqual(config.mm_model_config.mm_position_ids_style, 2)
        self.assertIs(_model_factory["qwen35_moe_text"], Qwen35MoeText)
        self.assertIs(_model_factory["qwen36_moe_text"], Qwen35MoeText)

    def test_original_multimodal_model_retains_vision_configuration(self):
        config = self.config()
        Qwen35Moe._parse_mm_config(
            {"vision_start_token_id": 11, "vision_end_token_id": 12}, config
        )
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[11, 12]])
        self.assertEqual(config.mm_related_params.config["ckpt_path"], "checkpoint")


if __name__ == "__main__":
    unittest.main()
