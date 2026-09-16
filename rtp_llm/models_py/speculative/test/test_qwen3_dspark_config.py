"""CPU configuration contracts for TorchSpec Qwen3 DSpark checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.qwen_3_dspark import Qwen3DSpark
from rtp_llm.ops import DataType
from rtp_llm.utils.model_weight import W


class Qwen3DSparkConfigTest(unittest.TestCase):
    @staticmethod
    def _torchspec_config():
        return {
            "architectures": ["Qwen3DSparkForCausalLM"],
            "model_type": "qwen_3_dspark",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 5,
            "vocab_size": 248320,
            "aux_hidden_state_layer_ids": [39, 47, 55],
            "mask_token_id": 248063,
            "markov_rank": 512,
            "block_size": 5,
            "sample_from_anchor": True,
            "lm_head_source": "target",
            "dtype": "bfloat16",
        }

    def test_torchspec_config_keeps_zero_based_layers_and_runtime_width(self):
        raw = self._torchspec_config()
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)

        self.assertEqual(ModelDict.get_ft_model_type_by_config(raw), "qwen_3_dspark")
        self.assertEqual(config.dspark_target_layer_ids, [39, 47, 55])
        self.assertTrue(config.dspark_sample_from_anchor)
        self.assertTrue(config.dspark_share_target_lm_head)
        self.assertEqual(config.attn_config.size_per_head, 128)
        for gamma in (5, 7):
            with self.subTest(gamma=gamma):
                sp_config = SimpleNamespace(
                    gen_num_per_cycle=gamma,
                    sp_dspark_mask_token_id=-1,
                    sp_dspark_sample_from_anchor=False,
                )
                target = SimpleNamespace(
                    num_layers=60, capture_aux_hidden_layer_ids=None
                )
                ModelFactory._setup_dspark_configs(sp_config, target, config)
                self.assertEqual(sp_config.gen_num_per_cycle, gamma)
                self.assertTrue(sp_config.sp_dspark_sample_from_anchor)
                self.assertEqual(target.capture_aux_hidden_layer_ids, [39, 47, 55])
                self.assertEqual(sp_config.sp_dspark_mask_token_id, 248063)

    def test_invalid_lm_head_source_is_rejected(self):
        raw = self._torchspec_config()
        raw["lm_head_source"] = "embedding"
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "lm_head_source"):
                Qwen3DSpark._create_config(path)

    def test_shared_head_requires_compatible_target_but_preserves_draft_weights(self):
        values = {
            "hidden_size": 4096,
            "vocab_size": 248320,
            "data_type": DataType.TYPE_BF16,
            "enable_fp32_lm_head": True,
            "normalize_lm_head_weight": False,
        }
        target = SimpleNamespace(model_config=SimpleNamespace(**values))
        draft = SimpleNamespace(dspark_share_target_lm_head=True, **values)
        self.assertEqual(
            Qwen3DSpark.speculative_weight_alias_names(target, draft), (W.lm_head,)
        )
        for name, invalid in (
            ("hidden_size", 5120),
            ("vocab_size", 20000),
            ("data_type", DataType.TYPE_FP16),
            ("enable_fp32_lm_head", False),
            ("normalize_lm_head_weight", True),
        ):
            with self.subTest(field=name):
                setattr(draft, name, invalid)
                with self.assertRaisesRegex(ValueError, name):
                    Qwen3DSpark.speculative_weight_alias_names(target, draft)
                setattr(draft, name, values[name])

    def test_checkpoint_owned_head_never_borrows_target_weights(self):
        draft = SimpleNamespace(dspark_share_target_lm_head=False)
        self.assertEqual(Qwen3DSpark.speculative_weight_alias_names(None, draft), ())

    def test_legacy_config_without_head_source_keeps_checkpoint_head(self):
        raw = self._torchspec_config()
        del raw["lm_head_source"]
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)
        self.assertFalse(config.dspark_share_target_lm_head)
        self.assertEqual(Qwen3DSpark.speculative_weight_alias_names(None, config), ())


if __name__ == "__main__":
    unittest.main()
