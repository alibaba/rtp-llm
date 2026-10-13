import json
import os
import tempfile
import unittest

from rtp_llm.models.minicpmv4_7_dspark import MiniCPMV47DSpark
from rtp_llm.utils.model_weight import W

# Trimmed mirror of openbmb/MiniCPM-V-4.7-35B-A3B-DSpark config.json.
_DSPARK_CONFIG = {
    "architectures": ["Qwen3DSparkModel"],
    "block_size": 15,
    "dflash_config": {
        "attention_mode": "gqa",
        "markov_rank": 256,
        "mask_token_id": 248076,
        "target_layer_ids": [3, 11, 19, 27, 35],
    },
    "draft_vocab_size": 248144,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "layer_types": ["full_attention"] * 5,
    "max_position_embeddings": 262144,
    "num_attention_heads": 8,
    "num_hidden_layers": 5,
    "num_key_value_heads": 2,
    "num_target_layers": 40,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "rope_dim": 256},
    "tie_word_embeddings": False,
    "vocab_size": 248144,
    "markov_rank": 256,
}


class MiniCPMV47DSparkConfigTest(unittest.TestCase):
    def _write_ckpt(self) -> str:
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w") as writer:
            json.dump(_DSPARK_CONFIG, writer)
        return tmp

    def test_dspark_config_parses_target_layers_and_markov(self):
        ckpt = self._write_ckpt()
        config = MiniCPMV47DSpark._create_config(ckpt)

        self.assertEqual(config.num_layers, 5)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.inter_size, 6144)
        self.assertEqual(config.attn_config.head_num, 8)
        self.assertEqual(config.attn_config.kv_head_num, 2)
        self.assertEqual(config.attn_config.size_per_head, 256)
        self.assertFalse(config.attn_config.is_causal)
        self.assertEqual(config.vocab_size, 248144)
        self.assertEqual(config.input_vocab_size, 248144)
        self.assertEqual(config.dspark_noise_token_id, 248076)
        self.assertEqual(config.dspark_target_layer_ids, [3, 11, 19, 27, 35])
        self.assertEqual(config.dspark_markov_rank, 256)
        self.assertFalse(config.dspark_sample_from_anchor)

    def test_target_layers_are_not_one_based_adjusted(self):
        # MiniCPM dflash_config.target_layer_ids are 0-based; [3, 11, ...] must
        # stay unchanged (no Speculators-style -1 shift).
        ckpt = self._write_ckpt()
        config = MiniCPMV47DSpark._create_config(ckpt)
        self.assertEqual(config.dspark_target_layer_ids[0], 3)

    def test_missing_markov_rank_is_rejected(self):
        bad = dict(_DSPARK_CONFIG)
        bad["markov_rank"] = 0
        bad["dflash_config"] = dict(bad["dflash_config"])
        bad["dflash_config"]["markov_rank"] = 0
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w") as writer:
            json.dump(bad, writer)
        with self.assertRaises(ValueError):
            MiniCPMV47DSpark._create_config(tmp)

    def test_weight_slots_exist_for_confidence_head(self):
        self.assertEqual(W.dspark_confidence_w, "dspark_confidence.weight")
        self.assertEqual(W.dspark_confidence_b, "dspark_confidence.bias")


if __name__ == "__main__":
    unittest.main()
