"""DSA schedule and cache contracts without loading CUDA/native operators."""

import copy
import unittest

from rtp_llm.config.dsa_config import DsaConfig


def text_config(layers=93):
    dsa_layers = list(range(4, layers + 1, 4))
    if layers == 93:
        dsa_layers.append(93)
    return {
        "num_hidden_layers": layers,
        "linear_attn_config": {"full_attn_layers": dsa_layers},
        "dsa_config": {
            "dsa_layers": dsa_layers,
            "index_num_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "indexer_use_fp8": False,
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
    }


class DsaConfigTest(unittest.TestCase):
    def test_full_schedule_and_consecutive_final_layers(self):
        config = DsaConfig.from_kimi_config(text_config())
        self.assertEqual(len(config.layers), 24)
        self.assertEqual(config.layers, tuple(range(3, 92, 4)) + (92,))
        self.assertTrue(config.has_layer(91))
        self.assertTrue(config.has_layer(92))
        self.assertFalse(config.has_layer(90))

    def test_four_layer_schedule(self):
        config = DsaConfig.from_kimi_config(text_config(4))
        self.assertEqual(config.layers, (3,))

    def test_parsing_does_not_modify_checkpoint_config(self):
        text = text_config()
        original = copy.deepcopy(text)
        DsaConfig.from_kimi_config(text)
        self.assertEqual(text, original)

    def test_bf16_cache_is_not_legacy_fp8_cache(self):
        config = DsaConfig.from_kimi_config(text_config())
        self.assertEqual(config.index_cache_entry_bytes, 256)
        text = text_config()
        text["dsa_config"]["indexer_use_fp8"] = True
        self.assertEqual(DsaConfig.from_kimi_config(text).index_cache_entry_bytes, 132)

    def test_invalid_schedules(self):
        for layers in ([0], [94], [True], [4, 4], [], "4"):
            with self.subTest(layers=layers):
                text = text_config()
                text["dsa_config"]["dsa_layers"] = layers
                with self.assertRaises(ValueError):
                    DsaConfig.from_kimi_config(text)

    def test_dsa_schedule_cannot_silently_differ_from_full_attention(self):
        text = text_config()
        text["dsa_config"]["dsa_layers"] = [4]
        with self.assertRaisesRegex(ValueError, "match full_attn_layers"):
            DsaConfig.from_kimi_config(text)

    def test_model_differences_are_explicit(self):
        for name, value in (
            ("index_rope_dim", 64), ("index_k_norm_type", "rmsnorm"),
            ("index_topk_freq", 2), ("index_skip_topk_offset", 0),
            ("index_head_dim", 64), ("indexer_use_fp8", 0),
            ("index_num_heads", True), ("index_topk", -1),
        ):
            with self.subTest(field=name):
                text = text_config()
                text["dsa_config"][name] = value
                with self.assertRaises(ValueError):
                    DsaConfig.from_kimi_config(text)


if __name__ == "__main__":
    unittest.main()
