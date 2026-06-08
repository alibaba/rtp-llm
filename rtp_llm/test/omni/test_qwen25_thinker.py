import json
import os
import tempfile
import unittest

from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker


class TestQwen25OmniThinker(unittest.TestCase):
    def _make_config_dir(self):
        tmpdir = tempfile.mkdtemp()
        config = {
            "architectures": ["Qwen2_5OmniModel"],
            "model_type": "qwen2_5_omni",
            "thinker_config": {
                "model_type": "qwen2_5_omni_thinker",
                "text_config": {
                    "hidden_size": 3584,
                    "intermediate_size": 18944,
                    "num_attention_heads": 28,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 28,
                    "vocab_size": 152064,
                    "rms_norm_eps": 1e-06,
                    "rope_theta": 1000000,
                    "tie_word_embeddings": False,
                    "torch_dtype": "bfloat16",
                },
                "audio_config": {
                    "model_type": "qwen2_5_omni_audio_encoder",
                    "d_model": 1280,
                    "encoder_attention_heads": 20,
                    "encoder_ffn_dim": 5120,
                    "encoder_layers": 32,
                    "num_mel_bins": 128,
                    "output_dim": 3584,
                },
                "vision_config": {
                    "model_type": "qwen2_5_omni_vision_encoder",
                    "embed_dim": 1280,
                    "depth": 32,
                    "num_heads": 16,
                },
                "audio_token_index": 151646,
                "image_token_index": 151655,
            },
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        return tmpdir

    def test_create_config_reads_thinker_text_config(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniThinker._create_config(tmpdir)
        self.assertEqual(config.attn_config.head_num, 28)
        self.assertEqual(config.attn_config.kv_head_num, 4)
        self.assertEqual(config.num_layers, 28)
        self.assertEqual(config.inter_size, 18944)
        self.assertEqual(config.vocab_size, 152064)
        self.assertEqual(config.hidden_size, 3584)

    def test_create_config_sets_ckpt_path(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniThinker._create_config(tmpdir)
        self.assertEqual(config.ckpt_path, tmpdir)

    def test_create_config_reads_rope_theta(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniThinker._create_config(tmpdir)
        self.assertEqual(config.attn_config.rope_config.base, 1000000)


if __name__ == "__main__":
    unittest.main()
