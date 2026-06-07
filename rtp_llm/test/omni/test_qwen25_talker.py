import json
import os
import tempfile
import unittest

from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker


class TestQwen25OmniTalker(unittest.TestCase):
    def _make_config_dir(self):
        tmpdir = tempfile.mkdtemp()
        config = {
            "architectures": ["Qwen2_5OmniModel"],
            "model_type": "qwen2_5_omni",
            "talker_config": {
                "model_type": "qwen2_5_omni_talker",
                "hidden_size": 896,
                "intermediate_size": 18944,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "num_hidden_layers": 24,
                "vocab_size": 8448,
                "rms_norm_eps": 1e-06,
                "rope_theta": 1000000,
                "tie_word_embeddings": False,
                "embedding_size": 3584,
                "torch_dtype": "bfloat16",
            },
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        return tmpdir

    def test_create_config_reads_talker_config(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniTalker._create_config(tmpdir)
        self.assertEqual(config.attn_config.head_num, 12)
        self.assertEqual(config.attn_config.kv_head_num, 4)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.inter_size, 18944)
        self.assertEqual(config.vocab_size, 8448)
        self.assertEqual(config.hidden_size, 896)

    def test_create_config_sets_embedding_size(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniTalker._create_config(tmpdir)
        self.assertEqual(config.embedding_size, 3584)

    def test_create_config_sets_ckpt_path(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniTalker._create_config(tmpdir)
        self.assertEqual(config.ckpt_path, tmpdir)


if __name__ == "__main__":
    unittest.main()
