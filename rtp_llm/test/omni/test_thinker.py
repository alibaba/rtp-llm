import json
import os
import tempfile
import unittest

from rtp_llm.model_factory_register import _model_factory
from rtp_llm.models.qwen_v2 import QWenV2Weight


class TestQwen25OmniThinkerRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni.thinker  # noqa: F401

    def test_model_registered(self):
        self.assertIn("qwen2_5_omni_thinker", _model_factory)

    def test_model_class_name(self):
        model_cls = _model_factory["qwen2_5_omni_thinker"]
        self.assertEqual(model_cls.__name__, "Qwen25OmniThinker")


class TestQwen25OmniThinkerWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen25OmniThinkerWeight

        cls.weight_cls = Qwen25OmniThinkerWeight

    def test_inherits_from_qwen_v2_weight(self):
        self.assertTrue(issubclass(self.weight_cls, QWenV2Weight))

    def test_prefix_is_thinker(self):
        w = self.weight_cls.__new__(self.weight_cls)
        w.prefix = "thinker."
        w.model_prefix = "model."
        self.assertEqual(w.prefix, "thinker.")

    def test_transformer_prefix(self):
        w = self.weight_cls.__new__(self.weight_cls)
        w.prefix = "thinker."
        w.model_prefix = "model."
        w.weight_style = None
        w._process_meta([{}], [])
        self.assertEqual(w.transformer_prefix, "thinker.model.")


class TestQwen25OmniThinkerConfig(unittest.TestCase):
    def test_create_config_from_omni_checkpoint(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen25OmniThinker

        config_json = {
            "architectures": ["Qwen2_5OmniModel"],
            "model_type": "qwen2_5_omni",
            "thinker_config": {
                "text_config": {
                    "hidden_size": 3584,
                    "intermediate_size": 18944,
                    "num_attention_heads": 28,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 28,
                    "vocab_size": 152064,
                    "rms_norm_eps": 1e-06,
                    "rope_theta": 1000000.0,
                    "tie_word_embeddings": False,
                    "torch_dtype": "bfloat16",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_json, f)

            config = Qwen25OmniThinker._create_config(tmpdir)
            self.assertEqual(config.hidden_size, 3584)
            self.assertEqual(config.num_layers, 28)
            self.assertEqual(config.attn_config.head_num, 28)
            self.assertEqual(config.attn_config.kv_head_num, 4)
            self.assertEqual(config.inter_size, 18944)
            self.assertEqual(config.vocab_size, 152064)


if __name__ == "__main__":
    unittest.main()
