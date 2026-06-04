import json
import os
import tempfile
import unittest

from rtp_llm.model_factory_register import _model_factory
from rtp_llm.models.qwen_v2 import QWenV2Weight


class TestQwen25OmniTalkerRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni.talker  # noqa: F401

    def test_model_registered(self):
        self.assertIn("qwen2_5_omni_talker", _model_factory)

    def test_model_class_name(self):
        model_cls = _model_factory["qwen2_5_omni_talker"]
        self.assertEqual(model_cls.__name__, "Qwen25OmniTalker")


class TestQwen25OmniTalkerWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen25OmniTalkerWeight

        cls.weight_cls = Qwen25OmniTalkerWeight

    def test_inherits_from_qwen_v2_weight(self):
        self.assertTrue(issubclass(self.weight_cls, QWenV2Weight))

    def test_prefix_is_talker(self):
        w = self.weight_cls.__new__(self.weight_cls)
        w.prefix = "talker."
        w.model_prefix = "model."
        self.assertEqual(w.prefix, "talker.")

    def test_transformer_prefix(self):
        w = self.weight_cls.__new__(self.weight_cls)
        w.prefix = "talker."
        w.model_prefix = "model."
        w.weight_style = None
        w._process_meta([{}], [])
        self.assertEqual(w.transformer_prefix, "talker.model.")


class TestQwen25OmniTalkerConfig(unittest.TestCase):
    def test_create_config_from_omni_checkpoint(self):
        from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen25OmniTalker

        config_json = {
            "architectures": ["Qwen2_5OmniModel"],
            "model_type": "qwen2_5_omni",
            "talker_config": {
                "hidden_size": 896,
                "intermediate_size": 18944,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "num_hidden_layers": 24,
                "vocab_size": 8448,
                "rms_norm_eps": 1e-06,
                "rope_theta": 1000000.0,
                "embedding_size": 3584,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_json, f)

            config = Qwen25OmniTalker._create_config(tmpdir)
            self.assertEqual(config.hidden_size, 896)
            self.assertEqual(config.num_layers, 24)
            self.assertEqual(config.attn_config.head_num, 12)
            self.assertEqual(config.attn_config.kv_head_num, 4)
            self.assertEqual(config.vocab_size, 8448)


if __name__ == "__main__":
    unittest.main()
