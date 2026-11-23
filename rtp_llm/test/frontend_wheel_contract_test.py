"""Frontend configuration must work without packaging the inference engine."""

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


class FrontendDescriptorChecks:
    def check_descriptors(self):
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.tools.api.model_basic_info_analyzer import parse_ft_model_type

        for model_type in ("qwen_3_dspark", "qwen_3_moe_eagle3", "deepseek_v41"):
            with self.subTest(model_type=model_type):
                self.assertTrue(
                    callable(ModelFactory.get_model_cls(model_type)._create_config)
                )
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "config.json").write_text(
                json.dumps(
                    dict(
                        architectures=["Qwen2ForCausalLM"],
                        hidden_size=16,
                        intermediate_size=32,
                        num_attention_heads=2,
                        num_hidden_layers=1,
                        vocab_size=24,
                    )
                )
            )
            self.assertEqual(parse_ft_model_type(root), {"ft_model_type": "qwen_2"})
            config = ModelFactory.get_model_cls("qwen_2")._create_config(root)
            self.assertEqual((config.hidden_size, config.vocab_size), (16, 24))


class FrontendDescriptorTest(FrontendDescriptorChecks, unittest.TestCase):
    def test_model_assistant_and_frontend_resolve_real_descriptors(self):
        self.check_descriptors()


class FrontendWheelContentsTest(FrontendDescriptorChecks, unittest.TestCase):
    def test_frontend_wheel_contains_config_but_no_compute_engine(self):
        with zipfile.ZipFile(os.environ["FRONTEND_WHEEL"]) as wheel:
            names = set(wheel.namelist())
        for path in (
            "rtp_llm/model_factory.py",
            "rtp_llm/models/qwen_v2.py",
            "rtp_llm/models_py/speculative/qwen3_dspark_checkpoint.py",
            "rtp_llm/async_decoder_engine/embedding/interface.py",
            "rtp_llm/libs/libth_transformer_config.so",
        ):
            self.assertIn(path, names)
        self.assertFalse(
            any(
                name.endswith(("/librtp_compute_ops.so", "/rtp_compute_ops.so"))
                for name in names
            )
        )
        self.assertNotIn("rtp_llm/async_decoder_engine/engine_creator.py", names)
        self.assertNotIn("rtp_llm/models_py/new_models/qwen3/language.py", names)
        # The wheel target runs only this case. Third-party dependencies come
        # from Bazel, but rtp_llm and its config DSO must come from the wheel.
        with tempfile.TemporaryDirectory() as root:
            with zipfile.ZipFile(os.environ["FRONTEND_WHEEL"]) as wheel:
                wheel.extractall(root)
            with patch.object(sys, "path", [root, *sys.path]):
                self.check_descriptors()
                from rtp_llm import model_factory

                self.assertEqual(
                    Path(model_factory.__file__).resolve(),
                    Path(root, "rtp_llm/model_factory.py").resolve(),
                )


if __name__ == "__main__":
    unittest.main()
