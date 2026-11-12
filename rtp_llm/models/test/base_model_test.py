import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.models.base_model import BaseModel


class BaseModelGenerateConfigTest(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel.__new__(BaseModel)
        self.model.default_generate_config = GenerateConfig()

    def load_checkpoint_config(self, config):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "generation_config.json").write_text(
                json.dumps(config)
            )
            self.model.load_default_generate_config(
                SimpleNamespace(generation_config_path=directory)
            )

    def test_non_nullable_checkpoint_null_keeps_defaults(self):
        fields = (
            "max_new_tokens",
            "min_new_tokens",
            "top_k",
            "top_p",
            "temperature",
            "repetition_penalty",
            "num_beams",
            "num_return_sequences",
            "do_sample",
            "stop_words_str",
            "stop_words_list",
            "aux_info",
            "top_logprobs",
        )
        before = self.model.default_generate_config.model_dump()
        self.load_checkpoint_config({name: None for name in fields})
        self.assertEqual(
            self.model.default_generate_config.model_dump(), before
        )

    def test_nullable_checkpoint_null_is_preserved(self):
        config = self.model.default_generate_config
        config.update({"random_seed": 42, "eos_token_id": 7})
        fields = (
            "random_seed",
            "eos_token_id",
            "timeout_ms",
            "traffic_reject_priority",
        )
        self.load_checkpoint_config({name: None for name in fields})
        for name in fields:
            with self.subTest(field=name):
                self.assertIsNone(getattr(config, name))
                self.assertIn(name, config.model_fields_set)

    def test_valid_values_and_unknown_checkpoint_metadata(self):
        self.load_checkpoint_config(
            {
                "max_new_tokens": 0,
                "capture_hidden_states": True,
                "temperature": 0.5,
                "do_sample": False,
                "transformers_version": "4.51.0",
                "_from_model_config": None,
            }
        )
        config = self.model.default_generate_config
        self.assertTrue(config.is_prefill_only())
        self.assertTrue(config.capture_hidden_states)
        self.assertEqual(config.temperature, 0.5)
        self.assertFalse(config.do_sample)
        self.assertFalse(hasattr(config, "transformers_version"))

    def test_invalid_checkpoint_values_are_not_swallowed(self):
        for updates, field in (
            ({"temperature": {}}, "temperature"),
            ({"do_sample": []}, "do_sample"),
            ({"top_logprobs": 2.0}, "top_logprobs"),
            ({"random_seed": {}}, "random_seed"),
            ({"max_new_tokens": -1}, "max_new_tokens"),
            ({"max_new_tokens": 0, "min_new_tokens": 1}, "min_new_tokens"),
        ):
            with self.subTest(updates=updates):
                before = self.model.default_generate_config.model_dump()
                with self.assertRaisesRegex(FtRuntimeException, field) as ctx:
                    self.load_checkpoint_config(updates)
                self.assertEqual(
                    ctx.exception.exception_type,
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                )
                self.assertEqual(
                    self.model.default_generate_config.model_dump(), before
                )

    def test_checkpoint_null_policy_does_not_relax_request_updates(self):
        for method in ("update", "update_and_pop"):
            for updates in (
                {"temperature": None},
                {"do_sample": None},
                {"min_new_tokens": None},
                {"max_new_tokens": None},
                {"top_logprobs": 2.0},
                {"temperature": {}},
                {"max_new_tokens": -1},
            ):
                with self.subTest(method=method, updates=updates):
                    request = dict(updates)
                    before = self.model.default_generate_config.model_dump()
                    with self.assertRaises(FtRuntimeException):
                        getattr(self.model.default_generate_config, method)(
                            request
                        )
                    self.assertEqual(request, updates)
                    self.assertEqual(
                        self.model.default_generate_config.model_dump(), before
                    )

    def test_absent_generation_path_is_noop(self):
        self.model.load_default_generate_config(None)
        self.model.load_default_generate_config(
            SimpleNamespace(generation_config_path=None)
        )
        self.assertEqual(self.model.default_generate_config, GenerateConfig())


if __name__ == "__main__":
    unittest.main()
