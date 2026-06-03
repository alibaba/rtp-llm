import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase, main

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory


class AllFakeModelTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data_path = "rtp_llm/test/tokenizer_test/testdata/qwen2_tokenizer"

    def test_simple(self):
        # test load success from bad tokenizer file
        tokenizer = Qwen2Tokenizer.from_pretrained(self.data_path)
        # test special tokens
        res = tokenizer.encode("<|im_start|>hello<|im_end|>")
        self.assertEqual(res, [151644, 14990, 151645])

    def test_qwen35_loads_tokenizers_backend_tokenizer_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_path = self._create_qwen35_tokenizer(Path(temp_dir))

            for model_type in ["qwen35_dense", "qwen35_moe", "qwen35_moe_mtp"]:
                with self.subTest(model_type=model_type):
                    tokenizer = TokenizerFactory.create(
                        str(tokenizer_path), str(tokenizer_path), model_type
                    )

                    self.assertEqual(tokenizer.encode("<|im_start|>"), [248045])
                    self.assertEqual(tokenizer.encode("<|im_end|>"), [248046])
                    self.assertEqual(tokenizer.im_start_id, 248045)
                    self.assertEqual(tokenizer.im_end_id, 248046)
                    self.assertEqual(tokenizer.stop_words_id_list, [[248046], [248045]])

    def test_qwen35_loads_chat_template_from_jinja_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chat_template = (
                "{% for message in messages %}{{'<|im_start|>' + "
                "message['role'] + '\\n' + message['content'] + '<|im_end|>' "
                "+ '\\n'}}{% endfor %}"
            )
            tokenizer_path = self._create_qwen35_tokenizer(
                Path(temp_dir), chat_template=None
            )
            (tokenizer_path / "chat_template.jinja").write_text(chat_template)

            tokenizer = TokenizerFactory.create(
                str(tokenizer_path), str(tokenizer_path), "qwen35_dense"
            )

            self.assertEqual(tokenizer.chat_template, chat_template)
            self.assertEqual(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "hello"}],
                    tokenize=False,
                    add_generation_prompt=False,
                ),
                "<|im_start|>user\nhello<|im_end|>\n",
            )

    def test_qwen35_uses_default_chat_template_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_path = self._create_qwen35_tokenizer(
                Path(temp_dir), chat_template=None
            )

            tokenizer = TokenizerFactory.create(
                str(tokenizer_path), str(tokenizer_path), "qwen35_dense"
            )

            self.assertIsNotNone(tokenizer.chat_template)
            self.assertEqual(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "hello"}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "<|im_start|>user\nhello<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n",
            )

    def test_qwen35_default_chat_template_preserves_tool_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_path = self._create_qwen35_tokenizer(
                Path(temp_dir), chat_template=None
            )

            tokenizer = TokenizerFactory.create(
                str(tokenizer_path), str(tokenizer_path), "qwen35_dense"
            )

            rendered_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "weather"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Hangzhou"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": '{"temperature": 30}',
                        "tool_call_id": "call_1",
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            )

            self.assertIn("get_weather", rendered_prompt)
            self.assertIn("<function=get_weather>", rendered_prompt)
            self.assertIn("<parameter=city>", rendered_prompt)
            self.assertIn("Hangzhou", rendered_prompt)
            self.assertIn('{"temperature": 30}', rendered_prompt)
            self.assertIn("<tool_call>", rendered_prompt)
            self.assertIn("<tool_response>", rendered_prompt)
            self.assertNotIn("None", rendered_prompt)

    def _create_qwen35_tokenizer(
        self, temp_dir: Path, chat_template: str | None = "default"
    ) -> Path:
        tokenizer_path = temp_dir / "qwen35_tokenizer"
        tokenizer_path.mkdir()

        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                self._special_token("<|endoftext|>", 248044),
                self._special_token("<|im_start|>", 248045),
                self._special_token("<|im_end|>", 248046),
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "hello": 1,
                    "<|endoftext|>": 248044,
                    "<|im_start|>": 248045,
                    "<|im_end|>": 248046,
                },
                "unk_token": "<unk>",
            },
        }
        (tokenizer_path / "tokenizer.json").write_text(json.dumps(tokenizer_json))

        if chat_template == "default":
            chat_template = (
                "{% for message in messages %}{{'<|im_start|>' + "
                "message['role'] + '\\n' + message['content'] + '<|im_end|>' "
                "+ '\\n'}}{% endfor %}"
            )

        tokenizer_config = {
            "added_tokens_decoder": {
                "248044": {
                    "content": "<|endoftext|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                },
                "248045": {
                    "content": "<|im_start|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                },
                "248046": {
                    "content": "<|im_end|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                },
            },
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            "eos_token": "<|im_end|>",
            "model_max_length": 32768,
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "TokenizersBackend",
            "unk_token": "<unk>",
        }
        if chat_template:
            tokenizer_config["chat_template"] = chat_template
        (tokenizer_path / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config)
        )
        (tokenizer_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5"})
        )
        return tokenizer_path

    def _special_token(self, content: str, token_id: int) -> dict[str, Any]:
        return {
            "id": token_id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }


if __name__ == "__main__":
    main()
