import json
import os
from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer as QwenTokenizerOrigin,
)

_QWEN35_DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for item in message['content'] %}"
    "{% if item is mapping and 'text' in item %}"
    "{{ item['text'] }}"
    "{% elif item is mapping and item.get('type') == 'text' %}"
    "{{ item.get('text', '') }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{'<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)


class QWenTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = QwenTokenizerOrigin.from_pretrained(tokenizer_path)
        self.tokenizer.decoder.update(
            {v: k for k, v in self.tokenizer.special_tokens.items()}
        )

    @property
    def im_start_id(self):
        return self.tokenizer.im_start_id

    @property
    def im_end_id(self):
        return self.tokenizer.im_end_id

    @property
    def stop_words_id_list(self):
        return [[self.im_end_id], [self.im_start_id]]


class QWenV2Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, verbose=False, trust_remote_code=True
            )
        except ValueError as e:
            if "TokenizersBackend" not in str(e):
                raise
            self.tokenizer = self._load_fast_tokenizer_from_tokenizer_json(
                tokenizer_path, config_json
            )
        self.tokenizer.im_start_id = self.tokenizer.encode("<|im_start|>")[0]
        self.tokenizer.im_end_id = self.tokenizer.encode("<|im_end|>")[0]

    def _load_fast_tokenizer_from_tokenizer_json(
        self, tokenizer_path: str, config_json: Dict[str, Any]
    ):
        tokenizer_config = {}
        tokenizer_config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as reader:
                tokenizer_config = json.load(reader)

        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        tokenizer_kwargs = {
            key: tokenizer_config[key]
            for key in [
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
                "additional_special_tokens",
                "model_max_length",
            ]
            if key in tokenizer_config
        }
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file, **tokenizer_kwargs
        )
        chat_template = self._load_chat_template(
            tokenizer_path, tokenizer_config, config_json
        )
        if chat_template:
            tokenizer.chat_template = chat_template
        return tokenizer

    def _load_chat_template(
        self,
        tokenizer_path: str,
        tokenizer_config: Dict[str, Any],
        config_json: Dict[str, Any],
    ):
        if tokenizer_config.get("chat_template"):
            return tokenizer_config["chat_template"]

        chat_template_path = os.path.join(tokenizer_path, "chat_template.jinja")
        if os.path.exists(chat_template_path):
            with open(chat_template_path, "r", encoding="utf-8") as reader:
                return reader.read()

        if self._is_qwen35_config(config_json, tokenizer_path):
            return _QWEN35_DEFAULT_CHAT_TEMPLATE
        return None

    def _is_qwen35_config(self, config_json: Dict[str, Any], tokenizer_path: str):
        text_config = config_json.get("text_config", {})
        model_type = str(
            config_json.get("model_type") or text_config.get("model_type") or ""
        )
        if model_type.startswith("qwen3_5"):
            return True
        lower_path = tokenizer_path.lower()
        return any(name in lower_path for name in ["qwen35", "qwen3.5", "qwen3.6"])

    @property
    def im_start_id(self):
        return self.tokenizer.im_start_id

    @property
    def im_end_id(self):
        return self.tokenizer.im_end_id

    @property
    def stop_words_id_list(self):
        return [[self.im_end_id], [self.im_start_id]]


register_tokenizer(["qwen", "qwen_7b", "qwen_13b", "qwen_1b8"], QWenTokenizer)
register_tokenizer(
    [
        "qwen_2",
        "qwen_agent",
        "qwen_2_embedding",
        "qwen_tool",
        "qwen_2-mtp",
        "qwen_3",
        "qwen_3_tool",
        "qwen35_dense",
        "qwen35_moe",
    ],
    QWenV2Tokenizer,
)
