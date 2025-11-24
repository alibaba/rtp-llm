from typing import Any, Dict, List, Union

from transformers import AutoTokenizer

from rtp_llm.config.py_config_modules import PyEnvConfigs


class BaseTokenizer:
    def __init__(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.py_env_configs = PyEnvConfigs()
        self.py_env_configs.update_from_env()
        self.path = tokenizer_path
        self.config_json = config_json
        self.init_tokenizer(tokenizer_path, self.config_json)

    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any]):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, verbose=False, use_fast=True
        )

    def encode(self, prompt: str, **kwargs):
        return self.tokenizer.encode(prompt, **kwargs)

    def decode(self, token_id: Union[int, List[int]], **kwargs):
        if isinstance(token_id, List) and len(token_id) == 0:
            return ""
        return self.tokenizer.decode(token_id, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    @property
    def stop_words_id_list(self):
        return []

    @property
    def stop_words_str_list(self):
        return []

    @property
    def chat_template(self):
        return self.tokenizer.chat_template

    @property
    def default_chat_template(self):
        return self.tokenizer.default_chat_template

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        if self.tokenizer.eos_token_id is None:
            return self.config_json.get("eos_token_id") or 0
        else:
            return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token(self):
        return self.tokenizer.bos_token

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    def get_real_tokenizer(self):
        return self.tokenizer

    def tokenize(self, text: str, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)

    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    @property
    def _added_tokens_encoder(self):
        return self.tokenizer._added_tokens_encoder

    @property
    def vocab_size(self):
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        else:
            return self.config_json.get("vocab_size", 0)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: List[str]):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)

    @property
    def is_fast(self):
        return self.tokenizer.is_fast

    def get_added_vocab(self):
        return self.tokenizer.get_added_vocab()

    @property
    def model_max_length(self):
        return self.tokenizer.model_max_length

    @property
    def special_tokens_map(self):
        return self.tokenizer.special_tokens_map

    def save_pretrained(self, save_directory, **kwargs):
        return self.tokenizer.save_pretrained(save_directory, **kwargs)

    @property
    def additional_special_tokens(self):
        return self.tokenizer.additional_special_tokens

    def add_special_tokens(
        self,
        special_tokens_dict: Dict[str, Any],
        replace_additional_special_tokens: bool = True,
    ):
        return self.tokenizer.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens
        )

    def __str__(self) -> str:
        return self.tokenizer.__str__()

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)
