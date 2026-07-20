import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union


class BaseTokenizer:
    def __init__(
        self, tokenizer_path: str, config_json: Optional[Dict[str, Any]] = None
    ):
        self.path = tokenizer_path
        self.config_json = config_json or {}
        self.init_tokenizer(tokenizer_path, self.config_json)

    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any]):
        from transformers import AutoTokenizer

        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        tokenizer_obj = None
        if os.path.exists(tokenizer_json_path):
            from tokenizers import Tokenizer as TokenizerFast

            tokenizer_obj = TokenizerFast.from_file(tokenizer_json_path)

        tokenizer_config = self._load_tokenizer_config(tokenizer_path)
        extra_kwargs = self._transformers_v5_kwargs(tokenizer_config, tokenizer_obj)
        extra_kwargs.update(self._additional_kwargs(tokenizer_config))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                verbose=False,
                use_fast=True,
                **extra_kwargs,
            )
        except Exception as e:
            logging.error(
                f"AutoTokenizer.from_pretrained failed for tokenizer_path={tokenizer_path}, "
                f"extra_kwargs={extra_kwargs}: {e}"
            )
            raise
        self._fix_post_processor(tokenizer_obj, extra_kwargs)

    def _additional_kwargs(self, tokenizer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses to inject extra kwargs before from_pretrained."""
        return {}

    @staticmethod
    def _load_tokenizer_config(tokenizer_path: str) -> Dict[str, Any]:
        """Parse tokenizer_config.json once (returns {} if absent)."""
        config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"failed to parse {config_path}: {e}") from e

    @staticmethod
    def _transformers_v5_kwargs(
        tokenizer_config: Dict[str, Any], tokenizer_obj=None
    ) -> Dict[str, Any]:
        """Workaround for transformers==5.2.0 from_pretrained regressions.

        Transformers 5.2.0 rewrote tokenizer loading via TokenizersBackend. Two issues
        require explicit kwargs to preserve correct behavior:

        1. add_eos_token / add_bos_token (found on gte-Qwen2-7B-instruct and
           DeepSeek-V2-Lite-Chat respectively):
           from_pretrained no longer passes these from tokenizer_config.json to
           custom tokenizer __init__. The custom class falls back to its default
           (False), so BOS/EOS is not appended during encode — breaking embedding
           models (last-token pooling) and chat models (missing BOS changes output).
           Fix: explicitly pass add_eos_token/add_bos_token from tokenizer_config.json.
           NOTE: upstream main (fd6bc380c8) intentionally pops these when
           tokenizer.json exists, expecting post_processor to handle it — but models
           like gte-Qwen2 have no EOS in post_processor. This workaround is needed
           long-term unless the model's tokenizer.json is updated.

        2. tokenizer_object (affects models with tokenizer_class: LlamaTokenizerFast,
           e.g. DeepSeek-R1 series):
           Class-specific __init__ (LlamaTokenizer) unconditionally rebuilds the
           internal _tokenizer with Metaspace pre_tokenizer, overriding what
           tokenizer.json defines (e.g. regex Split). This causes whitespace/newlines
           to be silently dropped during encode ("\\n\\n" -> []).
           Fix: pass tokenizer_object loaded directly from tokenizer.json.
           TokenizersBackend.__init__ uses this object to overwrite the class-built
           _tokenizer, preserving the correct pre_tokenizer/decoder from the file.
           NOTE: upstream tracks this as huggingface/transformers#45488.
           Our fix is model-agnostic and stable — keep until upstream is reliable.
        """
        kwargs: Dict[str, Any] = {}
        if "add_eos_token" in tokenizer_config:
            kwargs["add_eos_token"] = tokenizer_config["add_eos_token"]
        if "add_bos_token" in tokenizer_config:
            kwargs["add_bos_token"] = tokenizer_config["add_bos_token"]

        if tokenizer_obj is not None:
            kwargs["tokenizer_object"] = tokenizer_obj
        return kwargs

    def _fix_post_processor(self, tokenizer_obj, extra_kwargs):
        """Workaround for transformers==5.2.0 post_processor override.

        Transformers 5.2.0 tokenizer classes (e.g. XLMRobertaTokenizer) overwrite
        the post_processor in __init__ with a hardcoded template AFTER super().__init__()
        has already restored the correct one from tokenizer_object. This means
        tokenizer.json's post_processor (e.g. double </s></s> for RoBERTa pair inputs)
        is lost and replaced by the class's default (single </s>).

        Fix: two-phase restore:
        1. Unconditionally restore post_processor from tokenizer.json (undoes class
           __init__ corruption).
        2. If add_eos_token/add_bos_token was passed in extra_kwargs, call
           update_post_processor() to re-inject BOS/EOS via transformers' standard
           mechanism. This rebuilds a TemplateProcessing that includes the special tokens.
        """
        import transformers
        from packaging import version

        if version.parse(transformers.__version__).major < 5:
            return
        if tokenizer_obj is None:
            return
        if not hasattr(self.tokenizer, "_tokenizer"):
            # slow tokenizer, nothing to restore.
            return
        if tokenizer_obj.post_processor is not None:
            self.tokenizer._tokenizer.post_processor = tokenizer_obj.post_processor
        # If tokenizer.json's post_processor disagrees with tokenizer_config's add_eos/bos_token,
        # tokenizer.json wins: rebuild below only when add_eos/bos_token is explicitly True.
        if (
            extra_kwargs.get("add_eos_token") or extra_kwargs.get("add_bos_token")
        ) and hasattr(self.tokenizer, "update_post_processor"):
            self.tokenizer.update_post_processor()

    def encode(self, prompt: str, **kwargs):
        return self.tokenizer.encode(prompt, **kwargs)

    def decode(self, token_id: Union[int, List[int]], **kwargs):
        if isinstance(token_id, List) and len(token_id) == 0:
            return ""
        return self.tokenizer.decode(token_id, **kwargs)

    def batch_decode(self, token_ids: Union[List[int], List[List[int]]], **kwargs):
        return [
            self.tokenizer._decode(
                seq,
                **kwargs,
            )
            for seq in token_ids
        ]

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
        try:
            return self.tokenizer.additional_special_tokens
        except AttributeError:
            return getattr(self.tokenizer, "extra_special_tokens", [])

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

    @functools.cache
    def __len__(self) -> int:
        return self.tokenizer.__len__()
