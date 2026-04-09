import os
from typing import Any, Dict, List

from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import (
    LlamaTokenizer as LlamaTokenizerOrigin,
)

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class LlamaTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        tokenizer_config_file = os.path.join(tokenizer_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_file):
            super().init_tokenizer(tokenizer_path, config_json)
        else:
            self.tokenizer = LlamaTokenizerOrigin.from_pretrained(tokenizer_path)


register_tokenizer(
    [
        "llama",
        "internlm",
        "internlm2",
        "xverse",
        "aquila",
        "mistral",
        "baichuan",
        "baichuan2",
        "gemma",
        "cohere",
    ],
    LlamaTokenizer,
)

class Gemma4Tokenizer(BaseTokenizer):
    """Gemma4 uses tokenizer.json only — no sentencepiece .model file.
    Must bypass AutoTokenizer/GemmaTokenizerFast which try to load slow tokenizer.
    Also bypass extra_special_tokens incompatibility with transformers 4.51.
    """
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        import json
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")

        # Auto-download if tokenizer.json missing and HF_ENDPOINT is set
        if not os.path.exists(tokenizer_json):
            hf_endpoint = os.environ.get("HF_ENDPOINT")
            if hf_endpoint:
                import logging
                logging.info(f"tokenizer.json not found at {tokenizer_path}, downloading...")
                from huggingface_hub import snapshot_download
                snapshot_download("google/gemma-4-31B-it", local_dir=tokenizer_path)

        backend = Tokenizer.from_file(tokenizer_json)

        # Read special tokens from tokenizer_config.json
        kwargs = {}
        tokenizer_config = os.path.join(tokenizer_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config):
            with open(tokenizer_config) as f:
                tc = json.load(f)
            for key in ["bos_token", "eos_token", "pad_token", "unk_token"]:
                val = tc.get(key)
                if isinstance(val, dict):
                    kwargs[key] = val.get("content", "")
                elif isinstance(val, str):
                    kwargs[key] = val

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, **kwargs
        )


register_tokenizer("gemma4", Gemma4Tokenizer)
