import os
from typing import Any, Dict, List

from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import (
    LlamaTokenizer as LlamaTokenizerOrigin,
)

from rtp_llm.tokenizer_factory.tokenizer_factory_register import register_tokenizer
from rtp_llm.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


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
