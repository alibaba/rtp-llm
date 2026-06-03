from typing import Any, Dict, List

from transformers import AutoTokenizer

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class StarcoderTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


register_tokenizer(["gpt_bigcode", "wizardcoder", "starcoder2"], StarcoderTokenizer)
