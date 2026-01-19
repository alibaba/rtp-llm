from typing import Any, Dict, List

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

from rtp_llm.tokenizer_factory.tokenizer_factory_register import register_tokenizer
from rtp_llm.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class StarcoderTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)


register_tokenizer(["gpt_bigcode", "wizardcoder", "starcoder2"], StarcoderTokenizer)
