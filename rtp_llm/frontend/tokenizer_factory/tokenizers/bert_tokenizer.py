import logging
from typing import Any, Dict, List

from transformers import AutoTokenizer
from transformers import BertTokenizer as BertTokenizerHf

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class BertTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                use_fast=self.py_env_configs.load_config.use_fast_tokenizer,
            )
        except:
            logging.warning(
                "failed to load bert tokenizer using AutoTokenizer, try using BertTokenizer instead"
            )
            self.tokenizer = BertTokenizerHf.from_pretrained(tokenizer_path)

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def unk_token_id(self):
        return self.tokenizer.unk_token_id


register_tokenizer(["bert", "roberta", "vision_bert"], BertTokenizer)
