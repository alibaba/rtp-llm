import os
from typing import Any, Dict, List

from transformers import AutoTokenizer, LlamaTokenizer

from rtp_llm.tokenizer_factory.tokenizer_factory_register import register_tokenizer
from rtp_llm.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class MiniCPMVEmbeddingTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.point_start = "<point>"
        self.point_end = "</point>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"

    @property
    def eos_id(self):
        return self.tokenizer.sp_model.eos_id()

    @property
    def bos_id(self):
        return self.tokenizer.sp_model.bos_id()

    @property
    def unk_id(self):
        return self.tokenizer.sp_model.unk_id()

    @property
    def im_start_id(self):
        return self.tokenizer._convert_token_to_id(self.im_start)

    @property
    def im_end_id(self):
        return self.tokenizer._convert_token_to_id(self.im_end)


register_tokenizer("minicpmv_embedding", MiniCPMVEmbeddingTokenizer)
