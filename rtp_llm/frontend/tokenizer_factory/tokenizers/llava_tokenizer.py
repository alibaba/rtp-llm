from typing import Any, Dict, List

from transformers import AutoTokenizer

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class LlavaTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=self.py_env_configs.load_config.use_fast_tokenizer
        )
        self.mm_use_im_patch_token = config_json.get("mm_use_im_patch_token", False)
        self.mm_use_im_start_end = config_json.get("mm_use_im_start_end", False)

        extra_tokens: List[str] = []
        if self.mm_use_im_patch_token:
            extra_tokens.extend(["<im_patch>"])
        if self.mm_use_im_start_end:
            extra_tokens.extend(["<im_start>", "<im_end>"])
        self.tokenizer.add_tokens(extra_tokens, special_tokens=True)

        self.image_token_index: int = -200
        self.ignore_token_index: int = -100
        self.default_image_token = "<image>"
        self.default_im_start_token = "<im_start>"
        self.default_im_end_token = "<im_end>"
        self.bos_id = 1

    def encode(self, prompt: str, **kwargs) -> List[int]:
        s = prompt
        replace_token = self.default_image_token
        if self.mm_use_im_start_end:
            replace_token = (
                self.default_im_start_token + replace_token + self.default_im_end_token
            )
        s = s.replace(self.default_image_token, replace_token)

        prompt_chunks: List[List[int]] = [
            self.tokenizer.encode(chunk) for chunk in s.split(self.default_image_token)
        ]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        t: List[int] = []
        offset = 0
        if (
            len(prompt_chunks) > 0
            and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == self.bos_id
        ):
            offset = 1
            t.append(prompt_chunks[0][0])

        for x in insert_separator(
            prompt_chunks, [self.image_token_index] * (offset + 1)
        ):
            t.extend(x[offset:])

        return t


register_tokenizer("llava", LlavaTokenizer)
