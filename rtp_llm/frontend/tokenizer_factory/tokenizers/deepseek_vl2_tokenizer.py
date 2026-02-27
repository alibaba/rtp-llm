import logging
from typing import Any, Dict, List

from pydantic import config
from transformers import AutoTokenizer

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class DeepSeekVLV2Tokenizer(BaseTokenizer):

    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        self.tokenizer.padding_side = "left"
        pad_token = "<｜▁pad▁｜>"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
        logging.info(
            f"Add pad token = ['{pad_token}'] to the tokenizer\n"
            f"{pad_token}:{self.tokenizer.encode(pad_token, add_special_tokens=False)[0]}"
        )

        image_token = "<image>"
        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            self.tokenizer.add_special_tokens(
                dict(additional_special_tokens=[image_token])
            )
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        logging.info(
            f"Add image token = ['{image_token}'] to the tokenizer\n"
            f"{image_token}:{self.tokenizer.encode(image_token, add_special_tokens=False)[0]}"
        )

        special_tokens = [
            "<|ref|>",
            "<|/ref|>",
            "<|det|>",
            "<|/det|>",
            "<|grounding|>",
            "<|User|>",
            "<|Assistant|>",
        ]
        self.tokenizer.add_special_tokens(
            dict(additional_special_tokens=special_tokens)
        )
        logging.info(
            f"Add grounding-related tokens = {special_tokens} to the tokenizer with input_ids\n"
            f"<|ref|>:{self.tokenizer.encode('<|ref|>', add_special_tokens=False)[0]}\n"
            f"<|/ref|>:{self.tokenizer.encode('<|/ref|>', add_special_tokens=False)[0]}\n"
            f"<|det|>:{self.tokenizer.encode('<|det|>', add_special_tokens=False)[0]}\n"
            f"<|/det|>:{self.tokenizer.encode('<|/det|>', add_special_tokens=False)[0]}\n"
            f"<|grounding|>:{self.tokenizer.encode('<|grounding|>', add_special_tokens=False)[0]}\n"
            f"<|User|>:{self.tokenizer.encode('<|User|>', add_special_tokens=False)[0]}\n"
            f"<|Assistant|>:{self.tokenizer.encode('<|Assistant|>', add_special_tokens=False)[0]}"
        )

    def encode(self, prompt: str, **kwargs):
        return self.tokenizer.encode(prompt, **kwargs)


register_tokenizer("deepseek_vl_v2", DeepSeekVLV2Tokenizer)
