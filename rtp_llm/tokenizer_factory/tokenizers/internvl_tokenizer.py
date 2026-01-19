from typing import Any, Dict, List

from transformers import AutoTokenizer

from rtp_llm.tokenizer_factory.tokenizer_factory_register import register_tokenizer
from rtp_llm.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class InternVLTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    def encode(self, prompt: str, **kwargs):
        prompt_slices = prompt.split("<image>")
        new_prompt = prompt_slices[0]
        for slice in prompt_slices[1:]:
            new_prompt += "<img></img>" + slice
        return self.tokenizer.encode(new_prompt, add_special_tokens=False, **kwargs)


register_tokenizer("internvl", InternVLTokenizer)
