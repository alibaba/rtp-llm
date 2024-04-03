from typing import Any
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer as Qwen2TokenizerOrigin

class Qwen2Tokenizer(Qwen2TokenizerOrigin):
    def __init__(self, *args: Any, **kwargs: Any):        
        super().__init__(*args, **kwargs)