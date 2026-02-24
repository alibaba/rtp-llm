from typing import Any, Dict, List

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_chatglm2 import (
    ChatGLMTokenizer as ChatGLMV2TokenizerHf,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_chatglm3 import (
    ChatGLMTokenizer as ChatGLMV3TokenizerHf,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_chatglm4 import (
    ChatGLM4Tokenizer as ChatGLMV4TokenizerHf,
)


class ChatGLMV2Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = ChatGLMV2TokenizerHf.from_pretrained(tokenizer_path)


class ChatGLMV3Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = ChatGLMV3TokenizerHf.from_pretrained(
            tokenizer_path, encode_special_tokens=True
        )


class ChatGLMV4Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = ChatGLMV4TokenizerHf.from_pretrained(tokenizer_path)


class ChatGLMV5Tokenizer(BaseTokenizer):
    """
    GLM-5 specific tokenizer to handle invalid tokenizer_class in config.

    Problem: GLM-5's tokenizer_config.json contains "tokenizer_class": "TokenizersBackend"
             which doesn't exist in transformers library, and has format issues with
             extra_special_tokens that causes AttributeError.

    Solution:
    1. Load tokenizer.json directly using tokenizers library
    2. Wrap it with PreTrainedTokenizerFast manually
    3. Avoid reading tokenizer_config.json entirely

    This works because:
    - tokenizer.json contains all vocabulary and encoding rules
    - We bypass all config parsing that causes issues
    - No dependency on the problematic tokenizer_config.json
    """

    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        import os

        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        # Load tokenizer directly from tokenizer.json, bypassing tokenizer_config.json
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        tokenizer = Tokenizer.from_file(tokenizer_file)

        # Wrap it with PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            unk_token="<|endoftext|>",
        )


register_tokenizer(["chatglm2", "chat_glm_2"], ChatGLMV2Tokenizer)
register_tokenizer(["chatglm3", "chat_glm_3"], ChatGLMV3Tokenizer)
register_tokenizer(["chatglm4", "chatglm4v"], ChatGLMV4Tokenizer)
register_tokenizer(["glm_5"], ChatGLMV5Tokenizer)
