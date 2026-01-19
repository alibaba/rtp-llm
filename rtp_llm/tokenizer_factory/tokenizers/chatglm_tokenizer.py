from typing import Any, Dict, List

from rtp_llm.tokenizer_factory.tokenizer_factory_register import register_tokenizer
from rtp_llm.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.tokenizer_factory.tokenizers.tokenization_chatglm2 import (
    ChatGLMTokenizer as ChatGLMV2TokenizerHf,
)
from rtp_llm.tokenizer_factory.tokenizers.tokenization_chatglm3 import (
    ChatGLMTokenizer as ChatGLMV3TokenizerHf,
)
from rtp_llm.tokenizer_factory.tokenizers.tokenization_chatglm4 import (
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


register_tokenizer(["chatglm2", "chat_glm_2"], ChatGLMV2Tokenizer)
register_tokenizer(["chatglm3", "chat_glm_3"], ChatGLMV3Tokenizer)
register_tokenizer(["chatglm4", "chatglm4v"], ChatGLMV4Tokenizer)
