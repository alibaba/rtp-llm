import logging
import platform

from .base_tokenizer import BaseTokenizer
from .bert_tokenizer import BertTokenizer
from .chatglm_tokenizer import (
    ChatGLMV2Tokenizer,
    ChatGLMV3Tokenizer,
    ChatGLMV4Tokenizer,
)
from .llama_tokenizer import LlamaTokenizer
from .llava_tokenizer import LlavaTokenizer
from .qwen_tokenizer import QWenTokenizer, QWenV2Tokenizer
from .starcoder_tokenizer import StarcoderTokenizer

try:
    from internal_source.rtp_llm.tokenizers import internal_init
except ImportError as e:
    logging.info(f"import internal source failed, error: {str(e)}")
    pass
