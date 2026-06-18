import platform

from .base_tokenizer import BaseTokenizer
from .bert_tokenizer import BertTokenizer
from .chatglm_tokenizer import (
    ChatGLMV2Tokenizer,
    ChatGLMV3Tokenizer,
    ChatGLMV4Tokenizer,
    ChatGLMV5Tokenizer,
)
from .deepseek_vl2_tokenizer import DeepSeekVLV2Tokenizer
from .llama_tokenizer import LlamaTokenizer
from .llava_tokenizer import LlavaTokenizer
from .qwen_tokenizer import QWenTokenizer, QWenV2Tokenizer

# from .starcoder_tokenizer import StarcoderTokenizer

from rtp_llm.utils.import_util import has_internal_source

if has_internal_source():
    # Phase-25 namespace merge: rtp_llm.tokenizers resolves to
    # internal_source/rtp_llm/tokenizers via the extended __path__ (no OSS counterpart).
    from rtp_llm.tokenizers import internal_init  # noqa: F401
