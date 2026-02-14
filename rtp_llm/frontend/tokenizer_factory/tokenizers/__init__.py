import platform

from .base_tokenizer import BaseTokenizer
from .bert_tokenizer import BertTokenizer
from .chatglm_tokenizer import (
    ChatGLMV2Tokenizer,
    ChatGLMV3Tokenizer,
    ChatGLMV4Tokenizer,
    ChatGLMV5Tokenizer,
)
from .llama_tokenizer import LlamaTokenizer
from .llava_tokenizer import LlavaTokenizer
from .qwen_tokenizer import QWenTokenizer, QWenV2Tokenizer

# from .starcoder_tokenizer import StarcoderTokenizer

if platform.processor() != "aarch64":
    from .internvl_tokenizer import InternVLTokenizer
    from .minicpmv_embedding_tokenizer import MiniCPMVEmbeddingTokenizer

from rtp_llm.utils.import_util import has_internal_source

if has_internal_source():
    import internal_source.rtp_llm.tokenizers.internal_init
