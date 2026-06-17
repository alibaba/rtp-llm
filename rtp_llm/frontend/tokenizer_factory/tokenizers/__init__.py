import importlib
from typing import Any, Dict

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    ensure_all_tokenizers_registered,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer

_CLASS_TO_MODULE: Dict[str, str] = {
    "BertTokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.bert_tokenizer",
    "ChatGLMV2Tokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.chatglm_tokenizer",
    "ChatGLMV3Tokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.chatglm_tokenizer",
    "ChatGLMV4Tokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.chatglm_tokenizer",
    "ChatGLMV5Tokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.chatglm_tokenizer",
    "DeepSeekVLV2Tokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.deepseek_vl2_tokenizer",
    "LlamaTokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.llama_tokenizer",
    "LlavaTokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.llava_tokenizer",
    "QWenTokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.qwen_tokenizer",
    "QWenV2Tokenizer": "rtp_llm.frontend.tokenizer_factory.tokenizers.qwen_tokenizer",
}

__all__ = ["BaseTokenizer", "load_all_tokenizers"] + sorted(_CLASS_TO_MODULE)


def load_all_tokenizers() -> None:
    ensure_all_tokenizers_registered()


def __getattr__(name: str) -> Any:
    module_path = _CLASS_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
