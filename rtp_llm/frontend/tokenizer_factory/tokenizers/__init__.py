import importlib
import logging
import platform
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _apply_transformers_v5_2_0_compat():
    """Shim for transformers 5.x: tokenization_qwen2_fast.py was deleted,
    but model custom code still imports from it. Register the old path in sys.modules.

    Scope: currently only handles Qwen2 (tokenization_qwen2_fast). This is the only
    model whose custom code imports a deleted *_fast module as confirmed by
    test_transformers_imports.py scanning all source + model checkpoint directories.
    If other models hit similar issues, extend this function with additional shims.
    """
    module_path = "transformers.models.qwen2.tokenization_qwen2_fast"
    if module_path in sys.modules:
        return
    try:
        importlib.import_module(module_path)
        return
    except ImportError:
        pass
    fallback_path = "transformers.models.qwen2.tokenization_qwen2"
    try:
        mod = importlib.import_module(fallback_path)
    except ImportError as e:
        logger.warning(
            "transformers compat shim: cannot import %s or fallback %s: %s. "
            "This may indicate a broken transformers installation.",
            module_path,
            fallback_path,
            e,
        )
        return
    if not hasattr(mod, "Qwen2TokenizerFast"):
        qwen2_tokenizer = getattr(mod, "Qwen2Tokenizer", None)
        if qwen2_tokenizer is not None:
            mod.Qwen2TokenizerFast = qwen2_tokenizer
            logger.info(
                "transformers compat shim: aliased Qwen2Tokenizer as Qwen2TokenizerFast"
            )
        else:
            logger.warning(
                "transformers compat shim: %s has neither Qwen2TokenizerFast nor "
                "Qwen2Tokenizer — Qwen2 models may fail to load.",
                fallback_path,
            )
            return
    sys.modules[module_path] = mod


# Applied at import time (deliberate module-level side effect): the shim registers
# the deleted `transformers.models.qwen2.tokenization_qwen2_fast` module in sys.modules,
# and this MUST be in place before any AutoTokenizer.from_pretrained() runs a model's
# custom code that imports that path. Doing it here makes it safe-by-construction —
# as soon as this package is imported the shim is active — rather than relying on a
# lazy call ordering that a future caller could bypass.
_apply_transformers_v5_2_0_compat()


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
