import importlib
import logging
import platform
import sys

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
    except (ImportError, ModuleNotFoundError):
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
    sys.modules[module_path] = mod


_apply_transformers_v5_2_0_compat()

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

if platform.processor() != "aarch64":
    from .internvl_tokenizer import InternVLTokenizer
    from .minicpmv_embedding_tokenizer import MiniCPMVEmbeddingTokenizer

from rtp_llm.utils.import_util import has_internal_source

if has_internal_source():
    import internal_source.rtp_llm.tokenizers.internal_init
