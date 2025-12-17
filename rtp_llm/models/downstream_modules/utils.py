import logging
from typing import Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules import (
    ALLEmbeddingModule,
    BgeM3EmbeddingModule,
    ClassifierModule,
    ColBertEmbeddingModule,
    DenseEmbeddingModule,
    RerankerModule,
    SparseEmbeddingModule,
)
from rtp_llm.models.downstream_modules.reranker.qwen3_reranker import (
    Qwen3RerankerModule,
)
from rtp_llm.ops import TaskType


def create_custom_module(
    config: ModelConfig,
    tokenizer: Optional[BaseTokenizer],
):
    # try import internal module
    try:
        from internal_source.rtp_llm.models.downstream_modules.utils import (
            create_custom_module,
        )

        internal_module = create_custom_module(config, tokenizer)
        if internal_module is not None:
            return internal_module
    except ImportError:
        logging.exception("internal module not found, using external module")

    task_type = config.task_type
    if task_type == TaskType.LANGUAGE_MODEL:
        return None
    model_type = config.model_type
    assert tokenizer is not None, "tokenizer should not be None"
    if task_type == TaskType.DENSE_EMBEDDING:
        return DenseEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.ALL_EMBEDDING:
        return ALLEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.SPARSE_EMBEDDING:
        return SparseEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.COLBERT_EMBEDDING:
        return ColBertEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.SEQ_CLASSIFICATION:
        return ClassifierModule(config, tokenizer)
    elif task_type == TaskType.BGE_M3:
        return BgeM3EmbeddingModule(config, tokenizer)
    elif model_type == "qwen_3":
        return Qwen3RerankerModule(config, tokenizer)
    elif task_type == TaskType.RERANKER:
        return RerankerModule(config, tokenizer)
    raise Exception(f"unknown task_type: {task_type}")
