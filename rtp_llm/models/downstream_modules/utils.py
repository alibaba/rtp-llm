import logging
from typing import Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.ops import TaskType


def create_custom_module(
    config: ModelConfig,
    tokenizer: Optional[BaseTokenizer],
):
    task_type = config.task_type
    if task_type == TaskType.LANGUAGE_MODEL:
        return None

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

    model_type = config.model_type
    assert tokenizer is not None, "tokenizer should not be None"
    if task_type == TaskType.DENSE_EMBEDDING:
        from rtp_llm.models.downstream_modules.embedding.dense_embedding_module import (
            DenseEmbeddingModule,
        )

        return DenseEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.ALL_EMBEDDING:
        from rtp_llm.models.downstream_modules.embedding.all_embedding_module import (
            ALLEmbeddingModule,
        )

        return ALLEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.SPARSE_EMBEDDING:
        from rtp_llm.models.downstream_modules.embedding.sparse_emebdding_module import (
            SparseEmbeddingModule,
        )

        return SparseEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.COLBERT_EMBEDDING:
        from rtp_llm.models.downstream_modules.embedding.colbert_embedding_module import (
            ColBertEmbeddingModule,
        )

        return ColBertEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.SEQ_CLASSIFICATION:
        from rtp_llm.models.downstream_modules.classifier.classifier import (
            ClassifierModule,
        )

        return ClassifierModule(config, tokenizer)
    elif task_type == TaskType.BGE_M3:
        from rtp_llm.models.downstream_modules.embedding.bge_m3_embedding_module import (
            BgeM3EmbeddingModule,
        )

        return BgeM3EmbeddingModule(config, tokenizer)
    elif model_type == "qwen_3":
        from rtp_llm.models.downstream_modules.reranker.qwen3_reranker import (
            Qwen3RerankerModule,
        )

        return Qwen3RerankerModule(config, tokenizer)
    elif task_type == TaskType.RERANKER:
        from rtp_llm.models.downstream_modules.reranker.reranker_module import (
            RerankerModule,
        )

        return RerankerModule(config, tokenizer)
    raise Exception(f"unknown task_type: {task_type}")
