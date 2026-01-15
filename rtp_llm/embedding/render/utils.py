import logging
from typing import Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.all_embedding_renderer import ALLEmbeddingRenderer
from rtp_llm.embedding.render.bge_m3_renderer import BgeM3Renderer
from rtp_llm.embedding.render.classifier_renderer import ClassifierRenderer
from rtp_llm.embedding.render.colbert_embedding_renderer import ColbertEmbeddingRenderer
from rtp_llm.embedding.render.dense_embedding_renderer import DenseEmbeddingRenderer
from rtp_llm.embedding.render.qwen3_reranker_renderer import Qwen3RerankerRenderer
from rtp_llm.embedding.render.reranker_renderer import RerankerRenderer
from rtp_llm.embedding.render.sparse_embedding_renderer import SparseEmbeddingRenderer
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.ops import TaskType


def create_custom_render(
    config: ModelConfig,
    tokenizer: Optional[BaseTokenizer],
):
    # try import internal module
    try:
        from internal_source.rtp_llm.embedding.utils import create_custom_render

        internal_renderer = create_custom_render(config, tokenizer)
        if internal_renderer is not None:
            return internal_renderer
    except ImportError:
        logging.exception("internal module not found, using external module")

    task_type = config.task_type
    if task_type == TaskType.LANGUAGE_MODEL:
        return None
    model_type = config.model_type
    assert tokenizer is not None, "tokenizer should not be None"
    if task_type == TaskType.DENSE_EMBEDDING:
        return DenseEmbeddingRenderer(config, tokenizer)
    elif task_type == TaskType.ALL_EMBEDDING:
        return ALLEmbeddingRenderer(config, tokenizer)
    elif task_type == TaskType.SPARSE_EMBEDDING:
        return SparseEmbeddingRenderer(config, tokenizer)
    elif task_type == TaskType.COLBERT_EMBEDDING:
        return ColbertEmbeddingRenderer(config, tokenizer)
    elif task_type == TaskType.SEQ_CLASSIFICATION:
        return ClassifierRenderer(config, tokenizer)
    elif task_type == TaskType.BGE_M3:
        return BgeM3Renderer(config, tokenizer)
    elif model_type == "qwen_3":
        return Qwen3RerankerRenderer(config, tokenizer)
    elif task_type == TaskType.RERANKER:
        return RerankerRenderer(config, tokenizer)
    raise Exception(f"unknown task_type: {task_type}")
