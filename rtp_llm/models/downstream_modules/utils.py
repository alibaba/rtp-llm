import importlib
import importlib.util
import logging
import os
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
        return create_post_layers_module(config, tokenizer)
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


def create_post_layers_module(
    config: ModelConfig,
    tokenizer: Optional[BaseTokenizer],
):
    """Deployment-level gate for the generate-path post-layers handler.

    CUSTOM_OUTPUT_PROCESSOR names a python module (dotted path, or a .py file
    path) that defines `create_custom_module(config, tokenizer)` returning a
    CustomModule whose handler runs on the generate path (see
    CustomHandler.trigger_mode). Unset means the deployment has no handler and
    the engine code path is unchanged. Any load failure fails startup — a
    deployment that declares a processor it cannot load must not come up.
    """
    target = os.environ.get("CUSTOM_OUTPUT_PROCESSOR")
    if not target:
        return None

    mode = os.environ.get("CUSTOM_PROCESSOR_MODE", "eager")
    if mode != "eager":
        raise RuntimeError(
            f"CUSTOM_PROCESSOR_MODE={mode} is not implemented, only 'eager' is"
        )

    if target.endswith(".py"):
        spec = importlib.util.spec_from_file_location(
            "rtp_llm_custom_output_processor", target
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load CUSTOM_OUTPUT_PROCESSOR file: {target}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    custom_module = module.create_custom_module(config, tokenizer)
    if custom_module is None:
        raise RuntimeError(
            f"CUSTOM_OUTPUT_PROCESSOR {target} create_custom_module returned None"
        )
    logging.info(f"loaded post-layers custom module from {target}")
    return custom_module
