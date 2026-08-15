import importlib
import importlib.util
import logging
import os
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
        return create_post_layers_module(config, tokenizer)

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


def create_post_layers_module(
    config: ModelConfig,
    tokenizer: Optional[BaseTokenizer],
):
    """Deployment-level gate for the generate-path post-layers handler.

    CUSTOM_OUTPUT_PROCESSOR names a python module (dotted path, or a .py file
    path) that defines `create_custom_module(config, tokenizer)` returning a
    CustomModule whose handler runs on the generate path (see
    CustomHandler.trigger_mode). Relative .py paths are resolved from the
    checkpoint directory so the processor can be shipped with the model.
    Unset means the deployment has no handler and the engine code path is
    unchanged. Any load failure fails startup — a deployment that declares a
    processor it cannot load must not come up.
    """
    target = os.environ.get("CUSTOM_OUTPUT_PROCESSOR")
    if not target:
        return None

    mode = os.environ.get("CUSTOM_PROCESSOR_MODE", "eager")
    if mode not in ("eager", "compiled"):
        raise RuntimeError(
            f"CUSTOM_PROCESSOR_MODE={mode} is not implemented, "
            "only 'eager' and 'compiled' are"
        )

    resolved_target = target
    if target.endswith(".py"):
        processor_path = target
        if not os.path.isabs(processor_path):
            ckpt_path = getattr(config, "ckpt_path", None)
            if not ckpt_path:
                raise RuntimeError(
                    "relative CUSTOM_OUTPUT_PROCESSOR .py path requires "
                    "a configured checkpoint path"
                )
            processor_path = os.path.join(ckpt_path, processor_path)
        resolved_target = processor_path
        spec = importlib.util.spec_from_file_location(
            "rtp_llm_custom_output_processor", processor_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"cannot load CUSTOM_OUTPUT_PROCESSOR file: {processor_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    custom_module = module.create_custom_module(config, tokenizer)
    if custom_module is None:
        raise RuntimeError(
            f"CUSTOM_OUTPUT_PROCESSOR {target} create_custom_module returned None"
        )
    if mode == "compiled":
        handler = custom_module.get_handler()
        if handler.compiled_module() is None:
            raise RuntimeError(
                f"CUSTOM_PROCESSOR_MODE=compiled but {target} handler does not "
                "implement compiled_module()"
            )
        # the actual AOT compile runs at handler injection time, after
        # init(tensor_map) has loaded the real weights — see
        # CustomHandler.ensure_aoti_package
        handler._aoti_requested = True
    logging.info(
        f"loaded post-layers custom module from {resolved_target}, mode={mode}"
    )
    return custom_module
