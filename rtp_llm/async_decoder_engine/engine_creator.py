from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from rtp_llm.ops import TaskType
from rtp_llm.utils.time_util import timer_wrapper

if TYPE_CHECKING:
    from rtp_llm.async_decoder_engine.base_engine import BaseEngine
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.models.base_model import BaseModel
    from rtp_llm.models.propose_model.propose_model import ProposeModel


@timer_wrapper(description="create async engine")
def create_engine(
    model: "BaseModel",
    engine_config: "EngineConfig",
    alog_conf_path: str,
    world_info=None,
    propose_model: Optional["ProposeModel"] = None,
) -> "BaseEngine":
    """
    Create an engine for the given model and config.

    Args:
        model: The BaseModel instance
        engine_config: EngineConfig instance containing runtime and parallelism configs
        alog_conf_path: Path to the alog configuration file
        world_info: Distributed world info (e.g., from get_world_info())
        propose_model: Optional propose model for speculative decoding
        mm_process_engine: Optional MMProcessEngine instance for multimodal processing in EmbeddingCppEngine

    Returns:
        BaseEngine instance
    """
    import torch

    from rtp_llm.ops import ensure_engine_ops_loaded

    ensure_engine_ops_loaded()
    torch.ops.rtp_llm.init_engine(alog_conf_path)

    if model.model_config.task_type == TaskType.LANGUAGE_MODEL:
        from rtp_llm.async_decoder_engine.rpc_engine import LanguageCppEngine

        logging.info("create llm engine")
        return LanguageCppEngine(
            model=model,
            engine_config=engine_config,
            world_info=world_info,
            propose_model=propose_model,
        )
    else:
        from rtp_llm.async_decoder_engine.embedding.embedding_engine import (
            EmbeddingCppEngine,
        )

        logging.info("create embedding engine")
        return EmbeddingCppEngine(model, engine_config)
