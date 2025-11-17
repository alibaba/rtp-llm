import logging
from enum import Enum
from typing import Optional

import torch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from rtp_llm.async_decoder_engine.rpc_engine import LanguageCppEngine
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.ops import TaskType

def create_engine(
    model: BaseModel,
    engine_config: EngineConfig,
    alog_conf_path: str,
    world_info = None,
    propose_model: Optional[ProposeModel] = None
) -> BaseEngine:
    """
    Create an engine for the given model and config.
    
    Args:
        model: The BaseModel instance
        engine_config: EngineConfig instance containing runtime and parallelism configs
        alog_conf_path: Path to the alog configuration file
        gang_info: GangInfo instance from GangServer
        propose_model: Optional propose model for speculative decoding
    
    Returns:
        BaseEngine instance
    """
    torch.ops.rtp_llm.init_engine(alog_conf_path)

    if model.model_config.task_type == TaskType.LANGUAGE_MODEL:
        return LanguageCppEngine(
            model=model,
            engine_config=engine_config,
            world_info=gang_info,
            propose_model=propose_model
        )
        logging.info("create llm engine")
    else:
        logging.info("create embedding engine")
        return EmbeddingCppEngine(model, engine_config)

