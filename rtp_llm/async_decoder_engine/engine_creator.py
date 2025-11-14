import logging
from enum import Enum
from typing import Optional

import torch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from rtp_llm.async_decoder_engine.rpc_engine import RPCEngine
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel


class ExecutorType(Enum):
    Normal = "normal"
    Embedding = "embedding"


def check_exeutor_type(model: BaseModel):
    if model.custom_module is not None:
        return ExecutorType.Embedding
    return ExecutorType.Normal


def create_engine(
    model: BaseModel, 
    alog_conf_path: str,
    gang_info,
    propose_model: Optional[ProposeModel] = None
) -> BaseEngine:
    """
    Create an engine for the given model and config.
    
    Args:
        model: The BaseModel instance
        alog_conf_path: Path to the alog configuration file
        gang_info: GangInfo instance from GangServer
        propose_model: Optional propose model for speculative decoding
    
    Returns:
        BaseEngine instance
    """
    torch.ops.rtp_llm.init_engine(alog_conf_path)
    
    executor_type = check_exeutor_type(model)
    logging.info(f"executor_type: {executor_type}")
    if executor_type == ExecutorType.Normal:
        return RPCEngine(model, gang_info, propose_model)
    elif executor_type == ExecutorType.Embedding:
        return EmbeddingCppEngine(model)
    else:
        raise Exception(f"unsupported executor type: {executor_type}")
