import logging
import torch
from enum import Enum
from typing import Optional
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.models.base_model import BaseModel
from rtp_llm.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from rtp_llm.async_decoder_engine.rpc_engine import RPCEngine
from rtp_llm.async_decoder_engine.base_engine import BaseEngine


class ExecutorType(Enum):
    Normal = "normal"
    Embedding = 'embedding'


def check_exeutor_type(model: BaseModel):
    if model.custom_module is not None:
        return ExecutorType.Embedding
    return ExecutorType.Normal


def create_engine(model: BaseModel, propose_model: Optional[ProposeModel] = None) -> BaseEngine:
    torch.ops.rtp_llm.init_engine()
    executor_type = check_exeutor_type(model)
    logging.info(f"executor_type: {executor_type}")
    if executor_type == ExecutorType.Normal:
        return RPCEngine(model, propose_model)
    elif executor_type == ExecutorType.Embedding:
        return EmbeddingCppEngine(model)
    else:
        raise Exception(f"unsupported executor type: {executor_type}")
