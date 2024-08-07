import os
import logging
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union, Any, Dict
from maga_transformer.utils.util import get_mem_info
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.base_model import BaseModel
# from maga_transformer.async_decoder_engine.ptuning import PrefixParams
# from maga_transformer.async_decoder_engine.speculative.sp_model_executor import SpModelExecutor
# from maga_transformer.async_decoder_engine.medusa.medusa_model_executor import MedusaModelExecutor
# from maga_transformer.async_decoder_engine.medusa.utils import generate_medusa_buffers
from maga_transformer.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from maga_transformer.async_decoder_engine.rpc_engine import RPCEngine
# from maga_transformer.async_decoder_engine.ptuning.ptuning_utils import PtuningConstructor
from maga_transformer.async_decoder_engine.base_engine import BaseEngine

class ExecutorType(Enum):
    Normal = "normal"
    Speculative = "speculative"
    Medusa = "medusa"
    Embedding = 'embedding'

def check_exeutor_type(model: BaseModel, config: GptInitModelParameters, speculative_model: Any = None, speculative_config: Optional[GptInitModelParameters] = None):
    # if speculative_model is not None:
    #     assert speculative_config is not None and speculative_model is not None, "speculative_config should not be None"
    #     return ExecutorType.Speculative
    # if model.medusa_head is not None:
    #     return ExecutorType.Medusa
    if model.custom_module is not None:
        return ExecutorType.Embedding
    return ExecutorType.Normal

def create_engine(model: BaseModel, config: GptInitModelParameters, speculative_model: Any = None, speculative_config: Optional[GptInitModelParameters] = None) -> BaseEngine:
    executor_type = check_exeutor_type(model, config, speculative_model, speculative_config)
    logging.info(f"executor_type: {executor_type}")
    if executor_type == ExecutorType.Normal:
        return _create_cpp_engine(model, config)
    # elif executor_type == ExecutorType.Speculative:
    #     assert speculative_config
    #     return _create_sp_engine(model, config, speculative_model, speculative_config)
    # elif executor_type == ExecutorType.Medusa:
    #     return _create_medusa_engine(model, config)
    elif executor_type == ExecutorType.Embedding:
        return EmbeddingCppEngine(model)
    else:
        raise Exception(f"unsupported executor type: {executor_type}")

def _create_cpp_engine(model: BaseModel, config: GptInitModelParameters) -> RPCEngine:
    logging.info(f'ft op mem info: used: {get_mem_info().used} free: {get_mem_info().free}')
    return RPCEngine(model, None)
