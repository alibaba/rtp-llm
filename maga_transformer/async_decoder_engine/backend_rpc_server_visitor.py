import torch
import logging
from typing import Optional, Dict

from maga_transformer.models.base_model import GenerateInput
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.cpp.model_rpc.model_rpc_client import ModelRpcClient
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class BackendRPCServerVisitor:
    def __init__(self, model_config : GptInitModelParameters) -> None:
        self.config = model_config
        assert self.config.max_seq_len > 0
        self.model_rpc_client = ModelRpcClient(self.config)
    
    @torch.no_grad()
    def enqueue(self, input: GenerateInput):
        if input.prompt_length <= 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR,
                                     f"model tokens can not be empty, request length is {input.prompt_length}")
        max_new_tokens = min(self.config.max_seq_len - input.prompt_length, input.generate_config.max_new_tokens)
        if max_new_tokens <= 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR,
                f"model max tokens is {self.config.max_seq_len}, " \
                f"request length is {input.prompt_length}, max_new_tokens is {max_new_tokens}")
        return self.model_rpc_client.enqueue(input)