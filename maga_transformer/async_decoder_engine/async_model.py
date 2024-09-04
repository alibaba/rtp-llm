import torch
from typing import Optional, Dict

from maga_transformer.models.propose_model.propose_model import ProposeModel
from maga_transformer.models.base_model import BaseModel, GenerateInput
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.engine_creator import create_engine
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.task_type import TaskType
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.ops import LoadBalanceInfo


class AsyncModel:
    def __init__(self, model: BaseModel, propose_model: Optional[ProposeModel] = None) -> None:
        self.model = model
        self.propose_model = propose_model
        self.config = model.config

        assert self.config.max_seq_len > 0
        self.tokenizer = model.tokenizer
        self.decoder_engine_ = create_engine(self.model, self.propose_model)
        self.decoder_engine_.start()

    def is_multimodal(self) -> bool:
        return self.model.is_multimodal()

    @property
    def default_generate_config(self) -> GenerateConfig:
        return self.model.default_generate_config

    @property
    def task_type(self) -> TaskType:
        return self.model.task_type

    def stop(self):
        self.decoder_engine_.stop()

    @torch.no_grad()
    def enqueue(self, input: GenerateInput):
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            raise Exception('bug, not supposed to be here')
        if input.prompt_length <= 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR,
                                     f"model tokens can not be empty, request length is {input.prompt_length}")
        max_new_tokens = min(self.config.max_seq_len - input.prompt_length, input.generate_config.max_new_tokens)
        if max_new_tokens <= 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR,
                                     f"model max tokens is {self.config.max_seq_len}, request length is {input.prompt_length}, max_new_tokens is {max_new_tokens}")
        return self.decoder_engine_.decode(input)

    def get_load_balance_info(self) -> LoadBalanceInfo:
        return self.decoder_engine_.get_load_balance_info()
