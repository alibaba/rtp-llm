import torch
import logging
from typing import Optional, Dict

from maga_transformer.models.propose_model.propose_model import ProposeModel
from maga_transformer.models.base_model import BaseModel, GenerateInput
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.engine_creator import create_engine
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.task_type import TaskType
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.ops import LoadBalanceInfo, EngineScheduleInfo

from maga_transformer.utils.gemm_utils.device_map import get_device    

class AsyncModel:
    def __init__(self, model: BaseModel, propose_model: Optional[ProposeModel] = None) -> None:
        self.model = model
        self.propose_model = propose_model
        self.config = model.config
        self.model_runtime_meta = self._model_runtime_meta()

        assert self.config.max_seq_len > 0
        self.tokenizer = model.tokenizer
        self.decoder_engine_ = create_engine(self.model, self.propose_model)
        self.decoder_engine_.start()

    def is_multimodal(self) -> bool:
        return self.config.is_multimodal
    
    def _model_runtime_meta(self) -> str:        
        try:
            device_name = torch.cuda.get_device_name(0)
            manchine_name = get_device(device_name).upper()
        except Exception as e:
            logging.info(f"error get device name with error: {e}")
            manchine_name = "unknown"
        parallel_info = f"TP{g_parallel_info.tp_size}_PP{g_parallel_info.pp_size}_EP{g_parallel_info.ep_size}"
        weight_info = f"W{self.config.gpt_init_params.quant_algo.getWeightBits()}A{self.config.gpt_init_params.quant_algo.getActivationBits()}"
        return "_".join([manchine_name, parallel_info, weight_info])

    @property
    def default_generate_config(self) -> GenerateConfig:
        return self.model.default_generate_config

    @property
    def task_type(self) -> TaskType:
        return self.model.task_type

    def stop(self):
        self.decoder_engine_.stop()

    def ready(self):
        return self.decoder_engine_.ready()

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
                                     f"model max tokens is {self.config.max_seq_len}, " \
                                     f"request length is {input.prompt_length}, max_new_tokens is {max_new_tokens}")
        return self.decoder_engine_.decode(input)

    def get_load_balance_info(self) -> LoadBalanceInfo:
        return self.decoder_engine_.get_load_balance_info()

    def get_engine_schedule_info(self) -> EngineScheduleInfo:
        return self.decoder_engine_.get_engine_schedule_info()
