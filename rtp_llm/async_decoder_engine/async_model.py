import logging
from typing import Optional

import torch

from rtp_llm.async_decoder_engine.engine_creator import create_engine
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.ops import TaskType
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import EngineScheduleInfo, KVCacheInfo, WorkerStatusInfo
from rtp_llm.utils.gemm_utils.device_map import get_device


class AsyncModel:
    def __init__(
        self, model: BaseModel, propose_model: Optional[ProposeModel] = None
    ) -> None:
        self.model = model
        self.propose_model = propose_model
        self.config = model.config
        self.model_runtime_meta = self._model_runtime_meta()
        self.role_type = str(model.config.role_type)

        assert self.config.max_seq_len > 0
        self.tokenizer = model.tokenizer
        self.decoder_engine_ = create_engine(self.model, self.config, self.propose_model)
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
        quant_algo = self.model.py_model_config.quant_algo
        weight_info = f"W{quant_algo.getWeightBits()}A{quant_algo.getActivationBits()}"
        return "_".join([manchine_name, parallel_info, weight_info])

    @property
    def default_generate_config(self) -> GenerateConfig:
        return self.model.default_generate_config

    @property
    def task_type(self) -> TaskType:
        return self.model.py_model_config.task_type

    def stop(self):
        self.decoder_engine_.stop()

    @torch.no_grad()
    def enqueue(self, input: GenerateInput):
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            raise Exception("bug, not supposed to be here")
        if input.prompt_length <= 0:
            raise FtRuntimeException(
                ExceptionType.LONG_PROMPT_ERROR,
                f"model tokens can not be empty, request length is {input.prompt_length}",
            )
        max_new_tokens = min(
            self.config.max_seq_len - input.prompt_length,
            input.generate_config.max_new_tokens,
        )
        if max_new_tokens <= 0:
            raise FtRuntimeException(
                ExceptionType.LONG_PROMPT_ERROR,
                f"model max tokens is {self.config.max_seq_len}, "
                f"request length is {input.prompt_length}, max_new_tokens is {max_new_tokens}",
            )
        return self.decoder_engine_.decode(input)

    def get_worker_status_info(self, latest_finished_version: int) -> WorkerStatusInfo:
        return self.decoder_engine_.get_worker_status_info(latest_finished_version)

    def get_cache_status_info(self, latest_cache_version: int) -> KVCacheInfo:
        return self.decoder_engine_.get_cache_status_info(latest_cache_version)

    def get_engine_schedule_info(
        self, latest_finised_version: int
    ) -> EngineScheduleInfo:
        return self.decoder_engine_.get_engine_schedule_info(latest_finised_version)
