import gc
import asyncio
import torch
import logging
import traceback
from typing import Optional, Iterator, List, Any, Generator, AsyncGenerator, Dict, Union, Tuple
from PIL import Image
from transformers import PreTrainedTokenizer

from maga_transformer.utils.util import get_mem_info
from maga_transformer.utils.time_util import Timer
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.models.base_model import BaseModel, GenerateInput
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.engine_creator import create_engine
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.gpt_init_model_parameters import ModelType

class AsyncModel:
    def __init__(self, model: BaseModel, sp_model: Optional[BaseModel] = None) -> None:
        self.model = model
        self.sp_model = sp_model
        self.config = model.config
        self.vit_expand_token_id_lock = asyncio.Lock()

        assert self.config.max_seq_len > 0
        self.tokenizer = model.tokenizer        
        logging.info(f'first mem info: used:{get_mem_info().used} free: {get_mem_info().free}')
        if self.sp_model is not None:            
            self.decoder_engine_ = create_engine(self.model, self.config, self.sp_model, self.sp_model.config)
        else:
            self.decoder_engine_ = create_engine(model, self.config)
        self.decoder_engine_.start()

    def is_multimodal(self) -> bool:
        return self.model.is_multimodal()

    async def expand_token_id(self, token_ids: List[int], images: List[Image.Image]) -> Tuple[List[int], Union[torch.Tensor, List[torch.Tensor]]]:
        assert self.is_multimodal()
        async with self.vit_expand_token_id_lock:
            return self.model.expand_token_id(token_ids, images)

    def load(self, ref_model: Optional[torch.nn.Module] = None):
        self.model.load(ref_model)
        self.decoder_engine_.executor_.gpt_op.set_weight(self.model.weight)

    @property
    def default_generate_config(self) -> GenerateConfig:
        return self.model.default_generate_config
    
    @property
    def model_type(self) -> ModelType:
        return self.model.model_type

    # just for perf test
    def enable_perf_test_schedule_strategy(self):
        self.decoder_engine_.scheduler_.enable_perf_test_schedule_strategy()

    def stop(self):
        self.decoder_engine_.stop()

    def update(self, lora_infos: Dict[str, str]):
        with Timer() as timer:
            self.decoder_engine_.executor_.model_ops.gpt_op.weight.lora_resource.update(lora_infos)
        logging.info(f'update lora weights time: {timer.cost_ms() / 1000 :.2f} s')

    @torch.no_grad()
    def enqueue(self, input: GenerateInput):
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            raise Exception('bug, not supposed to be here')
        return self.decoder_engine_.decode(input)