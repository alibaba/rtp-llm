import torch
import logging
from typing import Optional, List, Dict, Union, Tuple
from PIL import Image

from maga_transformer.utils.util import get_mem_info
from maga_transformer.utils.time_util import Timer
from maga_transformer.models.base_model import BaseModel, GenerateInput
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.engine_creator import create_engine
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.task_type import TaskType
from maga_transformer.async_decoder_engine.base_engine import KVCacheInfo

class AsyncModel:
    def __init__(self, model: BaseModel, sp_model: Optional[BaseModel] = None, use_rpc: bool = True) -> None:
        self.model = model
        self.sp_model = sp_model
        self.config = model.config

        assert self.config.max_seq_len > 0
        self.tokenizer = model.tokenizer
        logging.info(f'first mem info: used:{get_mem_info().used} free: {get_mem_info().free}')
        self.decoder_engine_ = create_engine(self.model, self.config, self.sp_model, self.sp_model.config if self.sp_model else None, use_rpc)
        self.decoder_engine_.start()

    def is_multimodal(self) -> bool:
        return self.model.is_multimodal()

    def expand_token_id(self, token_ids: List[int], images: List[Image.Image]) -> Tuple[List[int], Union[torch.Tensor, List[torch.Tensor]], List[int]]:
        assert self.is_multimodal()
        return self.model.expand_token_id(token_ids, images)

    @property
    def default_generate_config(self) -> GenerateConfig:
        return self.model.default_generate_config

    @property
    def task_type(self) -> TaskType:
        return self.model.task_type

    def stop(self):
        self.decoder_engine_.stop()

    def update(self, lora_infos: Dict[str, str]):
        self.decoder_engine_.update_lora(lora_infos)

    @torch.no_grad()
    def enqueue(self, input: GenerateInput):
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            raise Exception('bug, not supposed to be here')
        return self.decoder_engine_.decode(input)

    def get_kv_cache_info(self) -> KVCacheInfo:
        return self.decoder_engine_.get_kv_cache_info()
