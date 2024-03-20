import torch
import logging
from enum import Enum
from typing import List, Dict, Tuple, NamedTuple, Union, Optional
from pydantic import BaseModel

from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.models.base_model import GenerateInput
from maga_transformer.utils.time_util import Timer
from maga_transformer.async_decoder_engine.cache_manager import CacheManager
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class PrefixType(Enum):
    PromptTuning = 0
    PTuningV2 = 1
    KVCacheReuse = 2
    NoPrefix = 3

# TODO(xinfei.sxf) refactor this class
class PrefixParams(NamedTuple):
    prefix_type: PrefixType
    prefix_length: int
    block_cache: List[int]
    prefix_tensor: Optional[torch.Tensor]

class PrefixInfo(BaseModel):
    ptuning: bool = False
    count_length: bool = True
    count_prefix_length: bool = True
    prefix_tensors: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

class PtuningBase:
    prefix_type: PrefixType

    def get_prefix_params(self, generate_config: GenerateConfig) -> Tuple[PrefixType, torch.Tensor]:
        raise NotImplementedError()

    def get_block_indice(self, block_num: int, generate_config: GenerateConfig) -> Tuple[List[int], int]:
        raise NotImplementedError()

    def calc_prefix_block_num(self, generate_config: GenerateConfig):
        raise NotImplementedError()

    def get_ptuning_info(self, generate_config: GenerateConfig):
        _, prefix_tensors = self.get_prefix_params(
            generate_config)
        return PrefixInfo(
            ptuning=True,
            count_length=self.prefix_type == PrefixType.PromptTuning,
            count_prefix_length=self.prefix_type != PrefixType.PTuningV2,
            prefix_tensors=prefix_tensors)

class Ptuning(PtuningBase):
    def __init__(self, config: GptInitModelParameters, cache_manage: CacheManager, 
                 prefix_params: PrefixParams, insert_resident_cache: bool=False):
        self.prefix_params = prefix_params
        self.cache_manage = cache_manage
        self.prefix_block_indice = self.prefix_params.block_cache
        self.prefix_type = self.prefix_params.prefix_type
        self._maybe_insert_prefix_cache(insert_resident_cache)

        self.prefix_additional_block = -1
        if self.prefix_params.prefix_length % config.seq_size_per_block != 0:
            self.prefix_additional_block = self.prefix_block_indice[-1]
            self.prefix_block_indice = self.prefix_block_indice[: -1]

    def _maybe_insert_prefix_cache(self, insert_resident_cache: bool):
        if insert_resident_cache:
            assert self.prefix_params.prefix_tensor is not None, "prefix tensor should not be None when insert prefix cache"
            self.cache_manage.insert_resident_cache(self.prefix_params.block_cache, self.prefix_params.prefix_tensor.numpy().tolist())

    def calc_prefix_block_num(self, generate_config: GenerateConfig):
        return len(self.prefix_block_indice)

    def get_prefix_params(self, generate_config: GenerateConfig):
        return self.prefix_type, self.prefix_params.prefix_tensor

    def get_block_indice(self, block_num: int, generate_config: GenerateConfig) -> Tuple[List[int], int]:
        # copy last block of prefix_block if mod != 0
        block_indice = self.cache_manage.malloc(block_num)
        if self.prefix_additional_block > 0:
            block_indice += self.cache_manage.malloc(1)
            self.cache_manage.block_copy(self.prefix_additional_block, block_indice[0])
        block_indice = self.prefix_block_indice + block_indice
        return block_indice, self.prefix_params.prefix_length

class MultiTaskPtuning(PtuningBase):
    def __init__(self, config: GptInitModelParameters, cache_manage: CacheManager, prefix_params_map: Dict[int, PrefixParams]):
        self.cache_manage_ = cache_manage
        self.ptunings_: Dict[int, Ptuning] = {id: Ptuning(config, cache_manage, prefix_params, True) for id, prefix_params in prefix_params_map.items()}
        self.prefix_type = PrefixType.PromptTuning

    def get_block_indice(self, block_num: int, generate_config: GenerateConfig) -> Tuple[List[int], int]:
        task_id = generate_config.task_id
        if not task_id or task_id not in self.ptunings_:
            return self.cache_manage_.malloc(block_num), 0
        return self.ptunings_[task_id].get_block_indice(block_num, generate_config)

    def get_prefix_params(self, generate_config: GenerateConfig) -> Tuple[PrefixType, Optional[torch.Tensor]]:

        task_id = generate_config.task_id
        if not task_id or task_id not in self.ptunings_:
            return PrefixType.NoPrefix, torch.zeros([0])

        return self.ptunings_[task_id].get_prefix_params(generate_config)

    def calc_prefix_block_num(self, generate_config: GenerateConfig):
        task_id = generate_config.task_id
        if not task_id or task_id not in self.ptunings_:
            return 0
        return self.ptunings_[task_id].calc_prefix_block_num(generate_config)
