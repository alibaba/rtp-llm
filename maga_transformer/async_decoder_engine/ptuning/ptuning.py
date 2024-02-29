import torch
import logging
from enum import Enum
from typing import List, Dict, Tuple, NamedTuple, Union, Optional
from pydantic import BaseModel

from maga_transformer.config.generate_config import GenerateConfig
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
    prefix_kvcache: Union[torch.Tensor, Dict[int, torch.Tensor]]
    prefix_type: PrefixType
    prefix_tensor: Optional[Dict[int, torch.Tensor]]

class PrefixInfo(BaseModel):
    ptuning: bool = False
    count_length: bool = True
    count_prefix_length: bool = True
    prefix_tensors: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

class PtuningBase:
    def get_ptuning_info(self, generate_config):
        _, prefix_tensors = self.get_prefix_params(
            generate_config)
        return PrefixInfo(
            ptuning=True,
            count_length=self.prefix_type == PrefixType.PromptTuning,
            count_prefix_length=self.prefix_type != PrefixType.PTuningV2,
            prefix_tensors=prefix_tensors)
    
class Ptuning(PtuningBase):
    def __init__(self, config: GptInitModelParameters, cache_manage: CacheManager, prefix_prompt: torch.Tensor,
                 prefix_tensor: torch.Tensor, prefix_type: PrefixType, insert_cache: bool = False) -> None:
        self.config = config
        self.cache_manage = cache_manage
        self.prefix_seq_length = prefix_prompt.size(-2)
        self.prefix_block_indice: List[int] = []
        self.prefix_additional_block = -1
        self.prefix_type = prefix_type
        self.prefix_tensor = prefix_tensor
        self.insert_cache = insert_cache
        with Timer() as t:
            self.create_prefix_block(prefix_prompt)
        logging.info(f"create prefix block time: {t.cost_ms()}ms")

    def calc_prefix_block_num(self, generate_config: GenerateConfig):
        return len(self.prefix_block_indice)

    # input shape [layer_num, pre_seq_len, head_num, size_per_head]
    # dest k shape [layer_num, block_nums, head_num, seq_num_block, size_per_head]
    # dest v shape [layer_num, block_nums, head_num, seq_num_block, size_per_head]
    def _set_kv_prefix_block(self, kv_prefix_prompt: torch.Tensor, prefix_block_indice: List[int]):
        k_prefix_prompt = kv_prefix_prompt[0]
        v_prefix_prompt = kv_prefix_prompt[1]
        layer_num = k_prefix_prompt.size(0)
        pre_seq_len = k_prefix_prompt.size(1)
        head_num = k_prefix_prompt.size(2)
        size_per_head = k_prefix_prompt.size(3)
        block_indice_length = len(prefix_block_indice)
        append_length = len(prefix_block_indice) * self.config.seq_size_per_block - pre_seq_len
        blank_tensor = torch.zeros(layer_num, append_length, head_num, size_per_head).to(k_prefix_prompt)
        # [layer_num, block_num * seq_num_per_block, head_num, size_per_head]
        tiled_k_prefix_prompt = torch.concat([k_prefix_prompt, blank_tensor], dim=1)
        tiled_v_prefix_prompt = torch.concat([v_prefix_prompt, blank_tensor], dim=1)
        tiled_k_prefix_prompt = tiled_k_prefix_prompt.reshape(layer_num, block_indice_length, self.config.seq_size_per_block, head_num, size_per_head).permute(0, 1, 3, 2, 4).contiguous()
        tiled_v_prefix_prompt = tiled_v_prefix_prompt.reshape(layer_num, block_indice_length, self.config.seq_size_per_block, head_num, size_per_head).permute(0, 1, 3, 2, 4).contiguous()
        for i in range(block_indice_length):
            self.cache_manage.set_kv_block_value(prefix_block_indice[i], tiled_k_prefix_prompt[ :, i, ...], tiled_v_prefix_prompt[ :, i, ...])
        if self.insert_cache:
            self.cache_manage.insert_resident_cache(prefix_block_indice, self.prefix_tensor.numpy().tolist())

    def create_prefix_block(self, prefix_prompt: torch.Tensor):
        assert isinstance(prefix_prompt,  torch.Tensor), "prefix prompt is not torch.Tensor"
        prefix_seq_length = prefix_prompt.size(-2)
        # prefix_prompt shape [layer_num * 2, head_num_kv, pre_seq_len, size_per_head]
        prefix_prompt = prefix_prompt.reshape(self.config.layer_num, 2, prefix_prompt.size(1), prefix_prompt.size(2), prefix_prompt.size(3)).permute(1, 0, 3, 2, 4).contiguous()
        prefix_blocks = (prefix_seq_length - 1) // self.config.seq_size_per_block + 1
        prefix_block_indice = self.cache_manage.malloc(prefix_blocks)

        self._set_kv_prefix_block(prefix_prompt, prefix_block_indice)
        self.prefix_block_indice = prefix_block_indice
        if prefix_seq_length % self.config.seq_size_per_block:
            self.prefix_additional_block = self.prefix_block_indice[-1]
            self.prefix_block_indice = self.prefix_block_indice[: -1]

    def get_prefix_params(self, generate_config: GenerateConfig):
        return self.prefix_type, self.prefix_tensor

    def get_block_indice(self, block_num: int, generate_config: GenerateConfig) -> Tuple[List[int], int]:
        # copy last block of prefix_block if mod != 0
        block_indice = self.cache_manage.malloc(block_num)
        if self.prefix_additional_block > 0:
            block_indice += self.cache_manage.malloc(1)
            self.cache_manage.block_copy(self.prefix_additional_block, block_indice[0])
        block_indice = self.prefix_block_indice + block_indice
        return block_indice, self.prefix_seq_length

'''
包含两种可能的类型:
1. prefix-tuning count_length = 0
2. prompt-tuning count_length = 1
'''

class MultiTaskPtuning(PtuningBase):
    def __init__(self, config: GptInitModelParameters, cache_manage: CacheManager,
                 prefix_prompts: Dict[int, torch.Tensor], prefix_type: PrefixType, prefix_tensors: Dict[int, torch.Tensor]):
        #TODO(xinfei.sxf) 这句对吗？
        assert prefix_type in [PrefixType.PromptTuning, PrefixType]
        self.cache_manage_ = cache_manage
        self.ptunings_ = {id: Ptuning(config, cache_manage, prompt, prefix_tensors[id], prefix_type, insert_cache=True) for id, prompt in prefix_prompts.items()}
        self.prefix_type = prefix_type
        self.prefix_tensors = prefix_tensors

    def get_block_indice(self, block_num: int, generate_config: GenerateConfig) -> Tuple[List[int], int]:
        task_id = generate_config.task_id
        if not task_id or task_id not in self.ptunings_:
            return self.cache_manage_.malloc(block_num), 0
        return self.ptunings_[task_id].get_block_indice(block_num, generate_config)

    def get_prefix_params(self, generate_config: GenerateConfig) -> Tuple[PrefixType, torch.Tensor]:
        task_id = generate_config.task_id
        if not task_id or task_id not in self.ptunings_:
            return PrefixType.NoPrefix, torch.zeros([0])
        return self.ptunings_[task_id].get_prefix_params(generate_config)

    def calc_prefix_block_num(self, generate_config: GenerateConfig):
        task_id = generate_config.task_id
        if not task_id or task_id not in self.ptunings_:
            return 0
        return self.ptunings_[task_id].calc_prefix_block_num(generate_config)
