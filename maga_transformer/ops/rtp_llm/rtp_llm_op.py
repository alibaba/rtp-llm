from typing import Any, Dict, Tuple
import torch
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info, g_master_info
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.ops import RtpLLMOp as CppRtpLLMOp

class RtpLLMOp(FTOPBase):
    def __init__(self, config: GptInitModelParameters, is_sp: bool, mm_engine):
        super().__init__()
        self.config = config
        self.is_sp = is_sp
        self.ft_op = CppRtpLLMOp()
        self.mm_engine = mm_engine

    def _initialize_op(self, force_init: bool=False):
        assert self.weight
        self.ft_op.init( # type: ignore
            self.config.gpt_init_params,
            self.weight.weights,
            self.weight.global_weights,
            self.mm_engine)

        for id, lora_weight in self.weight.lora_resource.lora_map.weights_map.items():
            self.ft_op.add_lora( # type: ignore
                id,
                lora_weight.lora_a_weights,
                lora_weight.lora_b_weights)

    def update_lora(self):
        if self.weight != None:
            for id in self.weight.lora_resource.to_remove_lora_id:
                self.ft_op.remove_lora(id)
            for id in self.weight.lora_resource.to_add_lora_id:
                lora_weight = self.weight.lora_resource.lora_map.weights_map[id]
                self.ft_op.add_lora(id, lora_weight.lora_a_weights, lora_weight.lora_b_weights)
                
    def stop(self):
        self.ft_op.stop() # type: ignore

    def get_kv_cache_info(self) -> Tuple[int, int]:
        available_kv_cache, total_kv_cache = self.ft_op.get_kv_cache_info() # type: ignore
        return available_kv_cache, total_kv_cache
