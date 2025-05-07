import logging
import torch

from typing import Any, Dict, Optional, Union
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.weight_module import WeightModule, CompositeWeight, QuantWeight, AtomicWeight
from maga_transformer.utils.database import BaseDatabase
from maga_transformer.utils.model_weight import W
from maga_transformer.model_loader.ffn_weight import MoeAtomicWeight


class WeightOnlyPerColWeight(CompositeWeight, QuantWeight):
    int8_attn_weights_map = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.mla_fusedqkrope_w: W.mla_fusedqkrope_s,
        W.mla_fusedqkrope_no_lora_w: W.mla_fusedqkrope_no_lora_s,
        W.mla_q_b_w: W.mla_q_b_s,
        W.mla_k_nope_w: W.mla_k_nope_s,
        W.mla_v_w: W.mla_v_s,
    }

    int8_attn_vision_weights_map = {
        W.vision_attn_qkv_w: W.vision_attn_qkv_s,
        W.vision_attn_o_w: W.vision_attn_o_s,
    }

    int8_ffn_weights_maps = {
        W.ffn_w1: W.ffn_s1,
        W.ffn_w3: W.ffn_s3,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w13: W.ffn_s13
    }

    int8_vision_ffn_weights_maps = {
        W.vision_ffn_w1: W.vision_ffn_s1,
        W.vision_ffn_w3: W.vision_ffn_s3,
        W.vision_ffn_w2: W.vision_ffn_s2,
    }

    int8_partial_moe_weights_maps = {
        W.moe_w1: W.moe_s1,
        W.moe_w2: W.moe_s2,
    }

    weight_only_w = {
        **int8_attn_weights_map,
        **int8_attn_vision_weights_map,
        **int8_ffn_weights_maps,
        **int8_vision_ffn_weights_maps,
        **int8_partial_moe_weights_maps,
    }

    @classmethod
    def support(cls, quant_algo: Any, src_weight_info: WeightModule) -> bool:
        name = src_weight_info.name
        return quant_algo.isWeightOnlyPerCol() and quant_algo.getWeightBits() == 8 and name in cls.weight_only_w

    def __init__(self, src_weight_info: AtomicWeight, quant_algo: Any, *args, **kwargs):
        kernel: AtomicWeight = src_weight_info
        params = src_weight_info.extract_params(src_weight_info.__class__, src_weight_info, quant_algo)
        params['name'] = self.weight_only_w.get(src_weight_info.name)
        scale: AtomicWeight = src_weight_info.from_params(params)
        sub_weights = {kernel.name: kernel, scale.name: scale}
        super().__init__(sub_weights, quant_algo=quant_algo, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale


    def _load_raw_tensor(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig):
        kernel = self.kernel._load_raw_tensor(database, layer_id, device, load_config)
        return kernel

    def _split(self, tensor: torch.Tensor, load_config: LoadConfig):
        kernel = self.kernel._split(tensor, load_config)
        return kernel

    def _postprocess(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str, load_config: LoadConfig) -> torch.Tensor:
        kernel = tensor.get(self.kernel.name)
        if isinstance(self.kernel, MoeAtomicWeight):
            weight, scale = load_config.exported_device.moe_apply_int8(kernel, device)
        else:
            weight, scale = load_config.exported_device.apply_int8(kernel, device)
        return {self.kernel.name: weight, self.scale.name: scale}
