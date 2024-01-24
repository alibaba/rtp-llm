
import functools
import logging
import torch
import re
from typing import Dict, Any, List

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, concat_1, concat_0, identity, zeros, transpose, sp_id
from maga_transformer.models.llama_weight import LlamaWeightInfo
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class LlavaVitWeights:
    def __init__(self, config: Dict[str, Any]):
        self.projector_type = config.get('mm_projector_type', 'linear')
        self.proj_layer_num = config['proj_layers']
        self.vit_layer_id_interval = config['vit_layer_id_interval']
        self.ckpt_prefix = self._get_ckpt_prefix()
        self.ft_prefix = 'self.visual.'
        self.weight_names = self._get_vit_params()
    
    def _get_ckpt_prefix(self):
        return 'model.'
    
    def _get_vit_params(self):
        weights_list: List[str] = []
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)

        if self.projector_type == 'linear':
            weights_list.append('mm_projector.weight')
            weights_list.append('mm_projector.bias')
        elif mlp_gelu_match:
            for i in range(0, self.proj_layer_num):
                weights_list.append(f'mm_projector.{i * self.vit_layer_id_interval}.weight')
                weights_list.append(f'mm_projector.{i * self.vit_layer_id_interval}.bias')
        
        return weights_list

class LlavaWeightInfo(LlamaWeightInfo):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        super().__init__(config, tp_size, tp_rank)
        self.vit_weights = config.vit_related_params["weights"]
    
    def _get_weight_info(self):
        llava_weight = super()._get_weight_info()
        weight_names = self.vit_weights.weight_names
        ckpt_prefix = self.vit_weights.ckpt_prefix

        for w in weight_names:
            w_name = ckpt_prefix + w
            llava_weight.weights.append(WeightInfo(w_name, [CkptWeightInfo(w_name, identity)], identity))
            llava_weight.tp_strategy[w_name] = sp_id

        return llava_weight