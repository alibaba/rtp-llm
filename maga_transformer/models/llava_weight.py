
import functools
import logging
import torch

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, concat_1, concat_0, identity, zeros, transpose, sp_id
from maga_transformer.models.llama_weight import LlamaWeightInfo
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class LlavaVitWeights:
    vit_proj_w = 'model.mm_projector.{i}.weight'
    vit_proj_b = 'model.mm_projector.{i}.bias'

class LlavaWeightInfo(LlamaWeightInfo):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        super().__init__(config, tp_size, tp_rank)
        self.proj_layers = config.vit_related_params["proj_layers"]
        self.vit_layer_id_interval = config.vit_related_params["vit_layer_id_interval"]
    
    def _get_weight_info(self):
        llava_weight = super()._get_weight_info()

        for i in range(self.proj_layers):
            w = LlavaVitWeights.vit_proj_w.format(i = i * self.vit_layer_id_interval)
            b = LlavaVitWeights.vit_proj_b.format(i = i * self.vit_layer_id_interval)
            llava_weight.weights.append(WeightInfo(w, [CkptWeightInfo(w, identity)], identity))
            llava_weight.weights.append(WeightInfo(b, [CkptWeightInfo(b, identity)], identity))
            llava_weight.tp_strategy[w] = sp_id
            llava_weight.tp_strategy[b] = sp_id

        return llava_weight