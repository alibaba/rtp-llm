
import functools
import logging
import torch

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, concat_1, concat_0, identity, zeros, transpose, sp_id
from maga_transformer.models.qwen import QWenWeight

class QWenVLWeightInfo(QWenWeight):
    def __init__(self, config, tp_size, tp_rank):
        super().__init__(config, tp_size, tp_rank)
        self.layers = config.vit_related_params["layers"]
        self.weights = config.vit_related_params["weights"]
    
    def _get_weight_info(self):
        qwen_vl_weight = super()._get_weight_info()
        weights_names = self.weights.weight_names
        ft_prefix = self.weights.ft_prefix
        hf_prefix = self.weights.hf_prefix

        for w in weights_names:
            w_name = hf_prefix + w
            qwen_vl_weight.weights.append(WeightInfo(w_name, [CkptWeightInfo(w_name, identity)], identity))
            qwen_vl_weight.tp_strategy[w_name] = sp_id

        return qwen_vl_weight