
import functools
import logging
import torch

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, concat_1, concat_0, identity, zeros, transpose, sp_id
from maga_transformer.models.qwen import QWenWeight
from maga_transformer.models.multimodal_mixin import BaseVitWeights, BaseMultiModalWeightInfo

class QwenVLVitWeight(BaseVitWeights):
    @property
    def ckpt_prefix(self) -> str:
        return "transformer.visual."

    @property
    def ft_prefix(self) -> str:
        return "self.visual.vit."

class QWenVLWeightInfo(QWenWeight, BaseMultiModalWeightInfo):
    
    def __init__(self, config, tp_size, tp_rank):
        QWenWeight.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)
    
    def _get_weight_info(self):
        qwen_vl_weight = super()._get_weight_info()
        self._get_vit_info(qwen_vl_weight)
        return qwen_vl_weight
