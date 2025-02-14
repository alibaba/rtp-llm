
from maga_transformer.models.qwen import QWenWeight
from maga_transformer.models.multimodal.multimodal_mixin import BaseVitWeights, BaseMultiModalWeightInfo

class QwenVLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "transformer.visual."
        self._ft_prefix = "self.mm_part.vit."

class QWenVLWeightInfo(QWenWeight, BaseMultiModalWeightInfo):
    
    def __init__(self, config, tp_size, tp_rank):
        QWenWeight.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)
    
    def _get_weight_info(self):
        qwen_vl_weight = super()._get_weight_info()
        qwen_vl_weight = self._get_vit_info(qwen_vl_weight)
        return qwen_vl_weight
