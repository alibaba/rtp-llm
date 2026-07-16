from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
)
from rtp_llm.models.qwen import QWenWeight


class QwenVLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "transformer.visual."
        self._ft_prefix = "self.mm_part.vit."


class QWenVLWeightInfo(QWenWeight, BaseMultiModalWeightInfo):

    def __init__(self, vit_weights, **kwargs):
        QWenWeight.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)
