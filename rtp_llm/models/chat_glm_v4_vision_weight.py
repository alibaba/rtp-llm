from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.glm_v2_weight import GlmV2WeightInfo
from rtp_llm.multimodal.multimodal_mixin import BaseMultiModalWeightInfo, BaseVitWeights


class ChatGlmV4VisionVitWeights(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "transformer.vision."
        self._ft_prefix = "self.mm_part.vit."


class ChatGlmV4VisionWeightInfo(GlmV2WeightInfo, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights, **kwargs):
        GlmV2WeightInfo.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)
