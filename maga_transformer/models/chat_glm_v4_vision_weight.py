from maga_transformer.config.gpt_init_model_parameters import \
    GptInitModelParameters
from maga_transformer.models.glm_v2_weight import GlmV2WeightInfo
from maga_transformer.models.multimodal.multimodal_mixin import (BaseMultiModalWeightInfo,
                                                      BaseVitWeights)


class ChatGlmV4VisionVitWeights(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "transformer.vision."
        self._ft_prefix = "self.mm_part.vit."


class ChatGlmV4VisionWeightInfo(GlmV2WeightInfo, BaseMultiModalWeightInfo):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        GlmV2WeightInfo.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)

    def _get_weight_info(self):
        glm_4v_weight = super()._get_weight_info()
        glm_4v_weight = self._get_vit_info(glm_4v_weight)
        return glm_4v_weight
