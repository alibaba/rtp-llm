from rtp_llm.models.llama_weight import LlamaWeightInfo
from rtp_llm.multimodal.multimodal_mixin import BaseMultiModalWeightInfo
from rtp_llm.utils.model_weight import W


class LlavaWeightInfo(LlamaWeightInfo, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights, **kwargs):
        LlamaWeightInfo.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)

    def _get_weight_info(self):
        llava_weight = super()._get_weight_info()

        # for llava-next
        for weight in llava_weight.layer_weights:
            if weight.name == W.attn_o_b:
                llava_weight.layer_weights.remove(weight)
                break

        return llava_weight
