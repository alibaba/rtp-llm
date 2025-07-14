from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.models.llama_weight import LlamaWeightInfo
from rtp_llm.models.multimodal.multimodal_mixin import BaseMultiModalWeightInfo
from rtp_llm.utils.model_weight import W


class LlavaWeightInfo(LlamaWeightInfo, BaseMultiModalWeightInfo):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        LlamaWeightInfo.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)

    def _get_weight_info(self):
        llava_weight = ModelWeightInfo(layer_weights=[], weights=[])
        llava_weight = super()._get_weight_info()

        # for llava-next
        for weight in llava_weight.layer_weights:
            if weight.name == W.attn_o_b:
                llava_weight.layer_weights.remove(weight)
                break

        llava_weight = self._get_vit_info(llava_weight)
        return llava_weight
