from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_audio.processor import Processor


class Qwen2_AudioMixin(BaseMultiModalMixin):
    def _init_multimodal(self):

        self.mm_part = Processor(self.mm_related_params)
        self.mm_related_params.vit_weights = BaseVitWeights(
            {
                "multi_modal_projector": self.mm_part.multi_modal_projector,
                "audio_tower": self.mm_part.audio_tower,
            },
            with_prefix=True,
        )
        self.mm_related_params.vit_weights._ckpt_prefix = ""

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return Processor(mm_related_params).audio_tower


register_multimodal_mixin(["qwen_v2_audio"], Qwen2_AudioMixin)
