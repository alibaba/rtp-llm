"""Whisper multimodal mixin: registers the audio encoder tower (A0+A1).

The tower weights load through BaseVitWeights with the default ckpt prefix
"model." plus the module name "encoder" — checkpoint keys are exactly
model.encoder.<state_dict_name> for openai/whisper-large-v3.
"""

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.whisper.processor import Processor
from rtp_llm.config.py_config_modules import VitConfig


class WhisperMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Processor(self.mm_related_params)
        self.mm_related_params.vit_weights = BaseVitWeights(
            {
                "encoder": self.mm_part.encoder,
            },
            with_prefix=True,
        )
        # ckpt prefix stays the default "model." so that state_dict name
        # "encoder.conv1.weight" resolves to "model.encoder.conv1.weight".

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        processor = Processor(mm_related_params)
        return processor.encoder


register_multimodal_mixin(["whisper"], WhisperMixin)
