import torch.nn as nn

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.omni.models.qwen2_5_omni.audio_processor import Processor


class Qwen2_5OmniMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Processor(self.mm_related_params, self.ckpt_path)
        self.mm_related_params.vit_weights = BaseVitWeights(
            {"audio_tower": self.mm_part.audio_tower},
            with_prefix=True,
        )
        self.mm_related_params.vit_weights._ckpt_prefix = "thinker."

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        ckpt_path = mm_related_params.config["ckpt_path"]
        processor = Processor(mm_related_params, ckpt_path)
        return nn.ModuleList([processor.audio_tower])


register_multimodal_mixin(
    ["qwen2_5_omni_thinker", "qwen2_5_omni"],
    Qwen2_5OmniMixin,
)
