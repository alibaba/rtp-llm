from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.models.qwen_v2_audio.processor import Processor
from rtp_llm.utils.util import get_config_from_path


class QWenV2AudioWeightinfo(QWenV2Weight, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights, **kwargs):
        QWenV2Weight.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)


class QWenV2Audio(QWenV2, MultiModalMixin):
    def _init_multimodal(
        self,
        mm_model_config,
        vit_config: VitConfig,
    ):
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = Processor(
            self.model_config.mm_related_params, self.model_config.ckpt_path
        )
        self.model_config.mm_related_params.vit_weights = BaseVitWeights(
            {
                "multi_modal_projector": self.mm_part.multi_modal_projector,
                "audio_tower": self.mm_part.audio_tower,
            },
            with_prefix=True,
        )
        self.model_config.mm_related_params.vit_weights._ckpt_prefix = ""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.qwen import create_qwen_v2_audio_config

        config = create_qwen_v2_audio_config(ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return QWenV2AudioWeightinfo


register_model("qwen_v2_audio", QWenV2Audio)
