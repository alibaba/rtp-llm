import json
import logging
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.omni.models.qwen2_5_omni.audio_processor import Processor
from rtp_llm.utils.util import get_config_from_path

logger = logging.getLogger(__name__)


class Qwen2_5OmniThinkerWeight(QWenV2Weight, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights=None, **kwargs):
        QWenV2Weight.__init__(self, prefix="thinker.", **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)


class Qwen2_5OmniThinker(MultiModalMixin, QWenV2):
    def _init_multimodal(
        self,
        mm_model_config,
        vit_config: VitConfig,
    ):
        self.mm_part = Processor(
            self.model_config.mm_related_params, self.model_config.ckpt_path
        )
        self.model_config.mm_related_params.vit_weights = BaseVitWeights(
            {"audio_tower": self.mm_part.audio_tower},
            with_prefix=True,
        )
        self.model_config.mm_related_params.vit_weights._ckpt_prefix = "thinker."

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.special_tokens.bos_token_id = 151644
        config.special_tokens.eos_token_id = 151645
        config.special_tokens.stop_words_id_list = [[151645], [151644]]

        cls._from_hf(config, ckpt_path)
        return config

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as f:
            root_config = json.load(f)

        thinker_config = root_config.get("thinker_config", root_config)
        text_config = thinker_config.get("text_config", thinker_config)

        QWenV2._from_config_json(config, text_config)

        audio_token_index = thinker_config.get("audio_token_index", 151646)
        config.mm_model_config.mm_sep_tokens = [[audio_token_index]]
        config.config_dtype = text_config.get("torch_dtype", None)

    @staticmethod
    def get_weight_cls():
        return Qwen2_5OmniThinkerWeight


register_model(
    "qwen2_5_omni_thinker",
    Qwen2_5OmniThinker,
    ["Qwen2OmniNaViTThinkerForConditionalGeneration"],
)
