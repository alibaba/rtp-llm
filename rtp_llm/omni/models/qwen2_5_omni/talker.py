from typing import Any

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.model_weight import W, CkptWeightInfo, identity
from rtp_llm.utils.util import get_config_from_path


class Qwen25OmniTalkerWeight(QWenV2Weight):
    def __init__(self, **kwargs: Any):
        super().__init__(prefix="talker.", **kwargs)

    def _get_hf_weight_info(self) -> ModelWeightInfo:
        weight_info = super()._get_hf_weight_info()
        for i, w in enumerate(weight_info.weights):
            if w.name == W.lm_head:
                weight_info.weights[i] = AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo("talker.codec_head.weight", identity)],
                    identity,
                )
                break
        return weight_info


class Qwen25OmniTalker(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.max_seq_len = 32768
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False

        config_json = get_config_from_path(ckpt_path)
        if config_json and "talker_config" in config_json:
            talker_cfg = config_json["talker_config"]
            QWenV2._from_config_json(config, talker_cfg)
        else:
            cls._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen25OmniTalkerWeight


register_model("qwen2_5_omni_talker", Qwen25OmniTalker)
