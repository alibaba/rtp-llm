import json
import os

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import (
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
)


class QWenV3Weight(QWenV2Weight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = False


class QwenV3(QWenV2):
    @staticmethod
    def get_weight_cls():
        return QWenV3Weight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.use_qk_norm = True
        return config


register_model("qwen_3", QwenV3, ["Qwen3ForCausalLM"])
register_model("qwen_3_tool", QwenV3)
