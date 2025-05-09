import os
import json

from maga_transformer.models.qwen_v2 import QWenV2, QWenV2Weight
from maga_transformer.utils.model_weight import W, CkptWeightInfo, identity, transpose, stack_, stack_moe_w1
from maga_transformer.model_loader.ffn_weight import FfnConfig, MoeConfig, MoeAtomicWeight, MoeWeight
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_factory_register import register_model

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
    
register_model('qwen_3', QwenV3, ["Qwen3ForCausalLM"])
register_model("qwen_3_tool", QwenV3)
