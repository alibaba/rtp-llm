import json
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.internvl_vit import InternVLImageEmbedding
from rtp_llm.models.internvl_weight import InternVLVitWeight, InternVLWeightInfo
from rtp_llm.models.llama import Llama
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models.qwen_v2 import QWenV2


class InternVL(BaseModel, MultiModalMixin):
    def _init_multimodal(
        self,
        mm_model_config: Any,  # MMModelConfig
        vit_config: VitConfig,
    ):
        # mm_related_params is in model_config, not mm_model_config
        mm_related_params = self.model_config.mm_related_params
        self.mm_part = InternVLImageEmbedding(mm_related_params)
        mm_related_params.vit_weights = InternVLVitWeight(
            {"vision_model": self.mm_part.vision_model, "mlp1": self.mm_part.mlp1}, True
        )
        mm_model_config.mm_sep_tokens = [
            [self.tokenizer.encode("<img>")[0], self.tokenizer.encode("</img>")[0]]
        ]

    @staticmethod
    def get_weight_cls():
        return InternVLWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False

        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                llm_config = config_json["llm_config"]
                if llm_config["architectures"][0] == "Qwen2ForCausalLM":
                    QWenV2._from_config_json(config, llm_config)
                elif (
                    llm_config["architectures"][0] == "InternLM2ForCausalLM"
                    or llm_config["architectures"][0] == "LlamaForCausalLM"
                ):
                    Llama.from_huggingface(config, llm_config)
                else:
                    raise Exception("unknown language model architecture")
                InternVL._init_vit_params(config, config_json)
        else:
            raise Exception("no config.json found")
        config.special_tokens.stop_words_str_list = ["<|im_end|>"]
        assert (
            config.attn_config.head_num > 0
            and config.attn_config.kv_head_num > 0
            and config.attn_config.size_per_head > 0
            and config.num_layers > 0
            and config.inter_size > 0
        ), "error config"
        config.mm_related_params.special_tokens.update({"default_mm_token": "<image>"})
        return config

    @staticmethod
    def _init_vit_params(config: ModelConfig, config_json: Dict[str, Any]):
        config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.config["select_layer"] = config_json["select_layer"]
        config.mm_related_params.config["llm_hidden_size"] = config_json["llm_config"][
            "hidden_size"
        ]
        config.mm_related_params.config["downsample_ratio"] = config_json[
            "downsample_ratio"
        ]
        config.mm_related_params.config["ps_version"] = config_json["ps_version"]


register_model("internvl", InternVL, ["InternVLChatModel"])
