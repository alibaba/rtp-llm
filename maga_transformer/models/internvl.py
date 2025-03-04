import os
import logging
import json
import math
import torch

from typing import Any, Dict, List

from transformers import AutoTokenizer
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.qwen_v2 import QWenV2
from maga_transformer.models.llama import Llama
from maga_transformer.models.base_model import BaseModel
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.models.internvl_weight import InternVLVitWeight, InternVLWeightInfo
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.internvl_vit import InternVLImageEmbedding

class InternVLTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def encode(self, prompt: str, **kwargs):
        prompt_slices = prompt.split("<image>")
        new_prompt = prompt_slices[0]
        for slice in prompt_slices[1:]:
            new_prompt += "<img></img>" + slice
        return self.tokenizer.encode(new_prompt, add_special_tokens=False, **kwargs)

    def decode(self, token_id: List[int], **kwargs):
        return self.tokenizer.decode(token_id, **kwargs)

class InternVL(BaseModel, MultiModalMixin):
    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = InternVLImageEmbedding(config)
        config.mm_related_params.vit_weights = InternVLVitWeight({"vision_model": self.mm_part.vision_model,
                                                                    "mlp1": self.mm_part.mlp1}, True)
        config.mm_sep_tokens = [[self.tokenizer.encode("<img>")[0], self.tokenizer.encode("</img>")[0]]]


    @staticmethod
    def get_weight_cls():
        return InternVLWeightInfo

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return InternVLTokenizer(config.tokenizer_path)

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            head_num_kv=0,
            size_per_head=0,
            layer_num=0,
            inter_size=0,
            vocab_size=0,
            max_seq_len=0,
            ckpt_path=ckpt_path,
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            activation_type='SiGLU',
            has_pre_decoder_layernorm=False,
            has_post_decoder_layernorm=True,
            norm_type='rmsnorm'
            )

        config_path = os.path.join(ckpt_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                llm_config = config_json["llm_config"]
                if llm_config["architectures"][0] == "Qwen2ForCausalLM":
                    QWenV2._from_config_json(config, llm_config)
                elif llm_config["architectures"][0] == "InternLM2ForCausalLM" or \
                    llm_config["architectures"][0] == "LlamaForCausalLM":
                    Llama.from_huggingface(config, llm_config)
                else:
                    raise Exception("unknown language model architecture")
                InternVL._init_vit_params(config, config_json)
        else:
            raise Exception("no config.json found")
        config.special_tokens.stop_words_str_list = ["<|im_end|>"]
        assert config.head_num > 0 and config.head_num_kv > 0 and config.size_per_head > 0 and config.layer_num > 0 and config.inter_size > 0, "error config"
        config.mm_related_params.special_tokens.update({'default_mm_token': '<image>'})
        return config

    @classmethod
    def _update_config(cls, config: GptInitModelParameters):
        if config.tokenizer_path:
            config.special_tokens.stop_words_id_list = [cls.get_tokenizer(config).encode("<|im_end|>")]

    @staticmethod
    def _init_vit_params(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.config["select_layer"] = config_json["select_layer"]
        config.mm_related_params.config["llm_hidden_size"] = config_json["llm_config"]["hidden_size"]
        config.mm_related_params.config["downsample_ratio"] = config_json["downsample_ratio"]
        config.mm_related_params.config["ps_version"] = config_json["ps_version"]

register_model("internvl", InternVL, ["InternVLChatModel"])
