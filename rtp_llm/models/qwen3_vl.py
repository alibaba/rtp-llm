import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.library as tl
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLVisionModel

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v3 import QwenV3, QWenV3Weight
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3vl import Qwen3VLModel

if not hasattr(tl, "wrap_triton"):

    def wrap_triton(fn):
        return fn

    tl.wrap_triton = wrap_triton


class QWen3_VL(QwenV3):
    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        quant_config = self.model_config.quant_config
        self.py_model = Qwen3VLModel(
            model_config,
            parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=quant_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return QWenV3Weight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config.qk_norm = True
        cls._from_hf(config, ckpt_path)
        return config

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")

        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
        QWen3_VL._from_config_json(config, config_json)
        return config

    @staticmethod
    def _from_config_json(config: ModelConfig, config_json: Dict[str, Any]):
        config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
        config.mm_model_config.mm_sep_tokens = [
            [config_json["vision_start_token_id"], config_json["vision_end_token_id"]]
        ]

        config_json = config_json["text_config"]
        config.inter_size = config_json["intermediate_size"]
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = (
            int(config_json.get("head_dim"))
            if "head_dim" in config_json
            else config_json["hidden_size"] // config.attn_config.head_num
        )
        if config_json.get("hidden_size") is not None:
            config.hidden_size = config_json["hidden_size"]
        config.num_layers = config_json["num_hidden_layers"]
        config.attn_config.rope_config.base = config_json.get(
            "rope_theta", config.attn_config.rope_config.base
        )
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)

        config.attn_config.rope_config.style = 7
        config.attn_config.rope_config.style = 7
        mrope_section = config_json["rope_scaling"].get("mrope_section", [16, 24, 24])
        config.attn_config.rope_config.index_factor = len(mrope_section)
        config.attn_config.rope_config.mrope_dim1 = mrope_section[0]
        config.attn_config.rope_config.mrope_dim2 = mrope_section[1]
        config.attn_config.rope_config.mrope_dim3 = mrope_section[2]
        config.mm_model_config.mm_position_ids_style = 2

        config.mm_related_params.config["ckpt_path"] = config.ckpt_path
        config.mm_model_config.is_multimodal = True


register_model("qwen3_vl", QWen3_VL, ["Qwen3VLForConditionalGeneration"])
