import json
import os
from typing import Any, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2_moe import Qwen2Moe
from rtp_llm.models.qwen_v3_moe import Qwen3Moe


class MiniMaxM3VLWeightInfo(object):
    def __init__(self, **kwargs):
        pass

    def _get_weight_info(self):
        return []


class MiniMax_M3_VL(Qwen3Moe):
    def _create_python_model(self):
        pass

    @staticmethod
    def get_weight_cls():
        return MiniMaxM3VLWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.ckpt_path = ckpt_path
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

        # 填充 vision 和 text MoE 参数
        if "vision_config" in config_json:
            config.mm_related_params.config.update(config_json["vision_config"])

        text_config = config_json.get("text_config", config_json)

        # 填充核心维度参数
        config.hidden_size = text_config.get("hidden_size", 6144)
        config.vocab_size = text_config.get("vocab_size", 200064)
        num_layers = text_config.get(
            "num_layers", text_config.get("num_hidden_layers", 60)
        )
        if hasattr(config, "num_layers"):
            config.num_layers = num_layers
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = num_layers

        # 填充注意力参数
        head_num = text_config.get("num_attention_heads", 64)
        kv_head_num = text_config.get("num_key_value_heads", head_num)
        size_per_head = text_config.get("head_dim", 128)

        config.attn_config.head_num = head_num
        config.attn_config.kv_head_num = kv_head_num
        config.attn_config.size_per_head = size_per_head
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = int(
            text_config.get("rope_theta", config.attn_config.rope_config.base)
        )
        partial_rotary_factor = text_config.get("partial_rotary_factor", 1.0)
        config.attn_config.rope_config.dim = int(
            text_config.get("rotary_dim", size_per_head * partial_rotary_factor)
        )
        config.attn_config.rope_config.max_pos = text_config.get(
            "max_position_embeddings", config.attn_config.rope_config.max_pos
        )
        if hasattr(config, "partial_rotary_factor"):
            config.partial_rotary_factor = partial_rotary_factor
        # Safely fetch MoE parameters using defaults
        config.expert_num = text_config.get(
            "expert_num",
            text_config.get("num_local_experts", text_config.get("num_experts", 128)),
        )
        config.moe_k = text_config.get(
            "moe_k", text_config.get("num_experts_per_tok", 4)
        )
        config.moe_inter_size = text_config.get(
            "moe_intermediate_size", text_config.get("intermediate_size", 3072)
        )
        config.inter_size = text_config.get(
            "shared_expert_intermediate_size",
            text_config.get("shared_intermediate_size", 0),
        )
        config.layernorm_eps = text_config.get("rms_norm_eps", 1e-06)
        config.has_moe_norm = text_config.get("norm_topk_prob", False)
        config.moe_style = 2
        scoring_func = text_config.get("scoring_func", "softmax")
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1
        else:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")
        config.routed_scaling_factor = float(
            text_config.get("routed_scaling_factor", 1.0)
        )
        if hasattr(config, "decoder_sparse_step"):
            config.decoder_sparse_step = text_config.get("decoder_sparse_step", 1)
        return config


register_model("minimax_m3_vl", MiniMax_M3_VL, ["MiniMaxM3VLForConditionalGeneration"])
