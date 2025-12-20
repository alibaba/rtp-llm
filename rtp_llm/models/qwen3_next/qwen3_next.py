import json
import os
from typing import List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen3NextWeight
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen_next import Qwen3NextModel
from rtp_llm.ops import HybridAttentionType


class Qwen3Next(BaseModel):
    @staticmethod
    def get_weight_cls():
        return Qwen3NextWeight

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        self.py_model = Qwen3NextModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return True

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        # Basic model structure
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.attn_config.size_per_head = config_json["head_dim"]
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.max_seq_len = config_json["max_position_embeddings"]

        # RoPE configuration
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = config_json["rope_theta"]
        config.partial_rotary_factor = config_json["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )

        # Normalization
        config.layernorm_eps = config_json["rms_norm_eps"]
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.qk_norm = True

        # Activation
        config.activation_type = "SiGLU"

        # MoE configuration
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["num_experts"]
        config.moe_inter_size = config_json["moe_intermediate_size"]
        config.inter_size = config_json["shared_expert_intermediate_size"]
        config.has_moe_norm = config_json.get("norm_topk_prob", False)
        config.moe_style = 2  # shared + expert

        # MoE layer indices
        moe_step = config_json["decoder_sparse_step"]
        moe_layer_index = []
        for i in range(config.num_layers):
            if (i + 1) % moe_step == 0:
                moe_layer_index.append(i)
        config.moe_layer_index = moe_layer_index

        # Hybrid attention configuration
        attention_step = config_json["full_attention_interval"]
        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for i in range(config.num_layers):
            if (i + 1) % attention_step == 0:
                hybrid_layer_types.append(HybridAttentionType.NONE)
            else:
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

        # Linear attention configuration
        config.linear_attention_config.linear_conv_kernel_dim = config_json[
            "linear_conv_kernel_dim"
        ]
        config.linear_attention_config.linear_key_head_dim = config_json[
            "linear_key_head_dim"
        ]
        config.linear_attention_config.linear_num_key_heads = config_json[
            "linear_num_key_heads"
        ]
        config.linear_attention_config.linear_num_value_heads = config_json[
            "linear_num_value_heads"
        ]
        config.linear_attention_config.linear_value_head_dim = config_json[
            "linear_value_head_dim"
        ]

        return config


register_model("qwen3_next", Qwen3Next, ["Qwen3NextForCausalLM"])
