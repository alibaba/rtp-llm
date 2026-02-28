import json
import os
from typing import Any, Dict, List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen3NextWeight, Qwen35MoeWeight
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.ops import HybridAttentionType


class Qwen3NextBase(BaseModel):
    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        from rtp_llm.models_py.utils.arch import is_cuda

        if not is_cuda():
            raise RuntimeError("Qwen3Next is only supported in cuda arch")
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextModel

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

        config_json = cls._preprocess_config_json(config_json)

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(config_json, config)
        cls._parse_rope_config(config_json, config)
        cls._parse_normalization_config(config_json, config)
        cls._parse_moe_config(config_json, config)
        cls._parse_hybrid_attention_config(config_json, config)
        cls._parse_linear_attention_config(config_json, config)

        return config

    @classmethod
    def _preprocess_config_json(cls, config_json: dict) -> dict:
        return config_json

    @classmethod
    def _parse_basic_config(cls, config_json: dict, config: ModelConfig):
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.attn_config.size_per_head = config_json["head_dim"]
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.max_seq_len = config_json["max_position_embeddings"]

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        raise NotImplementedError("Subclass must implement _parse_rope_config")

    @classmethod
    def _parse_normalization_config(cls, config_json: dict, config: ModelConfig):
        config.layernorm_eps = config_json["rms_norm_eps"]
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.qk_norm = True
        config.activation_type = "SiGLU"

    @classmethod
    def _parse_moe_config(cls, config_json: Dict[str, Any], config: ModelConfig):
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["num_experts"]
        config.moe_inter_size = config_json["moe_intermediate_size"]
        config.inter_size = config_json["shared_expert_intermediate_size"]
        config.has_moe_norm = config_json.get("norm_topk_prob", True)  # 默认 True
        config.moe_style = 2  # shared + expert

        moe_step = config_json.get("decoder_sparse_step", 1)
        moe_layer_index = []
        for i in range(config.num_layers):
            if (i + 1) % moe_step == 0:
                moe_layer_index.append(i)
        config.moe_layer_index = moe_layer_index

    @classmethod
    def _parse_hybrid_attention_config(cls, config_json: dict, config: ModelConfig):
        attention_step = config_json["full_attention_interval"]
        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for i in range(config.num_layers):
            if (i + 1) % attention_step == 0:
                hybrid_layer_types.append(HybridAttentionType.NONE)
            else:
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

    @classmethod
    def _parse_linear_attention_config(cls, config_json: dict, config: ModelConfig):
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


class Qwen3Next(Qwen3NextBase):
    @staticmethod
    def get_weight_cls():
        return Qwen3NextWeight

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = config_json["rope_theta"]
        config.partial_rotary_factor = config_json["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )


class Qwen35Moe(Qwen3NextBase):
    @staticmethod
    def get_weight_cls():
        return Qwen35MoeWeight

    @classmethod
    def _preprocess_config_json(cls, config_json: dict) -> dict:
        config_json = config_json["text_config"]
        return config_json

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        # rope_parameters 格式
        rope_parameters = config_json["rope_parameters"]
        # mrope_interleaved = rope_parameters["mrope_interleaved"]
        # assert mrope_interleaved, "mrope_interleaved should be True"
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = rope_parameters["rope_theta"]
        config.partial_rotary_factor = rope_parameters["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )
        # mrope_section = rope_parameters["mrope_section"]
        # config.attn_config.rope_config.index_factor = len(mrope_section)
        # config.attn_config.rope_config.mrope_dim1 = mrope_section[0]
        # config.attn_config.rope_config.mrope_dim2 = mrope_section[1]
        # config.attn_config.rope_config.mrope_dim3 = mrope_section[2]
        # config.mm_model_config.mm_position_ids_style = 2


register_model("qwen3_next", Qwen3Next, ["Qwen3NextForCausalLM"])
register_model("qwen35_moe", Qwen35Moe, ["Qwen3_5MoeForConditionalGeneration"])
