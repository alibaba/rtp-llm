from typing import List, Optional

from rtp_llm.config.gpt_init_model_parameters import (
    GptInitModelParameters,
    HybridAttentionType,
)
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen3NextWeight
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.utils.util import check_get_config_from_path


class Qwen3Next(BaseModel):
    @staticmethod
    def get_weight_cls():
        return Qwen3NextWeight

    def _create_python_model(self) -> Optional[GptModelBase]:
        from rtp_llm.models_py.model_desc.qwen_next import Qwen3NextModel

        self.py_model = Qwen3NextModel(self.config, self.weight)

    def support_cuda_graph(self) -> bool:
        return True

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_json = check_get_config_from_path(ckpt_path)

        config = GptInitModelParameters(
            head_num=config_json["num_attention_heads"],
            head_num_kv=config_json["num_key_value_heads"],
            size_per_head=config_json["head_dim"],
            layer_num=config_json["num_hidden_layers"],
            inter_size=config_json["intermediate_size"],
            vocab_size=config_json["vocab_size"],
            max_seq_len=config_json["max_position_embeddings"],
        )

        config.hidden_size = config_json["hidden_size"]
        # TODO fix rotary embedding config
        config.rotary_embedding_style = 1
        config.rotary_embedding_base = config_json["rope_theta"]
        config.partial_rotary_factor = config_json["partial_rotary_factor"]
        config.rotary_embedding_dim = int(
            config.size_per_head * config.partial_rotary_factor
        )
        config.layernorm_eps = config_json["rms_norm_eps"]

        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["num_experts"]
        config.moe_inter_padding_size = config_json["moe_intermediate_size"]
        config.inter_size = config_json["shared_expert_intermediate_size"]
        config.has_moe_norm = config_json.get("norm_topk_prob", False)
        config.moe_style = 2
        config.activation_type = "SiGLU"
        # Moe config
        moe_step = config_json["decoder_sparse_step"]
        moe_layer_index = []
        for i in range(config.layer_num):
            if (i + 1) % moe_step == 0:
                moe_layer_index.append(i)
        config.moe_layer_index = moe_layer_index

        # hybrid attention config
        attention_step = config_json["full_attention_interval"]
        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for i in range(config.layer_num):
            if (i + 1) % attention_step == 0:
                hybrid_layer_types.append(HybridAttentionType.NONE)
            else:
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types
        # linear attention config
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
