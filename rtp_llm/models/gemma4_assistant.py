import json
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.gemma4 import Gemma4
from rtp_llm.models.gemma4_assistant_weight import Gemma4AssistantWeightInfo
from rtp_llm.ops import HybridAttentionType, RopeStyle


class Gemma4Assistant(BaseModel):
    @staticmethod
    def get_weight_cls():
        return Gemma4AssistantWeightInfo

    @classmethod
    def _load_config_json(cls, ckpt_path: str) -> Dict[str, Any]:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            return json.load(reader)

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        source = cls._load_config_json(ckpt_path)
        text = source["text_config"]
        layer_types = list(text["layer_types"])
        num_layers = int(text["num_hidden_layers"])
        if len(layer_types) != num_layers:
            raise ValueError(
                "assistant layer_types length does not match num_hidden_layers"
            )
        if int(text.get("num_kv_shared_layers", 0)) != num_layers:
            raise ValueError("all Gemma4 assistant layers must share target KV")
        if text.get("enable_moe_block") is not False:
            raise ValueError("Gemma4 assistant must use dense MLP layers")
        if text.get("use_double_wide_mlp") is not False:
            raise ValueError("Gemma4 assistant does not support double-wide MLP")

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.num_layers = num_layers
        config.hidden_size = int(text["hidden_size"])
        config.vocab_size = int(text["vocab_size"])
        config.inter_size = int(text["intermediate_size"])
        config.attn_config.head_num = int(text["num_attention_heads"])
        config.attn_config.kv_head_num = int(text["num_key_value_heads"])
        config.attn_config.size_per_head = int(text["head_dim"])
        config.attn_config.sliding_window = int(text["sliding_window"])
        config.attn_config.tokens_per_block = 128
        sliding_rope = text["rope_parameters"]["sliding_attention"]
        config.attn_config.rope_config.style = RopeStyle.Base
        config.attn_config.rope_config.base = int(sliding_rope["rope_theta"])
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = [
            (
                HybridAttentionType.NONE
                if layer_type == "full_attention"
                else HybridAttentionType.SLIDING_WINDOW
            )
            for layer_type in layer_types
        ]
        config.layernorm_eps = float(text["rms_norm_eps"])
        config.norm_type = "rmsnorm"
        config.activation_type = "geglu"
        config.has_post_decoder_layernorm = True
        config.qk_norm = True
        config.tie_word_embeddings = bool(source.get("tie_word_embeddings", True))
        config.enable_fp32_lm_head = False
        config.max_seq_len = int(text["max_position_embeddings"])
        config.is_mtp = True
        config.shares_target_kv = True
        config.config_dtype = text.get("torch_dtype") or text.get("dtype")
        config.special_tokens.bos_token_id = int(text.get("bos_token_id", 2))
        config.special_tokens.eos_token_id = int(text.get("eos_token_id", 1))
        config.special_tokens.pad_token_id = int(text.get("pad_token_id", 0))

        generation_path = os.path.join(ckpt_path, "generation_config.json")
        if os.path.exists(generation_path):
            with open(generation_path) as reader:
                generation = json.load(reader)
            config.gen_num_per_cycle = int(generation.get("num_assistant_tokens", 6))

        full_rope = text["rope_parameters"]["full_attention"]
        config.mm_related_params.config.update(
            {
                "assistant_backbone_hidden_size": int(source["backbone_hidden_size"]),
                "assistant_all_kv_shared": True,
                "full_layer_kv_head_num": int(text["num_global_key_value_heads"]),
                "full_layer_size_per_head": int(text["global_head_dim"]),
                "full_layer_rope_theta": float(full_rope["rope_theta"]),
                "full_layer_partial_rotary_factor": float(
                    full_rope["partial_rotary_factor"]
                ),
            }
        )
        return config

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        Gemma4._post_build_model_config(model_config)

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.gemma4_assistant import (
            Gemma4AssistantRuntimeModel,
        )

        self.py_model = Gemma4AssistantRuntimeModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return False


register_model(
    "gemma4_assistant",
    Gemma4Assistant,
    ["Gemma4AssistantForCausalLM"],
)
