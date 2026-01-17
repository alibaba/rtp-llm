import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.llama_weight import GemmaWeightInfo, LlamaWeightInfo
from rtp_llm.ops import (
    DeviceResourceConfig,
    FMHAConfig,
    HWKernelConfig,
    KVCacheConfig,
    ModelSpecificConfig,
    MoeConfig,
    ParallelismConfig,
    RuntimeConfig,
)


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * (
        (int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of
    )


class Llama(BaseModel):
    @staticmethod
    def get_mscale(scale: float):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    @staticmethod
    def get_weight_cls():
        return LlamaWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.llama import create_llama_config

        config = create_llama_config(ckpt_path)
        return config

    @staticmethod
    def from_huggingface(config: PyModelConfig, config_json: Dict[str, Any]):
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.hidden_size = config_json["hidden_size"]
        config.attn_config.size_per_head = (
            config_json["hidden_size"] // config_json["num_attention_heads"]
        )
        config.attn_config.size_per_head = config_json.get(
            "head_dim", config.attn_config.size_per_head
        )
        config.num_layers = config_json["num_hidden_layers"]
        config.max_seq_len = config_json.get("max_sequence_length", 2048)
        config.vocab_size = config_json["vocab_size"]
        config.layernorm_eps = config_json.get(
            "rms_norm_eps", config_json.get("layer_norm_eps", 1e-05)
        )
        config.inter_size = config_json["intermediate_size"]
        config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        rope_scaling = config_json.get("rope_scaling")
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            if rope_type == "linear":
                config.attn_config.rope_config.scale = rope_scaling["factor"]
                config.attn_config.rope_config.max_pos = config_json.get(
                    "max_position_embeddings", 2048
                )
            elif rope_type == "dynamic":
                config.attn_config.rope_config.style = 3
            elif rope_type == "yarn":
                config.attn_config.rope_config.style = 5
                config.attn_config.rope_config.scale = rope_scaling["factor"]
                config.attn_config.rope_config.factor1 = rope_scaling.get(
                    "beta_slow", 1
                )
                config.attn_config.rope_config.factor2 = rope_scaling.get(
                    "beta_fast", 32
                )
                config.attn_config.rope_config.max_pos = rope_scaling[
                    "original_max_position_embeddings"
                ]
                config.attn_config.rope_config.mscale = Llama.get_mscale(
                    config.attn_config.rope_config.scale
                )
            elif rope_type == "llama3":
                config.attn_config.rope_config.style = 6
                config.attn_config.rope_config.scale = rope_scaling["factor"]
                config.attn_config.rope_config.factor1 = rope_scaling["low_freq_factor"]
                config.attn_config.rope_config.factor2 = rope_scaling[
                    "high_freq_factor"
                ]
                config.attn_config.rope_config.max_pos = rope_scaling[
                    "original_max_position_embeddings"
                ]
            else:
                raise Exception(f"unsupport rope_scaling {rope_scaling}")
        eos_token_id = config_json.get("eos_token_id", 0)
        if isinstance(eos_token_id, list):
            config.special_tokens.eos_token_id = eos_token_id[0]
        else:
            config.special_tokens.eos_token_id = eos_token_id
        config.attn_config.use_logn_attn = config_json.get("use_logn_attn", False)
        config.config_dtype = config_json.get("torch_dtype", None)

    @staticmethod
    def from_params(config: PyModelConfig, params_json: Dict[str, Any]):
        config.attn_config.head_num = params_json["n_heads"]
        config.attn_config.kv_head_num = params_json.get(
            "n_kv_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = params_json["dim"] // params_json["n_heads"]
        config.num_layers = params_json["n_layers"]
        config.max_seq_len = 2048
        config.vocab_size = 32000
        config.layernorm_eps = params_json["norm_eps"]
        config.inter_size = compute_intermediate_size(
            params_json["dim"],
            params_json.get("ffn_dim_multiplier", 1),
            params_json["multiple_of"],
        )
        config.special_tokens.bos_token_id = 1
        config.special_tokens.eos_token_id = 2
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.tie_word_embeddings = params_json.get("tie_word_embeddings", False)
        config.config_dtype = params_json.get("torch_dtype", None)
        return config


class Baichuan(Llama):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.llama import create_baichuan_config

        config = create_baichuan_config(ckpt_path)
        return config


class Baichuan2(Baichuan):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.llama import create_baichuan2_config

        config = create_baichuan2_config(ckpt_path)
        return config


class Gemma(Llama):
    def __init__(
        self,
        model_config: PyModelConfig,
        parallelism_config: ParallelismConfig,
        model_specific_config: ModelSpecificConfig,
        hw_kernel_config: HWKernelConfig,
        kv_cache_config: KVCacheConfig,
        fmha_config: FMHAConfig,
        moe_config: MoeConfig,
        runtime_config: RuntimeConfig,
        device_resource_config: DeviceResourceConfig,
        vit_config: Optional[Any] = None,
        merge_lora: bool = False,
    ):
        if fmha_config.enable_open_source_fmha:
            logging.warn(
                "opensource fmha does not support head dim 256, thus disabled for gemma model"
            )
            os.environ["ENABLE_OPENSOURCE_FMHA"] = "OFF"
        super().__init__(
            model_config=model_config,
            parallelism_config=parallelism_config,
            model_specific_config=model_specific_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            fmha_config=fmha_config,
            moe_config=moe_config,
            runtime_config=runtime_config,
            device_resource_config=device_resource_config,
            vit_config=vit_config,
            merge_lora=merge_lora,
        )

    @staticmethod
    def get_weight_cls():
        return GemmaWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.llama import create_gemma_config

        config = create_gemma_config(ckpt_path)
        return config


class Cohere(Llama):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.llama import create_llama_config

        config = create_llama_config(ckpt_path)

        return config


register_model("internlm", Llama, ["InternLMForCausalLM"])
register_model("internlm2", Llama, ["InternLM2ForCausalLM"])
register_model("llama", Llama, ["LlamaForCausalLM", "YiForCausalLM"])
register_model("xverse", Llama, ["XverseForCausalLM"])
register_model("aquila", Llama, ["AquilaModel"])
register_model("mistral", Llama, ["MistralForCausalLM"])
register_model("baichuan", Baichuan, ["BaichuanForCausalLM"])
register_model("baichuan2", Baichuan2)
register_model("gemma", Gemma, ["GemmaForCausalLM"])
register_model("cohere", Cohere, ["CohereForCausalLM"])
