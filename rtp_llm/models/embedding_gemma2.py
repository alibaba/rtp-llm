"""EmbeddingGemma 2 text encoder, using the standard model/weight lifecycle."""

import json
import math
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.embedding_gemma2_weight import EmbeddingGemma2WeightInfo
from rtp_llm.ops import TaskType


class EmbeddingGemma2(BaseModel):
    @staticmethod
    def get_weight_cls():
        return EmbeddingGemma2WeightInfo

    @staticmethod
    def _validate_text_config(text):
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "hidden_size_per_layer_input",
            "embedding_dim",
        ):
            if type(text.get(name)) is not int or text[name] <= 0:
                raise ValueError(f"EmbeddingGemma2 {name} must be a positive integer")
        if text.get("attention_bias", False):
            raise ValueError("EmbeddingGemma2 attention_bias is unsupported")
        if text.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("EmbeddingGemma2 requires gelu_pytorch_tanh")
        eps = float(text.get("rms_norm_eps", 1e-6))
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("EmbeddingGemma2 rms_norm_eps must be finite and positive")
        return eps

    @classmethod
    def _parse_text_config(cls, text):
        eps = cls._validate_text_config(text)
        layers = text.get("layer_types")
        if layers is None:
            layers = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(text["num_hidden_layers"])
            ]
            layers[-1] = "full_attention"
        if len(layers) != text["num_hidden_layers"] or any(
            name not in ("full_attention", "sliding_attention") for name in layers
        ):
            raise ValueError("EmbeddingGemma2 layer_types must describe every layer")
        layers = list(layers)
        layers[-1] = "full_attention"
        window = text.get("sliding_window", 512)
        if type(window) is not int or window <= 0:
            raise ValueError("EmbeddingGemma2 sliding_window must be a positive radius")
        rope = text.get("rope_parameters") or {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000},
        }
        geometry = []
        overrides = text.get("per_layer_config") or {}
        if "per_layer_config" not in text:
            overrides = {
                str(i): {
                    "head_dim": text.get("global_head_dim", 512),
                    "num_key_value_heads": text.get("num_global_key_value_heads", 1),
                }
                for i, kind in enumerate(layers)
                if kind == "full_attention"
            }
        for i, layer_type in enumerate(layers):
            layer = overrides.get(f"{i:02d}", overrides.get(str(i), {}))
            dim = layer.get("head_dim", text["head_dim"])
            kv_heads = layer.get("num_key_value_heads", text["num_key_value_heads"])
            if type(dim) is not int or dim <= 0 or dim % 2:
                raise ValueError(
                    "EmbeddingGemma2 per-layer head_dim must be positive and even"
                )
            if (
                type(kv_heads) is not int
                or kv_heads <= 0
                or text["num_attention_heads"] % kv_heads
            ):
                raise ValueError("EmbeddingGemma2 KV heads must divide Q heads")
            params = rope[layer_type]
            theta = float(params["rope_theta"])
            if params.get("rope_type", "default") != "default" or not (
                math.isfinite(theta) and theta > 0
            ):
                raise ValueError(
                    "EmbeddingGemma2 requires finite positive default RoPE"
                )
            geometry.append(
                {
                    "head_dim": dim,
                    "kv_heads": kv_heads,
                    "rope_theta": theta,
                    "window": window if layer_type == "sliding_attention" else 0,
                }
            )

        return eps, window, rope, geometry

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        with open(os.path.join(ckpt_path, "config.json")) as reader:
            source = json.load(reader)
        text = dict(source.get("text_config", source))
        eps, window, rope, geometry = cls._parse_text_config(text)
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.num_layers = text["num_hidden_layers"]
        config.hidden_size = text["hidden_size"]
        config.vocab_size = text["vocab_size"]
        config.inter_size = text["intermediate_size"]
        config.input_embedding_scalar = config.hidden_size**0.5
        config.attn_config.head_num = text["num_attention_heads"]
        config.attn_config.kv_head_num = text["num_key_value_heads"]
        config.attn_config.size_per_head = text["head_dim"]
        config.attn_config.is_causal = False
        config.attn_config.need_rope_kv_cache = False
        config.attn_config.sliding_window = window
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.dim = text["head_dim"]
        config.attn_config.rope_config.base = int(
            rope["sliding_attention"]["rope_theta"]
        )
        config.layernorm_eps = eps
        config.norm_type = "rmsnorm"
        config.activation_type = "geglu"
        config.qk_norm = True
        config.has_post_decoder_layernorm = True
        config.has_lm_head = False
        config.use_kvcache = False
        config.task_type = TaskType.DENSE_EMBEDDING
        config.max_seq_len = min(int(text.get("max_position_embeddings", 8192)), 8192)
        config.config_dtype = text.get("dtype", text.get("torch_dtype", "bfloat16"))
        config.special_tokens.bos_token_id = text.get("bos_token_id", 2)
        config.special_tokens.eos_token_id = text.get("eos_token_id", 1)
        config.special_tokens.pad_token_id = text.get("pad_token_id", 0)
        config.mm_related_params.config.update(
            {
                "embeddinggemma2_layers": geometry,
                "embeddinggemma2_ple_dim": text["hidden_size_per_layer_input"],
                "embeddinggemma2_embedding_dim": text["embedding_dim"],
            }
        )
        if isinstance(source.get("vision_config"), dict):
            config.mm_related_params.config.update(source["vision_config"])
            config.mm_related_params.config["ckpt_path"] = ckpt_path
            config.mm_related_params.special_tokens["default_mm_token"] = "<|image|>"
            config.mm_model_config.is_multimodal = True
            config.mm_model_config.include_sep_tokens = False
            config.mm_model_config.mm_sep_tokens = [
                [
                    int(source.get("boi_token_id", 255999)),
                    int(source.get("eoi_token_id", 258882)),
                ]
            ]
            if isinstance(source.get("audio_config"), dict):
                config.mm_model_config.mm_sep_tokens = (
                    config.mm_model_config.mm_sep_tokens
                    + [[int(source["boa_token_id"]), int(source["eoa_token_index"])]]
                )
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.embedding_gemma2 import EmbeddingGemma2Model

        self.py_model = EmbeddingGemma2Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model(
    "embedding_gemma2",
    EmbeddingGemma2,
    ["EmbeddingGemma2Model", "EmbeddingGemma2TextModel"],
)
