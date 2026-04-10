import json
import os
from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.gemma4.gemma4_weight import Gemma4WeightInfo
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.ops import HybridAttentionType


class Gemma4VitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model."
        self._ft_prefix = "self.mm_part."


class Gemma4(BaseModel, MultiModalMixin):
    @staticmethod
    def get_weight_cls():
        return Gemma4WeightInfo

    def _create_python_model(self):
        from rtp_llm.models_py.utils.arch import is_cuda

        if not is_cuda():
            raise RuntimeError("Gemma4 is only supported in cuda arch")

        from rtp_llm.models_py.model_desc.gemma4 import Gemma4Model

        self.py_model = Gemma4Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def _init_multimodal(self, mm_model_config, vit_config: VitConfig):
        from rtp_llm.models.gemma4.gemma4_vit import Gemma4ImageEmbedding

        self.mm_part = Gemma4ImageEmbedding(
            self.model_config.mm_related_params, model_config=self.model_config
        )
        # Register vision weights for loading.
        # Checkpoint prefix: model.vision_tower.* and model.embed_vision.*
        # Module prefix: self.mm_part.vision_tower.* and self.mm_part.embed_vision.*
        # We name the projector as "embed_vision" to match checkpoint naming.
        vit_parts = {
            "vision_tower": self.mm_part.vision_tower,
            "embed_vision": self.mm_part.embed_vision,
        }
        self.model_config.mm_related_params.vit_weights = Gemma4VitWeight(
            vit_parts, with_prefix=True
        )

    def support_cuda_graph(self) -> bool:
        # Disabled: Gemma4 creates two separate FMHA impls (sliding + global) in forward(),
        # but CudaGraphRunner only calls prepare_cuda_graph() on the single one returned by
        # prepare_fmha_impl(). During replay, the internal impls' params won't be updated.
        # Enabling requires a composite FMHA wrapper or multi-impl graph runner support.
        return False

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            # Auto-download from HuggingFace if HF_ENDPOINT is set
            hf_endpoint = os.environ.get("HF_ENDPOINT")
            if hf_endpoint:
                import logging
                logging.info(f"Model not found at {ckpt_path}, attempting HF download...")
                from huggingface_hub import snapshot_download
                snapshot_download("google/gemma-4-31B-it", local_dir=ckpt_path)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        # Gemma4 nests text config under "text_config"
        text_config = config_json.get("text_config", config_json)

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(text_config, config)
        cls._parse_hybrid_attention_config(text_config, config)
        cls._parse_normalization_config(text_config, config)
        cls._parse_rope_config(text_config, config)
        cls._parse_gemma4_specific(text_config, config_json, config)

        return config

    @classmethod
    def _parse_basic_config(cls, text_config: dict, config: ModelConfig):
        config.attn_config.head_num = text_config["num_attention_heads"]
        # Use sliding window KV head num as global max (for cache allocation)
        config.attn_config.kv_head_num = text_config["num_key_value_heads"]
        config.attn_config.size_per_head = text_config["head_dim"]
        config.num_layers = text_config["num_hidden_layers"]
        config.hidden_size = text_config["hidden_size"]
        config.inter_size = text_config["intermediate_size"]
        config.vocab_size = text_config["vocab_size"]
        config.max_seq_len = text_config.get("max_position_embeddings", 262144)
        config.tie_word_embeddings = text_config.get("tie_word_embeddings", True)

    @classmethod
    def _parse_hybrid_attention_config(cls, text_config: dict, config: ModelConfig):
        layer_types = text_config.get("layer_types", [])
        if not layer_types:
            return

        hybrid_layer_types: List[HybridAttentionType] = []
        for lt in layer_types:
            if lt == "sliding_attention":
                hybrid_layer_types.append(HybridAttentionType.SLIDING_WINDOW)
            else:  # "full_attention"
                hybrid_layer_types.append(HybridAttentionType.NONE)

        # Enable C++ hybrid attention with heterogeneous KV cache groups
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

        # Per-type KV dimensions for separate MHAKVCacheSpec per group
        config.hybrid_attention_config.sliding_window_kv_head_num = text_config["num_key_value_heads"]
        config.hybrid_attention_config.sliding_window_size_per_head = text_config["head_dim"]
        config.hybrid_attention_config.sliding_window_size = text_config.get("sliding_window", 1024)

        # Global layers use head_dim=512 but no decode kernel supports >256.
        # Reshape: [4 KV×512] → [8 KV×256] to fit FlashInfer constraints.
        # Physical block stride mismatch is handled by reshape_paged_kv_cache slicing.
        global_kv_heads = text_config.get("num_global_key_value_heads", 4)
        global_head_dim = text_config.get("global_head_dim", 512)
        sliding_head_dim = text_config["head_dim"]  # 256
        reshape_factor = global_head_dim // sliding_head_dim  # 2
        config.hybrid_attention_config.global_kv_head_num = global_kv_heads * reshape_factor
        config.hybrid_attention_config.global_size_per_head = sliding_head_dim

    @classmethod
    def _parse_normalization_config(cls, text_config: dict, config: ModelConfig):
        config.layernorm_eps = text_config.get("rms_norm_eps", 1e-6)
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        # Gemma4 uses gelu_pytorch_tanh (approximate GELU, gated)
        config.activation_type = "gated-gelu"

    @classmethod
    def _parse_rope_config(cls, text_config: dict, config: ModelConfig):
        # Default RoPE config uses sliding attention params (majority of layers)
        rope_params = text_config.get("rope_parameters", {})
        sliding_rope = rope_params.get("sliding_attention", {})

        config.attn_config.rope_config.style = 1  # RopeStyle::Base
        config.attn_config.rope_config.base = sliding_rope.get("rope_theta", 10000.0)
        config.attn_config.rope_config.dim = text_config["head_dim"]  # Full rotation for sliding

    @classmethod
    def _parse_gemma4_specific(cls, text_config: dict, config_json: dict, config: ModelConfig):
        # Gemma-specific: scale embedding by sqrt(hidden_size)
        config.input_embedding_scalar = config.hidden_size ** 0.5

        # Final logit softcapping
        final_logit_softcapping = text_config.get("final_logit_softcapping", None)
        if final_logit_softcapping:
            config.final_logit_softcapping = float(final_logit_softcapping)

        # Store per-type attention configs as Python-only fields for model_desc
        rope_params = text_config.get("rope_parameters", {})
        full_rope = rope_params.get("full_attention", {})

        # Reshape global attention: [4 KV×512] → [8 KV×256] for FlashInfer decode
        global_kv_heads = text_config.get("num_global_key_value_heads", 4)
        global_head_dim = text_config.get("global_head_dim", 512)
        sliding_head_dim = text_config["head_dim"]  # 256
        reshape_factor = global_head_dim // sliding_head_dim
        orig_partial_rotary = full_rope.get("partial_rotary_factor", 0.25)
        config.gemma4_global_attn_config = {
            "kv_head_num": global_kv_heads * reshape_factor,  # 8
            "head_dim": sliding_head_dim,  # 256
            "head_num": text_config["num_attention_heads"] * reshape_factor,  # 64
            "rope_theta": full_rope.get("rope_theta", 1000000.0),
            "partial_rotary_factor": orig_partial_rotary * reshape_factor,  # 0.5
        }
        config.gemma4_sliding_attn_config = {
            "kv_head_num": text_config["num_key_value_heads"],
            "head_dim": text_config["head_dim"],
            "rope_theta": rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0),
            "partial_rotary_factor": 1.0,
            "sliding_window": text_config.get("sliding_window", 1024),
        }

        # K=V sharing flag
        config.gemma4_attention_k_eq_v = text_config.get("attention_k_eq_v", False)

        # Layer types for model_desc
        config.gemma4_layer_types = text_config.get("layer_types", [])

        # Vision config
        vision_config = config_json.get("vision_config", None)
        if vision_config:
            config.gemma4_vision_config = vision_config
            boi_token_id = config_json.get("boi_token_id", 255999)
            eoi_token_id = config_json.get("eoi_token_id", 258882)
            image_token_id = config_json.get("image_token_id", 258880)
            config.gemma4_boi_token_id = boi_token_id
            config.gemma4_eoi_token_id = eoi_token_id
            config.gemma4_image_token_id = image_token_id
            config.gemma4_vision_soft_tokens = config_json.get("vision_soft_tokens_per_image", 280)

            # Set up mm_related_params for multimodal pipeline
            config.mm_related_params.config = vision_config
            config.mm_related_params.config["ckpt_path"] = config.ckpt_path
            if config.mm_related_params.special_tokens is None:
                from rtp_llm.config.model_config import SpecialTokens
                config.mm_related_params.special_tokens = SpecialTokens()
            config.mm_related_params.special_tokens.update({"default_mm_token": "<image>"})
            config.mm_model_config.mm_sep_tokens = [[boi_token_id, eoi_token_id]]


register_model("gemma4", Gemma4, ["Gemma4ForConditionalGeneration"])
