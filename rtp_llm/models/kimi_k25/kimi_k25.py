"""Kimi-K2.5 model: DeepSeekV3 text core + 27-layer ViT + patchmerger.

Top-level HF config:
  - architectures = ["KimiK25ForConditionalGeneration"]
  - model_type    = "kimi_k25"
  - text_config: DeepseekV3-shaped (MLA, MoE, YaRN); routed-expert MoE is
    pre-quantized as compressed-tensors INT4 g=32.
  - vision_config: 27-layer ViT (vt_hidden=1152, 16 heads).
  - media_placeholder_token_id (163605) marks where vision tokens splice in.

This wrapper:
  1. Re-uses DeepSeekV2 text path (config / weights / python_model).
  2. Promotes `text_config.quantization_config` to the top level so the
     compressed loader picks it up.
  3. Wires the ViT + projector through MultiModalMixin.
"""

import json
import logging
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models.kimi_k25.kimi_k25_vit import KimiK25ImageEmbedding
from rtp_llm.models.kimi_k25.kimi_k25_weight import (
    KimiK25VitWeight,
    KimiK25Weight,
)
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin


_DEFAULT_MEDIA_PLACEHOLDER_ID = 163605
_DEFAULT_VIDEO_PLACEHOLDER = "<|kimi_k25_video_placeholder|>"


def _read_top_config(ckpt_path: str) -> Dict[str, Any]:
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


class KimiK25(DeepSeekV2, MultiModalMixin):
    """Kimi-K2.5 multimodal model class."""

    def support_cuda_graph(self) -> bool:
        return True

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        top_config = _read_top_config(ckpt_path)

        # Promote `text_config.quantization_config` to the top level so the
        # standard loader path discovers compressed-tensors INT4. Done in
        # the on-disk JSON-equivalent dict before any further processing.
        text_config = top_config.get("text_config", {}) if top_config else {}
        if isinstance(text_config, dict) and "quantization_config" in text_config:
            top_config.setdefault(
                "quantization_config", text_config["quantization_config"]
            )

        # Build a transient ckpt_path-equivalent config view for DeepSeekV2.
        # DeepSeekV2._from_hf reads from the file on disk, so for the text
        # fields we rewrite `text_config` into the top of the config and
        # write it back to a temp `config.json` in-memory? Simplest path:
        # populate ModelConfig manually by passing `text_config` through
        # the existing parser.
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"

        # `_from_hf` opens config.json — cheaper than re-implementing it
        # here. To honour the `text_config` nesting we temporarily monkey
        # the loaded JSON: write a flattened version to a tmp file and
        # point `ckpt_path` at it would be invasive, so instead replicate
        # the small subset of `_from_hf` we need on `text_config`.
        cls._populate_text_config(config, text_config or top_config)
        cls._populate_vision_config(config, top_config)

        # Mark this as multimodal so `BaseModel._may_init_multimodal`
        # picks the path.
        if hasattr(config, "mm_model_config") and config.mm_model_config is not None:
            config.mm_model_config.is_multimodal = True
        config.model_name = "kimi_k25"
        return config

    @classmethod
    def _populate_text_config(
        cls, config: ModelConfig, text_config: Dict[str, Any]
    ) -> None:
        """Apply DeepSeekV2._from_hf-style parsing to `text_config`.

        Mirrors the subset of fields needed for runtime correctness.
        Falls back to defaults when entries are missing.
        """
        if not text_config:
            return

        config.inter_size = text_config["intermediate_size"]
        config.attn_config.head_num = text_config["num_attention_heads"]
        config.attn_config.kv_head_num = text_config.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.num_layers = text_config["num_hidden_layers"]
        config.attn_config.rope_config.base = int(
            text_config.get("rope_theta", 50000)
        )
        config.vocab_size = text_config["vocab_size"]
        config.layernorm_eps = text_config.get("rms_norm_eps", 1e-6)
        config.tie_word_embeddings = text_config.get("tie_word_embeddings", False)
        config.hidden_size = text_config["hidden_size"]

        # MLA
        config.attn_config.use_mla = True
        config.attn_config.q_lora_rank = int(text_config.get("q_lora_rank") or 0)
        config.attn_config.kv_lora_rank = int(text_config.get("kv_lora_rank") or 0)
        config.attn_config.nope_head_dim = text_config["qk_nope_head_dim"]
        config.attn_config.rope_head_dim = text_config["qk_rope_head_dim"]
        config.attn_config.v_head_dim = text_config["v_head_dim"]
        config.attn_config.size_per_head = (
            config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
        )
        config.attn_config.rope_config.dim = config.attn_config.rope_head_dim
        config.attn_config.rope_config.style = 0
        config.attn_config.rope_config.offset = config.attn_config.nope_head_dim
        config.attn_config.rope_config.is_neox_style = not text_config.get(
            "rope_interleave", True
        )

        # YaRN scaling
        rope_scaling = text_config.get("rope_scaling")
        if rope_scaling is not None:
            from rtp_llm.utils.model_weight import yarn_get_mscale

            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.factor1 = float(
                rope_scaling.get("beta_slow", 1)
            )
            config.attn_config.rope_config.factor2 = float(
                rope_scaling.get("beta_fast", 32)
            )
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            scaling_factor = rope_scaling["factor"]
            mscale = rope_scaling["mscale"]
            mscale_all_dim = rope_scaling["mscale_all_dim"]
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            config.attn_config.rope_config.mscale = yarn_get_mscale(
                scaling_factor, mscale
            ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

        config.max_seq_len = text_config.get(
            "max_position_embeddings", config.max_seq_len or 8192
        )

        # MoE
        scoring_func = text_config.get("scoring_func", "softmax")
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1
        else:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")
        config.routed_scaling_factor = text_config.get("routed_scaling_factor", 1.0)
        config.moe_k = text_config["num_experts_per_tok"]
        config.expert_num = text_config["n_routed_experts"]
        moe_intermediate_size = text_config["moe_intermediate_size"]
        config.moe_n_group = text_config.get("n_group", 1)
        config.moe_topk_group = text_config.get("topk_group", 1)
        n_shared_experts = text_config.get("n_shared_experts", 1)
        config.inter_size = n_shared_experts * moe_intermediate_size
        config.has_moe_norm = text_config.get("norm_topk_prob", False)
        config.moe_style = 2  # shared + routed

        moe_step = text_config.get("moe_layer_freq", 1)
        first_k_dense_replace = text_config.get("first_k_dense_replace", 0)
        config.moe_layer_index = [
            i
            for i in range(config.num_layers)
            if i >= first_k_dense_replace and i % moe_step == 0
        ]

        config.config_dtype = text_config.get("torch_dtype", None)

    @classmethod
    def _populate_vision_config(
        cls, config: ModelConfig, top_config: Dict[str, Any]
    ) -> None:
        vision_config = top_config.get("vision_config", {}) if top_config else {}
        # Drop unsupported HF keys (e.g. `_attn_implementation`) before
        # passing to KimiK25VisionConfig so PretrainedConfig.__init__ does
        # not raise on unknown kwargs.
        sanitized_vision_config = {
            k: v for k, v in vision_config.items() if not k.startswith("_")
        }
        config.mm_related_params.config["vision_config"] = sanitized_vision_config
        # The renderer / image processor need the ckpt path to locate the
        # HF AutoImageProcessor (trust_remote_code) bundle.
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path
        media_placeholder_id = top_config.get(
            "media_placeholder_token_id", _DEFAULT_MEDIA_PLACEHOLDER_ID
        )
        video_placeholder = top_config.get(
            "video_placeholder", _DEFAULT_VIDEO_PLACEHOLDER
        )
        config.mm_related_params.special_token_ids.update(
            {
                "image_token_index": media_placeholder_id,
                "ignore_token_index": -100,
            }
        )
        config.mm_related_params.special_tokens.update(
            {
                "default_mm_token": "<|media_pad|>",
                "video_placeholder": video_placeholder,
            }
        )
        if hasattr(config, "mm_model_config") and config.mm_model_config is not None:
            try:
                config.mm_model_config.mm_sep_tokens = [[int(media_placeholder_id)]]
            except Exception as exc:  # mm_model_config schema may differ
                logging.warning(
                    f"unable to set mm_sep_tokens for kimi_k25: {exc}"
                )

    def _init_multimodal(self, mm_model_config, vit_config):
        mm_related_params = self.model_config.mm_related_params
        self.ignore_id = -100
        self.mm_part = KimiK25ImageEmbedding(
            mm_related_params,
            model_config=self.model_config,
            ignore_id=self.ignore_id,
        )
        # Dict keys must match the on-disk ckpt prefix segment so that
        # `BaseVitWeights._get_vit_params` builds names like
        # `vision_tower.encoder.blocks.X.xxx` and `mm_projector.proj.X.xxx`.
        mm_related_params.vit_weights = KimiK25VitWeight(
            {
                "vision_tower": self.mm_part.vision_tower,
                "mm_projector": self.mm_part.mm_projector,
            },
            with_prefix=True,
        )

    @staticmethod
    def get_weight_cls():
        return KimiK25Weight


# Primary registration: HF official architecture name for Kimi-K2.5.
register_model("kimi_k25", KimiK25, ["KimiK25ForConditionalGeneration"])
