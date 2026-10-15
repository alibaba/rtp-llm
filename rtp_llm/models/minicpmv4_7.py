"""MiniCPM-V-4.7 model registration (openbmb/MiniCPM-V-4.7 family).

Text backbone is the stock ``qwen3_5_moe_text`` (Qwen3.5-MoE: 30 GDN linear
attention layers + 10 full attention layers, all-MoE with shared expert) for
the 35B-A3B variants and ``qwen3_5_text`` (dense) for the 1B variant; both
share the MiniCPM SigLIP vision tower (window merger at insert_layer_id 6 +
final merger) and canvas M-RoPE.  See
runs/minicpmv47_h20_20261013/adaptation-plan.md for the wiring contract.
"""
import json
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Dense, Qwen35Moe, Qwen3NextBase
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    Qwen35MoeWeight,
    build_qwen35_dense_ffn_weights,
)
from rtp_llm.ops import HybridAttentionType


class MiniCPMV4_7Weight(Qwen35MoeWeight):
    """MiniCPM-V-4.7 weight loading.

    Same layout as Qwen3.5 (dynamic prefix detection resolves
    ``model.language_model.``); dispatches between the MoE expert stack
    (35B-A3B, stacked 3D experts) and the dense FFN (1B) on ``expert_num``.
    Vision tower (``model.vision_tower.*``) and final merger
    (``model.merger.*``) weights are loaded by the multimodal mixin loader.
    """

    def _create_ffn_weight(self) -> List[Any]:
        if self.model_config.expert_num and self.model_config.expert_num > 0:
            return super()._create_ffn_weight()
        return build_qwen35_dense_ffn_weights(self.prefix, self._align_size, self.ffn_config)


class MiniCPMV4_7(Qwen35Moe):
    @staticmethod
    def get_weight_cls():
        return MiniCPMV4_7Weight

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        text_config_json = config_json["text_config"]

        # Qwen3.5 text-side parsing chain, with MoE (35B-A3B) or dense (1B)
        # selection and the MiniCPM multimodal token contract.
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        # lm_head runs in BF16: the vocab GEMV (248144 x 2048) is the largest
        # per-step decode kernel and the fp32 upcast doubles its HBM traffic
        # (598us -> 281us on H20, measured).  DSV4 serves with the same flag.
        config.enable_fp32_lm_head = False
        cls._parse_basic_config(text_config_json, config)
        cls._parse_rope_config(text_config_json, config)
        cls._parse_normalization_config(text_config_json, config)
        if "num_experts" in text_config_json:
            Qwen35Moe._parse_moe_config(text_config_json, config)
        else:
            Qwen35Dense._parse_moe_config(text_config_json, config)
        cls._parse_hybrid_attention_config(text_config_json, config)
        cls._parse_linear_attention_config(text_config_json, config)
        cls._parse_mm_config(config_json, config)
        return config

    @classmethod
    def _parse_hybrid_attention_config(cls, config_json: Dict[str, Any], config: ModelConfig):
        # Prefer the explicit layer_types list; fall back to the interval form.
        layer_types = config_json.get("layer_types")
        if layer_types is not None:
            if len(layer_types) != config.num_layers:
                raise ValueError(
                    f"layer_types length {len(layer_types)} != num_hidden_layers {config.num_layers}"
                )
            hybrid_layer_types = []
            for layer_type in layer_types:
                if layer_type == "full_attention":
                    hybrid_layer_types.append(HybridAttentionType.NONE)
                elif layer_type == "linear_attention":
                    hybrid_layer_types.append(HybridAttentionType.LINEAR)
                else:
                    raise ValueError(f"unknown layer type {layer_type!r}")
            config.hybrid_attention_config.enable_hybrid_attention = True
            config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types
        else:
            Qwen3NextBase._parse_hybrid_attention_config(config_json, config)

    @classmethod
    def _parse_mm_config(cls, config_json: Dict[str, Any], config):
        language_model_only = config_json.get("language_model_only")
        if isinstance(language_model_only, str):
            normalized = language_model_only.lower()
            if normalized not in ("true", "1", "false", "0"):
                raise ValueError(
                    f"language_model_only must be a boolean, got {language_model_only!r}"
                )
            language_model_only = normalized in ("true", "1")
        if language_model_only is True or (
            isinstance(language_model_only, int) and language_model_only == 1
        ):
            return

        config.mm_model_config.is_multimodal = True
        # One <image>...</image> span per picture; markers stay in the token
        # stream (include_sep_tokens=False) and the span interior (thumbnail
        # + slice pads) is replaced by that image's concatenated features.
        config.mm_model_config.mm_sep_tokens = [
            [config_json["image_start_id"], config_json["image_end_id"]]
        ]
        config.mm_model_config.include_sep_tokens = False

        if config.mm_related_params.special_tokens is None:
            from rtp_llm.config.model_config import SpecialTokens

            config.mm_related_params.special_tokens = SpecialTokens()
        config.mm_related_params.special_tokens.update({"default_mm_token": "<|image_pad|>"})
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path


register_model("minicpmv4_7", MiniCPMV4_7, ["MiniCPMV4_7ForConditionalGeneration"])
