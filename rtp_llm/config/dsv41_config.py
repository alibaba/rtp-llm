"""Validated metadata for the DeepSeek-V4.1-Flash release checkpoint."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class V41Config:
    text: dict[str, Any]
    vision: dict[str, Any]
    quantization: dict[str, Any]
    dtype: str
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    image_token_id: int
    kv_owners: tuple[int, ...]
    index_owners: tuple[int, ...]
    global_owner_by_layer: tuple[int, ...]
    topk_owner_by_layer: tuple[int, ...]

    @classmethod
    def from_path(cls, checkpoint: str | Path) -> "V41Config":
        with (Path(checkpoint) / "config.json").open(encoding="utf-8") as reader:
            return cls.from_dict(json.load(reader))

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> "V41Config":
        if source.get("model_type") != "deepseek_v41" or source.get(
            "architectures"
        ) != ["DeepseekV41ForCausalLM"]:
            raise ValueError("expected DeepseekV41ForCausalLM / deepseek_v41")
        text = copy.deepcopy(source.get("text_config"))
        vision = copy.deepcopy(source.get("vision_config"))
        quantization = copy.deepcopy(source.get("quantization_config"))
        if not all(isinstance(item, dict) for item in (text, vision, quantization)):
            raise ValueError(
                "text_config, vision_config and quantization_config are required"
            )
        required = (
            "vocab_size",
            "hidden_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "qk_rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "max_position_embeddings",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "sliding_window",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "candidate_topk_blocks",
            "candidate_block_size",
            "hc_mult",
            "hc_sinkhorn_iters",
            "num_nextn_predict_layers",
            "engram_max_ngram_size",
            "engram_vocab_size",
            "engram_n_heads",
            "engram_head_dim",
            "engram_compressed_vocab_size",
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_n_routed_experts",
            "dspark_num_experts_per_tok",
        )
        for name in required:
            _positive_int(text.get(name), f"text_config.{name}")
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
            "patch_size",
            "downsample_ratio",
            "max_image_tokens",
            "min_pixels",
        ):
            _positive_int(vision.get(name), f"vision_config.{name}")
        if vision["hidden_size"] % vision["num_attention_heads"]:
            raise ValueError(
                "vision hidden_size must be divisible by its attention heads"
            )
        if (
            vision["min_pixels"] != 295936
            or vision["max_image_tokens"] != 1024
            or vision.get("max_wh_ratio") is not None
        ):
            raise ValueError(
                "V4.1 image preprocessing requires min_pixels=295936, 1024 tokens and no aspect-ratio cap"
            )
        n_layers = text["num_hidden_layers"]
        n_draft = text["num_nextn_predict_layers"]
        if (n_layers, n_draft, text["hc_mult"]) != (40, 3, 4):
            raise ValueError(
                "V4.1 Flash requires 40 target / 3 draft layers and hc_mult=4"
            )
        expected_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * n_draft
        ratios = text.get("compress_ratios")
        if ratios != expected_ratios or any(type(ratio) is not int for ratio in ratios):
            raise ValueError(
                "compress_ratios must describe all 40 target and 3 draft layers"
            )
        kv_owners = tuple(text.get("kv_source_layer_ids", ()))
        index_owners = tuple(text.get("index_source_layer_ids", ()))
        if kv_owners != (2, 8, 14, 20) or index_owners != (
            2,
            8,
            14,
            20,
            24,
            28,
            32,
            36,
        ):
            raise ValueError("invalid V4.1 global-KV or query-index owner schedule")
        if text.get("candidate_source_layer_id") != 20:
            raise ValueError("candidate_source_layer_id must be 20")
        global_map, topk_map = [], []
        for layer in range(n_layers):
            global_owner = max(
                (owner for owner in kv_owners if owner <= layer), default=-1
            )
            index_owner = max(
                (owner for owner in index_owners if owner <= layer), default=-1
            )
            if global_owner >= 0 and ratios[global_owner] != ratios[layer]:
                raise ValueError(
                    f"layer {layer} has a different ratio from its KV owner"
                )
            global_map.append(global_owner)
            topk_map.append(index_owner)
        if text["num_attention_heads"] % text["o_groups"]:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if text["qk_rope_head_dim"] > text["head_dim"] or text["qk_rope_head_dim"] % 2:
            raise ValueError("qk_rope_head_dim must be even and fit in head_dim")
        for name in (
            "hidden_size",
            "moe_intermediate_size",
            "q_lora_rank",
            "o_lora_rank",
        ):
            if text[name] % 32:
                raise ValueError(
                    f"{name} must be divisible by the weight quantization group32"
                )
        if source.get("dtype") != "bfloat16" or quantization != {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }:
            raise ValueError(
                "V4.1 requires BF16 compute, dense FP8 block32 and FP4 experts"
            )
        if text.get("num_hash_layers", 0) != 0:
            raise ValueError("V4.1 has no token-id MoE hash router")
        if (
            text.get("scoring_func") != "sqrtsoftplus"
            or text.get("swiglu_limit") != 10.0
        ):
            raise ValueError("V4.1 requires sqrtsoftplus routing and SwiGLU limit10")
        if text.get("rms_norm_eps") != 1e-20 or text.get("hc_eps") != 1e-6:
            raise ValueError(
                "text rms_norm_eps and hc_eps must retain their distinct values"
            )
        if tuple(text.get("engram_layer_ids", ())) != (1, 14):
            raise ValueError("Engram layers must be L1 and L14")
        rows = text.get("engram_num_embeddings", ())
        if len(rows) != 2 or any(type(row) is not int or row <= 0 for row in rows):
            raise ValueError(
                "engram_num_embeddings must contain both complete host tables"
            )
        if (
            text.get("engram_pad_token_id") != 2
            or text["engram_compressed_vocab_size"] != 99092
        ):
            raise ValueError(
                "Engram must retain pad token2 and compressed vocabulary99092"
            )
        if (
            tuple(text.get("dspark_target_layer_ids", ())) != (37, 38, 39)
            or text["dspark_block_size"] != 5
        ):
            raise ValueError("DSpark requires L37/L38/L39 input capture and block5")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id"):
            value = source.get(name)
            if type(value) is not int or not 0 <= value < text["vocab_size"]:
                raise ValueError(f"{name} must be an in-vocabulary integer")
        return cls(
            text=text,
            vision=vision,
            quantization=quantization,
            dtype=source["dtype"],
            bos_token_id=source["bos_token_id"],
            eos_token_id=source["eos_token_id"],
            pad_token_id=source["pad_token_id"],
            image_token_id=source["image_token_id"],
            kv_owners=kv_owners,
            index_owners=index_owners,
            global_owner_by_layer=tuple(global_map),
            topk_owner_by_layer=tuple(topk_map),
        )

    def validate_parallelism(self, *, tp_size: int, ep_size: int) -> None:
        _positive_int(tp_size, "tp_size")
        _positive_int(ep_size, "ep_size")
        for field in ("num_attention_heads", "o_groups", "index_n_heads", "vocab_size"):
            if self.text[field] % tp_size:
                raise ValueError(f"{field} must be divisible by TP={tp_size}")
        for field in ("n_routed_experts", "dspark_n_routed_experts"):
            if self.text[field] % ep_size:
                raise ValueError(f"{field} must be divisible by EP={ep_size}")

    @property
    def engram_host_bytes(self) -> int:
        dim = self.text["engram_head_dim"]
        return sum(self.text["engram_num_embeddings"]) * (dim + dim // 32)

    def vision_parameters(self) -> dict[str, Any]:
        return {
            "vision_dim": self.vision["hidden_size"],
            "vision_inter_dim": self.vision["intermediate_size"],
            "vision_n_heads": self.vision["num_attention_heads"],
            "vision_n_layers": self.vision["num_hidden_layers"],
            "vision_patch_size": self.vision["patch_size"],
            "vision_downsample_ratio": self.vision["downsample_ratio"],
            "vision_rope_theta": self.vision["rope_theta"],
            "hidden_size": self.text["hidden_size"],
        }
