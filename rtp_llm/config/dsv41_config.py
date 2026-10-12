"""CPU-only metadata and source ownership for DeepSeek V4.1 Flash."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


def v41_rotary_compute_device(text: Mapping[str, Any]) -> Optional[str]:
    """Read a portable arithmetic origin; CUDA ordinals remain worker-local."""
    value = text.get("rotary_compute_device")
    if value is not None and (type(value) is not str or value not in ("cpu", "cuda")):
        raise ValueError("V4.1 rotary_compute_device must be cpu, cuda or None")
    return value


def v41_sparse_attention_mode(text: Mapping[str, Any]) -> Optional[str]:
    """Read the explicit, bounded short sparse attention choice."""
    value = text.get("sparse_attention_mode")
    if value is not None and (type(value) is not str or value != "short_preserved"):
        raise ValueError("V4.1 sparse_attention_mode must be short_preserved or None")
    return value


@dataclass(frozen=True)
class V41LayerSources:
    layer_id: int
    compress_ratio: int
    kv_source_layer_id: Optional[int]
    index_source_layer_id: Optional[int]
    candidate_source_layer_id: Optional[int]

    @property
    def owns_kv(self) -> bool:
        return self.kv_source_layer_id == self.layer_id

    @property
    def owns_index(self) -> bool:
        return self.index_source_layer_id == self.layer_id


def _positive_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"V4.1 {key} must be a positive integer, got {value!r}")
    return value


def _layer_ids(text: Mapping[str, Any], key: str) -> list[int]:
    values = text.get(key)
    count = text["num_hidden_layers"]
    if not isinstance(values, list) or any(
        type(i) is not int or not 0 <= i < count for i in values
    ):
        raise ValueError(f"V4.1 {key} must contain layer IDs in [0, {count})")
    if values != sorted(set(values)):
        raise ValueError(f"V4.1 {key} must be sorted and unique")
    return values


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

    @classmethod
    def from_path(cls, checkpoint: str | Path) -> "V41Config":
        with (Path(checkpoint) / "config.json").open(encoding="utf-8") as reader:
            return cls.from_dict(json.load(reader))

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> "V41Config":
        for key in ("text_config", "quantization_config"):
            if not isinstance(source.get(key), Mapping):
                raise ValueError(f"V4.1 requires a nested {key} object")
        config = cls(
            text=copy.deepcopy(dict(source["text_config"])),
            vision=copy.deepcopy(dict(source.get("vision_config", {}))),
            quantization=copy.deepcopy(dict(source["quantization_config"])),
            dtype=source.get("dtype", "bfloat16"),
            bos_token_id=source.get("bos_token_id", 0),
            eos_token_id=source.get("eos_token_id", 1),
            pad_token_id=source.get("pad_token_id", 2),
            image_token_id=source.get("image_token_id", 129264),
        )
        config.validate()
        return config

    def validate(self) -> None:
        text = self.text
        v41_rotary_compute_device(text)
        v41_sparse_attention_mode(text)
        for key in (
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "n_routed_experts",
            "num_experts_per_tok",
            "n_shared_experts",
            "num_attention_heads",
            "head_dim",
            "qk_rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "sliding_window",
            "hc_mult",
        ):
            _positive_int(text, key)
        if text.get("num_key_value_heads", 1) != 1:
            raise ValueError("V4.1 requires one KV head")
        if text["num_experts_per_tok"] > text["n_routed_experts"]:
            raise ValueError("V4.1 num_experts_per_tok exceeds n_routed_experts")
        if text["num_attention_heads"] % text["o_groups"]:
            raise ValueError("V4.1 num_attention_heads must be divisible by o_groups")
        if text["qk_rope_head_dim"] > text["head_dim"] or text["qk_rope_head_dim"] % 2:
            raise ValueError("V4.1 qk_rope_head_dim must be even and <= head_dim")
        if text.get("num_hash_layers", 0) != 0:
            raise ValueError("V4.1 does not use hash-router layers")
        for key in ("hidden_size", "moe_intermediate_size"):
            if text[key] % 32:
                raise ValueError(
                    f"V4.1 {key} must be divisible by quantization group 32"
                )
        expected = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
        for key, value in expected.items():
            if self.quantization.get(key) != value:
                raise ValueError(f"V4.1 quantization_config.{key} must be {value!r}")
        if self.dtype != "bfloat16":
            raise ValueError("V4.1 requires bfloat16 compute metadata")
        for key in ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id"):
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value < text["vocab_size"]:
                raise ValueError(f"V4.1 {key} must be in the model vocabulary")
        self.layer_sources()
        engram_ids = _layer_ids(text, "engram_layer_ids")
        rows = text.get("engram_num_embeddings", [])
        if (
            not isinstance(rows, list)
            or len(rows) != len(engram_ids)
            or any(type(n) is not int or n <= 0 for n in rows)
        ):
            raise ValueError("V4.1 engram_num_embeddings must match engram_layer_ids")
        if engram_ids:
            for key in (
                "engram_head_dim",
                "engram_n_heads",
                "engram_vocab_size",
                "engram_compressed_vocab_size",
            ):
                _positive_int(text, key)
            if text["engram_head_dim"] % 32:
                raise ValueError("V4.1 engram_head_dim must be divisible by 32")
            if text.get("engram_max_ngram_size") != 4:
                raise ValueError("V4.1 token history requires engram_max_ngram_size=4")

    def layer_sources(self) -> tuple[V41LayerSources, ...]:
        """Resolve producer IDs globally, including dependencies across PP stages."""
        text = self.text
        count = _positive_int(text, "num_hidden_layers")
        draft_count = text.get("num_nextn_predict_layers", 0)
        if type(draft_count) is not int or draft_count < 0:
            raise ValueError("V4.1 num_nextn_predict_layers must be nonnegative")
        ratios = text.get("compress_ratios")
        if not isinstance(ratios, list) or len(ratios) != count + draft_count:
            raise ValueError("V4.1 compress_ratios must cover target and draft layers")
        if any(type(r) is not int or r not in (0, 1, 2) for r in ratios):
            raise ValueError("V4.1 compress_ratios must contain only 0, 1, 2")
        if any(ratios[count:]):
            raise ValueError("V4.1 draft layers must use SWA only")
        kv_ids = _layer_ids(text, "kv_source_layer_ids")
        index_ids = _layer_ids(text, "index_source_layer_ids")
        if any(ratios[i] == 0 for i in kv_ids + index_ids):
            raise ValueError("V4.1 SWA-only layers cannot be global sources")
        if not set(kv_ids).issubset(index_ids):
            raise ValueError("V4.1 every KV source must also produce a new selection")
        candidate = text.get("candidate_source_layer_id", -1)
        if type(candidate) is not int or candidate < -1:
            raise ValueError("V4.1 candidate_source_layer_id must be -1 or a layer ID")
        if candidate >= 0:
            if candidate not in kv_ids or ratios[candidate] != 1:
                raise ValueError(
                    "V4.1 candidate source must be a ratio-1 KV/index source"
                )
            if kv_ids[-1] != candidate:
                raise ValueError(
                    "V4.1 candidate filtering requires the final KV source"
                )
            _positive_int(text, "candidate_topk_blocks")
            _positive_int(text, "candidate_block_size")
        kv_source = index_source = None
        bindings = []
        for layer_id, ratio in enumerate(ratios[:count]):
            if layer_id in kv_ids:
                kv_source = layer_id
            if layer_id in index_ids:
                index_source = layer_id
            if ratio:
                if kv_source is None or index_source is None:
                    raise ValueError(
                        f"V4.1 layer {layer_id} has no preceding KV/index source"
                    )
                if ratios[kv_source] != ratio or ratios[index_source] != ratio:
                    raise ValueError(
                        f"V4.1 layer {layer_id} changes ratio without new sources"
                    )
            bindings.append(
                V41LayerSources(
                    layer_id,
                    ratio,
                    kv_source if ratio else None,
                    index_source if ratio else None,
                    candidate if ratio and 0 <= candidate <= layer_id else None,
                )
            )
        return tuple(bindings)

    def local_expert_ids(self, ep_size: int, ep_rank: int) -> range:
        count = self.text["n_routed_experts"]
        if type(ep_size) is not int or ep_size <= 0 or count % ep_size:
            raise ValueError("V4.1 n_routed_experts must be divisible by ep_size")
        if type(ep_rank) is not int or not 0 <= ep_rank < ep_size:
            raise ValueError("V4.1 ep_rank is outside its EP group")
        local_count = count // ep_size
        return range(ep_rank * local_count, (ep_rank + 1) * local_count)

    @property
    def engram_host_bytes(self) -> int:
        if not self.text["engram_layer_ids"]:
            return 0
        dim = self.text["engram_head_dim"]
        return sum(self.text["engram_num_embeddings"]) * (dim + dim // 32)

    def runtime_text_config(self, checkpoint: str) -> dict[str, Any]:
        text = copy.deepcopy(self.text)
        text.update(
            checkpoint_path=str(checkpoint),
            dtype=self.dtype,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            image_token_id=self.image_token_id,
            quantization_config=copy.deepcopy(self.quantization),
            vision_config=copy.deepcopy(self.vision),
        )
        return text


def dsv41_pp_partition(num_layers: int, pp_size: int, model_config=None) -> list[int]:
    """Keep all GLOBAL producers and both Engram tables on prefill stage zero."""
    if num_layers <= 0:
        raise ValueError("V4.1 PP requires a positive layer count")
    if pp_size == 1:
        return [num_layers]
    if pp_size != 2 or num_layers != 40:
        raise ValueError(
            "V4.1 prefill PP requires 40 layers and PP2 with counts [21, 19]"
        )
    if model_config is not None:
        text = model_config.deepseek_v41_config
        if (
            text["kv_source_layer_ids"][-1:] != [20]
            or text.get("candidate_source_layer_id") != 20
            or any(layer >= 21 for layer in text["engram_layer_ids"])
        ):
            raise ValueError(
                "V4.1 PP2 requires the final KV/candidate source at L20 "
                "and all Engram layers on stage zero"
            )
    return [21, 19]
