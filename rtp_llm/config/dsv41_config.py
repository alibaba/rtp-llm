"""Metadata for the DeepSeek-V4.1-Flash release checkpoint."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


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
        return cls(
            text=copy.deepcopy(source["text_config"]),
            vision=copy.deepcopy(source["vision_config"]),
            quantization=copy.deepcopy(source["quantization_config"]),
            dtype=source["dtype"],
            bos_token_id=source["bos_token_id"],
            eos_token_id=source["eos_token_id"],
            pad_token_id=source["pad_token_id"],
            image_token_id=source["image_token_id"],
        )

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
