"""GLM-4.x GQA (not MLA) decoder/checkpoint ownership.

The runtime config's inter_size describes the shared expert. It is not the
width of the dense prefix. Appended NextN layers do not belong to the target.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from rtp_llm.models_py.weight_mapper import (
    _unwrap_pytorch_state_dict,
    checkpoint_catalog,
    discover_ckpt_files,
)

def _integer(value, field, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid GLM4 {field}: {value!r}")
    return value


def _has_bias(names, layers, prefix):
    for name in names:
        # Only HF names carry the language namespace. FT writes the same
        # top-level rank_* runtime names for text-only and multimodal models.
        if name.startswith("rank_"):
            parts = name.split(".")
        elif name.startswith(prefix + "model.layers."):
            parts = name.removeprefix(prefix).split(".")
        else:
            continue
        if (
            len(parts) == 6
            and parts[:2] == ["model", "layers"]
            and parts[2].isdigit()
            and int(parts[2]) < layers
            and parts[3:] == ["mlp", "gate", "e_score_correction_bias"]
        ):
            return True
        # Old FT checkpoints persist the GLM correction tensor under its
        # per-rank runtime name. It is not guaranteed to have topk_method
        # metadata; the old loader, like the HF path, detects the actual key.
        if (
            len(parts) == 5
            and re.fullmatch(r"rank_[0-9]+_[0-9]+_[0-9]+", parts[0])
            and parts[1] == "layers"
            and parts[2].isdigit()
            and int(parts[2]) < layers
            and parts[3:] == ["partial_moe_weights", "e_score_correction_bias"]
        ):
            return True
    return False


def has_correction_bias(path, layers, *, prefix=""):
    def matches(names):
        return _has_bias(names, layers, prefix)

    # Use the same logical checkpoint as the loader, not optimizer/adapter shards
    # or extra keys that are physically present but absent from the index.
    catalog = checkpoint_catalog(str(path))
    if catalog is not None:
        return matches(name for name, _ in catalog.entries)
    for shard in discover_ckpt_files(str(path)):
        if shard.endswith(".safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                if matches(tensors.keys()):
                    return True
            continue
        state = torch.load(shard, map_location="meta", weights_only=True)
        if matches(_unwrap_pytorch_state_dict(state)):
            return True
    return False


@dataclass(frozen=True)
class Glm4CheckpointLayout:
    num_layers: int
    non_target_layers: int
    routed_layers: frozenset
    dense_width: int
    shared_width: int
    use_qk_norm: bool
    correction_bias: bool

    @classmethod
    def read(cls, config):
        path = Path(config.ckpt_path)
        hf = json.loads((path / "config.json").read_text())
        if hf.get("model_type", "glm4_moe") != "glm4_moe":
            raise ValueError("expected a non-Lite glm4_moe checkpoint")
        # num_layers may be reduced by the deployment, so prefer the declared
        # HF depth for distinguishing dropped target layers from draft layers.
        target = _integer(config.num_layers, "num_layers", 1)
        depth = _integer(
            hf.get("num_hidden_layers", target), "num_hidden_layers", target
        )
        draft = _integer(
            hf.get("num_nextn_predict_layers", 0), "num_nextn_predict_layers", 0
        )
        first = _integer(hf["first_k_dense_replace"], "first_k_dense_replace", 0)
        if first > depth:
            raise ValueError("GLM4 first_k_dense_replace exceeds checkpoint depth")
        routed = frozenset(range(first, target))
        if routed != {i for i in config.moe_layer_index if i < target}:
            raise ValueError(
                "GLM4 first_k_dense_replace disagrees with ModelConfig.moe_layer_index"
            )
        dense = (
            _integer(hf.get("intermediate_size"), "dense intermediate_size", 1)
            if first
            else 0
        )
        shared = (
            _integer(hf["n_shared_experts"], "n_shared_experts", 0)
            * config.moe_inter_size
        )
        qknorm = hf.get("use_qk_norm", config.qk_norm)
        if type(qknorm) is not bool:
            raise TypeError("GLM4 use_qk_norm must be bool")
        correction = hf.get("topk_method") == "noaux_tc" or has_correction_bias(
            path, target
        )
        return cls(
            target, depth + draft - target, routed, dense, shared, qknorm, correction
        )

    def target_owns(self, name):
        """Skip declared non-target layers; unknown names still fail strict dispatch."""
        parts = name.removeprefix("model.").split(".", 2)
        return not (
            len(parts) == 3
            and parts[0] == "layers"
            and parts[1].isdigit()
            and self.num_layers
            <= int(parts[1])
            < self.num_layers + self.non_target_layers
        )
