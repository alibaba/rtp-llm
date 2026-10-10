"""Checkpoint contract for the selective Qwen4 expert FP8 path."""

import json
import re
import struct
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.ffn_weight import MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.utils.model_weight import W

_EXPERT_WEIGHT = re.compile(
    r"(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate|up|down)_proj\.weight$"
)
_PLE_WEIGHT = re.compile(
    r"(?:model\.language_model|mtp)\.layers\.\d+\.ple\.ple_embedding\."
    r"ngram_embedding\.shard_\d+\.weight$"
)


def validate_moe_fp8_manifest(config_json: dict, weight_map: Mapping) -> None:
    quant = config_json.get("quantization_config") or {}
    if (
        quant.get("quant_method") != "fp8"
        or quant.get("activation_scheme") != "dynamic"
        or quant.get("weight_block_size") != [128, 128]
        or quant.get("weight_per_tensor", False)
        or quant.get("act_per_tensor", False)
    ):
        raise ValueError("Qwen4 expert FP8 requires dynamic 128x128 block quantization")
    if not quant.get("modules_to_not_convert"):
        raise ValueError("Qwen4 expert FP8 requires explicit non-expert exclusions")
    keys = set(weight_map)
    experts = {key for key in keys if _EXPERT_WEIGHT.fullmatch(key)}
    scales = {key for key in keys if key.endswith("_scale_inv")}
    expected_scales = {key + "_scale_inv" for key in experts}
    if not experts or scales != expected_scales:
        raise ValueError("Qwen4 FP8 scales must cover exactly the split MoE experts")
    text = config_json["text_config"]
    expected_target = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        for layer in range(int(text["num_hidden_layers"]))
        for expert in range(int(text["num_experts"]))
        for projection in ("gate", "up", "down")
    }
    target = {key for key in experts if key.startswith("model.language_model.")}
    if target != expected_target:
        raise ValueError("Qwen4 FP8 manifest is missing target expert projections")
    # Validate the complete one-layer MTP namespace if the checkpoint includes it.
    draft = {key for key in experts if key.startswith("mtp.")}
    expected_draft = {
        f"mtp.layers.0.mlp.experts.{expert}.{projection}_proj.weight"
        for expert in range(int(text["num_experts"]))
        for projection in ("gate", "up", "down")
    }
    if draft and draft != expected_draft:
        raise ValueError("Qwen4 FP8 manifest is missing draft expert projections")


def validate_moe_fp8_checkpoint(config_json: dict, weight_map: Mapping, root) -> None:
    """Check the stored tensor contract before materializing any weights.

    The official split experts use E4M3 weights and BF16 or FP32 block scales.
    Read headers only; safe_open also verifies their offsets against file sizes.
    """
    validate_moe_fp8_manifest(config_json, weight_map)
    text = config_json["text_config"]
    hidden = int(text["hidden_size"])
    intermediate = int(text["moe_intermediate_size"])
    if hidden <= 0 or intermediate <= 0:
        raise ValueError("Qwen4 FP8 expert dimensions must be positive")
    by_shard = {}
    ple_scales = set()
    for name in weight_map:
        if _PLE_WEIGHT.fullmatch(name):
            scale_name = name.rsplit(".shard_", 1)[0] + ".weight_scale"
            if scale_name not in weight_map:
                raise ValueError("Qwen4 FP8 PLE table is missing its global scale")
            ple_scales.add(scale_name)
    for name, shard in weight_map.items():
        path = Path(shard)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
            raise ValueError("Qwen4 FP8 requires relative safetensors shard paths")
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        path = Path(root) / shard
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            with path.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(header_size))
            if set(tensors.keys()) != set(names):
                raise ValueError(f"Qwen4 FP8 shard/index keys disagree: {shard}")
            for name in names:
                metadata = header[name]
                if name in ple_scales:
                    if metadata["dtype"] not in ("BF16", "F32") or metadata[
                        "shape"
                    ] != [1]:
                        raise ValueError(
                            "Qwen4 FP8 PLE global scale dtype/shape mismatch"
                        )
                    scale = tensors.get_tensor(name).float()
                    if not bool(torch.isfinite(scale).all()) or not bool(
                        (scale > 0).all()
                    ):
                        raise ValueError(
                            "Qwen4 FP8 PLE global scale must be finite and positive"
                        )
                    continue
                if _PLE_WEIGHT.fullmatch(name):
                    shape = metadata["shape"]
                    if (
                        metadata["dtype"] != "F8_E4M3"
                        or len(shape) != 2
                        or min(shape) <= 0
                    ):
                        raise ValueError("Qwen4 FP8 PLE table dtype/shape mismatch")
                    continue
                weight_name = name.removesuffix("_scale_inv")
                if not _EXPERT_WEIGHT.fullmatch(weight_name):
                    if metadata["dtype"].startswith("F8_"):
                        raise ValueError(f"Qwen4 FP8 non-expert tensor: {name}")
                    continue
                shape = (
                    [hidden, intermediate]
                    if weight_name.endswith("down_proj.weight")
                    else [intermediate, hidden]
                )
                if name.endswith("_scale_inv"):
                    shape = [(size + 127) // 128 for size in shape]
                    allowed_dtypes = ("BF16", "F32")
                else:
                    allowed_dtypes = ("F8_E4M3",)
                if (
                    metadata["dtype"] not in allowed_dtypes
                    or metadata["shape"] != shape
                ):
                    raise ValueError(f"Qwen4 FP8 tensor dtype/shape mismatch: {name}")


class Qwen4MoeFp8WeightInfo(ModelWeightInfo):
    """Keep attention, GDN, shared experts, PLE and HC in their BF16 descriptors."""

    def to_quant_weight_info(self, quant_config):
        if (
            not isinstance(quant_config, Fp8BlockWiseQuantConfig)
            or not quant_config.is_quanted()
            or quant_config.group_size() != 128
        ):
            raise ValueError("Qwen4 expert FP8 requires prequantized 128x128 weights")
        layers = []
        for weights in self.layer_weights:
            layer = []
            for weight in weights:
                if isinstance(weight, MoeWeight):
                    children = [
                        (
                            child.create(child, quant_config)
                            if child.name in (W.moe_w1, W.moe_w2)
                            else child
                        )
                        for child in weight.sub_weights.values()
                    ]
                    weight = MoeWeight(sub_weights=children, config=weight.config)
                layer.append(weight)
            layers.append(layer)
        return ModelWeightInfo(list(self.weights), layers)
