"""Validate the split-expert Qwen4 ModelOpt mixed-precision checkpoint.

The target experts are stored as NVFP4, while the optional MTP experts and
PLE table retain their own FP8 formats.  Validation does not convert tensors
or imply that the MTP runtime supports the target's NVFP4 recipe.
"""

import json
import re
import struct
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

_PLE_MODULE = re.compile(
    r"model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding$"
)
_PLE_WEIGHT = re.compile(
    r"(model\.language_model\.layers\.\d+\.ple\.ple_embedding\."
    r"ngram_embedding)\.shard_(\d+)\.weight$"
)
_DRAFT_MODULE = "mtp.layers.0.mlp.experts"
_PROJECTIONS = ("gate", "up", "down")
_NVFP4_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def is_qwen4_nvfp4_checkpoint(config_json: dict) -> bool:
    """Recognize the family; the validator rejects unsupported mixed recipes."""
    if not isinstance(config_json, Mapping):
        return False
    text = config_json.get("text_config") or {}
    quant = config_json.get("quantization_config") or {}
    return (
        isinstance(text, Mapping)
        and isinstance(quant, Mapping)
        and (
            config_json.get("model_type") == "qwen4_exp"
            or text.get("model_type") == "qwen4_exp_text"
        )
        and quant.get("quant_method") == "modelopt"
        and quant.get("quant_algo") == "MIXED_PRECISION"
    )


def _positive_int(text, name):
    value = text.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Qwen4 NVFP4 requires positive integer {name}")
    return value


def _group_spec(algo):
    if algo == "FP8":
        return {"weights": {"dynamic": False, "num_bits": 8, "type": "float"}}
    bits, size = (4, 16) if algo == "NVFP4" else (8, 128)
    weights = {
        "dynamic": False,
        "num_bits": bits,
        "type": "float",
        "group_size": size,
    }
    activations = dict(weights, dynamic=algo == "FP8_PB_WO")
    return {"weights": weights, "input_activations": activations}


def _quantized_modules(config_json):
    if not is_qwen4_nvfp4_checkpoint(config_json):
        raise ValueError("Qwen4 NVFP4 requires ModelOpt MIXED_PRECISION metadata")
    text = config_json.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("Qwen4 NVFP4 requires text_config geometry")
    layers = _positive_int(text, "num_hidden_layers")
    experts = _positive_int(text, "num_experts")
    quant = config_json["quantization_config"]
    modules = quant.get("quantized_layers")
    groups = quant.get("config_groups")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Qwen4 NVFP4 requires explicit quantized_layers")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("Qwen4 NVFP4 requires explicit config_groups")

    target = {
        f"model.language_model.layers.{layer}.mlp.experts" for layer in range(layers)
    }
    ple = set()
    for name, spec in modules.items():
        if not isinstance(name, str) or not isinstance(spec, Mapping):
            raise ValueError("Qwen4 NVFP4 malformed quantized_layers entry")
        if name in target:
            expected = {"quant_algo": "NVFP4", "group_size": 16}
        elif name == _DRAFT_MODULE:
            expected = {"quant_algo": "FP8_PB_WO", "group_size": 128}
        elif (match := _PLE_MODULE.fullmatch(name)) and int(match[1]) < layers:
            expected = {"quant_algo": "FP8"}
            ple.add(name)
        else:
            raise ValueError(f"Qwen4 NVFP4 unsupported quantized module: {name}")
        if spec != expected:
            raise ValueError(
                f"Qwen4 NVFP4 unsupported quantization specification: {name}"
            )
    if not target.issubset(modules):
        raise ValueError("Qwen4 NVFP4 quantized_layers must cover every target layer")
    if len(ple) > 1:
        raise ValueError("Qwen4 NVFP4 supports at most one FP8 PLE module")

    grouped = set()
    for group in groups.values():
        if not isinstance(group, Mapping):
            raise ValueError("Qwen4 NVFP4 malformed config_groups entry")
        targets = group.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("Qwen4 NVFP4 config_groups requires explicit targets")
        for name in targets:
            if not isinstance(name, str) or name not in modules or name in grouped:
                raise ValueError(
                    "Qwen4 NVFP4 config_groups targets disagree with quantized_layers"
                )
            expected = _group_spec(modules[name]["quant_algo"])
            actual = {key: value for key, value in group.items() if key != "targets"}
            if actual != expected:
                raise ValueError(
                    f"Qwen4 NVFP4 config_groups specification disagrees: {name}"
                )
            grouped.add(name)
    if grouped != set(modules):
        raise ValueError(
            "Qwen4 NVFP4 config_groups must cover exactly quantized_layers"
        )
    return target, _DRAFT_MODULE in modules, ple, experts


def _manifest_tensors(config_json, weight_map):
    target, has_draft, ple, experts = _quantized_modules(config_json)
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Qwen4 NVFP4 requires a non-empty weight_map")
    if not all(isinstance(name, str) for name in weight_map):
        raise ValueError("Qwen4 NVFP4 weight_map names must be strings")
    keys = set(weight_map)
    nvfp4 = {
        f"{module}.{expert}.{projection}_proj.{suffix}"
        for module in target
        for expert in range(experts)
        for projection in _PROJECTIONS
        for suffix in _NVFP4_SUFFIXES
    }
    draft = (
        {
            f"{_DRAFT_MODULE}.{expert}.{projection}_proj.{suffix}"
            for expert in range(experts)
            for projection in _PROJECTIONS
            for suffix in ("weight", "weight_scale_inv")
        }
        if has_draft
        else set()
    )
    if not nvfp4.issubset(keys):
        raise ValueError(
            "Qwen4 NVFP4 manifest is missing target expert projections or companions"
        )
    if not draft.issubset(keys):
        raise ValueError(
            "Qwen4 NVFP4 manifest is missing draft FP8 projections or companions"
        )

    ple_weights = {name for name in keys if _PLE_WEIGHT.fullmatch(name)}
    if {name.rsplit(".shard_", 1)[0] for name in ple_weights} != ple:
        raise ValueError("Qwen4 NVFP4 PLE manifest disagrees with quantized_layers")
    for module in ple:
        indices = sorted(
            int(_PLE_WEIGHT.fullmatch(name)[2])
            for name in ple_weights
            if name.startswith(module + ".")
        )
        if indices != list(range(len(indices))):
            raise ValueError(
                "Qwen4 NVFP4 PLE shard indices must be contiguous from zero"
            )
    ple_scales = {module + ".weight_scale" for module in ple}
    if not ple_scales.issubset(keys):
        raise ValueError("Qwen4 NVFP4 PLE table is missing its global scale")
    known = nvfp4 | draft | ple_weights | ple_scales
    for name in keys - known:
        suffix = name.rsplit(".", 1)[-1]
        if ".mlp.experts." in name or suffix.startswith(
            ("weight_scale", "input_scale")
        ):
            raise ValueError(
                f"Qwen4 NVFP4 unexpected expert or quantization tensor: {name}"
            )
    return nvfp4, draft, ple_weights, ple_scales


def validate_moe_nvfp4_manifest(config_json: dict, weight_map: Mapping) -> None:
    """Validate complete namespaces without opening checkpoint shard files."""
    _manifest_tensors(config_json, weight_map)


def _tensor_contract(name, nvfp4, draft, ple_weights, ple_scales, hidden, intermediate):
    if name in ple_scales:
        return ("BF16", "F32"), [1], True
    if name in ple_weights:
        return ("F8_E4M3",), None, False
    if name not in nvfp4 and name not in draft:
        return None
    n, k = (hidden, intermediate) if ".down_proj." in name else (intermediate, hidden)
    suffix = name.rsplit(".", 1)[-1]
    if name in draft:
        if suffix == "weight_scale_inv":
            return ("BF16", "F32"), [(n + 127) // 128, (k + 127) // 128], False
        return ("F8_E4M3",), [n, k], False
    if suffix == "weight":
        return ("U8",), [n, k // 2], False
    if suffix == "weight_scale":
        return ("F8_E4M3",), [n, k // 16], False
    return ("F32",), [1], True


def validate_moe_nvfp4_checkpoint(config_json: dict, weight_map: Mapping, root) -> None:
    """Validate shard headers and small scalar scales, never materialize weights.

    safe_open verifies offsets against file sizes.  Only per-tensor scalar
    scales are read; expert/PLE weights and block scale arrays remain unmapped
    as tensors.  Cross-expert activation-scale and gate/up-scale compatibility
    is a separate loader/consumer contract.
    """
    namespaces = _manifest_tensors(config_json, weight_map)
    text = config_json["text_config"]
    hidden = _positive_int(text, "hidden_size")
    intermediate = _positive_int(text, "moe_intermediate_size")
    if hidden % 16 or intermediate % 16:
        raise ValueError(
            "Qwen4 NVFP4 expert dimensions must be multiples of group size 16"
        )
    root = Path(root).resolve()
    by_shard = {}
    for name, shard in weight_map.items():
        if not isinstance(shard, str):
            raise ValueError("Qwen4 NVFP4 requires relative safetensors shard paths")
        path = Path(shard)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
            raise ValueError("Qwen4 NVFP4 requires relative safetensors shard paths")
        if not (root / path).resolve().is_relative_to(root):
            raise ValueError("Qwen4 NVFP4 shard path escapes checkpoint directory")
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        path = root / shard
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            with path.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(header_size))
            if set(tensors.keys()) != set(names):
                raise ValueError(f"Qwen4 NVFP4 shard/index keys disagree: {shard}")
            for name in names:
                metadata = header[name]
                contract = _tensor_contract(name, *namespaces, hidden, intermediate)
                if contract is None:
                    if metadata["dtype"].startswith("F8_") or (
                        name.endswith(".weight") and metadata["dtype"] == "U8"
                    ):
                        raise ValueError(
                            f"Qwen4 NVFP4 undeclared quantized tensor: {name}"
                        )
                    continue
                dtypes, shape, scalar = contract
                actual_shape = metadata["shape"]
                shape_matches = (
                    actual_shape == shape
                    if shape is not None
                    else len(actual_shape) == 2 and min(actual_shape) > 0
                )
                if metadata["dtype"] not in dtypes or not shape_matches:
                    raise ValueError(f"Qwen4 NVFP4 tensor dtype/shape mismatch: {name}")
                if scalar:
                    value = tensors.get_tensor(name).float()
                    if not bool(torch.isfinite(value).all()) or not bool(
                        (value > 0).all()
                    ):
                        raise ValueError(
                            "Qwen4 NVFP4 scalar scale must be finite and positive: "
                            f"{name}"
                        )
