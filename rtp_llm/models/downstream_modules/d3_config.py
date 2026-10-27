"""D3 code-readout checkpoint sidecar contract.

A d3 checkpoint is a Qwen3.5 backbone whose full-attention layers read the
whole prompt (the Gated DeltaNet layers keep their causal scan) plus a
trained ``readout.safetensors`` fp32 ``[255, hidden]`` weight that scores 255
single-token answer codes at the last prompt position.
``decision_config.json`` beside the weights carries the format identity, the
answer codes with their token ids, the attention mode, pooling, temperature,
the input limit and the readout dtype.

Importing this module loads neither PyTorch nor the native RTP-LLM runtime:
the model factory, the frontend executor and the downstream module all parse
the sidecar through it. The answer codes are verified against a real
tokenizer only where one exists (frontend executor / downstream module),
never here. ``MODEL_MANIFEST.json`` provenance stays with the package; d3,
JEV and Kev sidecars are mutually exclusive (see downstream
``utils.create_custom_module``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
FORMAT_ID = "d3-code-readout-v1"
PROMPTS = ("d3",)
ATTENTION_MODES = ("causal", "noncausal_full_attention")
POOLINGS = ("last",)
READOUT_DTYPES = ("float32", "bfloat16")
MAX_OPTIONS = 255
SIDE_FILE = "decision_config.json"
READOUT_FILE = "readout.safetensors"
READOUT_KEY = "weight"
# This engine path wires the Qwen3.5 dense backbone (Qwen35Dense). The
# official runtime also knows a qwen3_vl backbone, which RTP routes through a
# different model class that has no d3 support yet.
BACKBONES = ("qwen3_5",)
LAYER_TYPES = ("linear_attention", "full_attention")


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must contain distinct nonempty strings")
    return tuple(value)


def _integers(value: Any, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item < 0 for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must contain distinct nonnegative integers")
    return tuple(value)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return float(value)


@dataclass(frozen=True)
class D3Config:
    format_version: int
    format_id: str
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: float
    attention_mode: str
    pooling: str
    max_length: int | None
    readout_dtype: str
    hidden_size: int
    readout_file: str = READOUT_FILE
    readout_key: str = READOUT_KEY

    @classmethod
    def load(cls, checkpoint: str | Path) -> D3Config:
        """Read and validate ``decision_config.json`` and the backbone config.

        Raises ValueError (including JSONDecodeError) for a malformed or
        incompatible checkpoint; filesystem errors propagate. The sidecar
        fields follow the official d3 runtime, including their defaults.
        """
        root = Path(checkpoint)
        with (root / SIDE_FILE).open() as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"{SIDE_FILE} must contain a JSON object")
        for key, expected in (
            ("format_version", FORMAT_VERSION),
            ("format_id", FORMAT_ID),
            ("prompt", PROMPTS[0]),
        ):
            if type(data.get(key)) is not type(expected) or data[key] != expected:
                raise ValueError(f"{SIDE_FILE} requires {key}={expected!r}")
        attention_mode = data.get("attention_mode", "causal")
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"{SIDE_FILE}: attention_mode must be one of "
                f"{', '.join(ATTENTION_MODES)}, got {attention_mode!r}"
            )
        pooling = data.get("pooling", "last")
        if pooling not in POOLINGS:
            raise ValueError(
                f"{SIDE_FILE}: pooling must be one of {', '.join(POOLINGS)}, "
                f"got {pooling!r}"
            )
        readout_dtype = data.get("readout_dtype", "float32")
        if readout_dtype not in READOUT_DTYPES:
            raise ValueError(
                f"{SIDE_FILE}: readout_dtype must be one of "
                f"{', '.join(READOUT_DTYPES)}, got {readout_dtype!r}"
            )
        temperature = _number(
            data.get("temperature", 1.0), f"{SIDE_FILE}: temperature", positive=True
        )
        limit = data.get("max_length")
        if limit is None:
            max_length = None
        else:
            max_length = _integer(limit, f"{SIDE_FILE}: max_length")
        codes = _strings(data.get("codes"), f"{SIDE_FILE}: codes")
        token_ids = _integers(data.get("token_ids"), f"{SIDE_FILE}: token_ids")
        if len(codes) != MAX_OPTIONS or len(token_ids) != MAX_OPTIONS:
            raise ValueError(
                f"{SIDE_FILE}: codes and token_ids must each hold {MAX_OPTIONS} entries"
            )
        hidden_size = cls._load_hidden_size(root, attention_mode)
        return cls(
            format_version=FORMAT_VERSION,
            format_id=FORMAT_ID,
            codes=codes,
            token_ids=token_ids,
            temperature=temperature,
            attention_mode=attention_mode,
            pooling=pooling,
            max_length=max_length,
            readout_dtype=readout_dtype,
            hidden_size=hidden_size,
        )

    @staticmethod
    def _load_hidden_size(root: Path, attention_mode: str) -> int:
        """Validate the backbone identity in config.json and return its hidden size."""
        config_file = root / "config.json"
        if not config_file.is_file():
            raise ValueError(f"D3 checkpoints require a config.json beside {SIDE_FILE}")
        with config_file.open() as file:
            backbone = json.load(file)
        if not isinstance(backbone, dict):
            raise ValueError(f"config.json must contain a JSON object for D3")
        if backbone.get("model_type") not in BACKBONES:
            raise ValueError(
                f"D3 requires a {' or '.join(BACKBONES)} backbone; config.json "
                f"model_type is {backbone.get('model_type')!r}"
            )
        text = backbone.get("text_config")
        text_config = backbone if not isinstance(text, dict) else text
        layer_types = text_config.get("layer_types")
        if layer_types is not None:
            if (
                not isinstance(layer_types, list)
                or not layer_types
                or any(layer not in LAYER_TYPES for layer in layer_types)
            ):
                raise ValueError(
                    f"config.json layer_types must list {LAYER_TYPES} entries"
                )
            if (
                attention_mode == "noncausal_full_attention"
                and "full_attention" not in layer_types
            ):
                raise ValueError(
                    "noncausal_full_attention requires full_attention layers "
                    "in config.json layer_types"
                )
        return _integer(
            text_config.get("hidden_size"), "config.json text_config.hidden_size"
        )


def load_d3_config(checkpoint: str | Path) -> D3Config | None:
    """Read a validated D3 sidecar, or None for an ordinary checkpoint.

    A malformed existing sidecar raises ValueError (including JSONDecodeError);
    filesystem errors propagate. This function needs no native runtime.
    """
    if not (Path(checkpoint) / SIDE_FILE).is_file():
        return None
    return D3Config.load(checkpoint)
