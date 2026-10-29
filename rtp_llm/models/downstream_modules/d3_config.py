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
# Joint inline-readout checkpoints (pplx-decider-v1.2 and successors) declare
# decision_config.json format_version 2 plus a joint_config.json protocol
# block; the sidecar fields follow the official pplx_decider package.
JOINT_FORMAT_VERSION = 2
JOINT_CONFIG_FILE = "joint_config.json"
JOINT_PROTOCOL_VERSION = 3
JOINT_TEMPLATE = "joint-question-inline-readout-v1"
JOINT_MARKER = "<|fim_suffix|>"
JOINT_READOUT_OFFSET = 0
JOINT_USAGE = "nonpadding_joint_sequence_including_template_and_image_positions"
JOINT_BACKBONE_LIMIT = 262_144
# Official image pixel bounds of the joint protocol: the smaller bound is the
# processor floor for every image, the larger the default cap when the sidecar
# carries no max_pixels (text-trained checkpoints).
JOINT_MIN_PIXELS = 65_536
JOINT_LEGACY_MAX_PIXELS = 262_144
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
class D3JointConfig:
    """The joint inline-readout protocol of a version-2 sidecar.

    The fields mirror the official joint_config.json, which the official
    runtime requires to equal its protocol constant exactly; a joint prompt
    holds the state and every question of a request, one ``<|fim_suffix|>``
    marker per question, with the readout applied at each marker.
    """

    marker: str
    readout_offset: int
    max_questions: int | None
    max_pixels: int | None

    @classmethod
    def load(cls, checkpoint: str | Path) -> D3JointConfig:
        """Read and validate joint_config.json beside the weights.

        Raises ValueError when the file is missing, malformed or does not
        declare the joint inline-readout protocol this runtime serves.
        """
        root = Path(checkpoint)
        protocol_file = root / JOINT_CONFIG_FILE
        if not protocol_file.is_file():
            raise ValueError(f"joint decision_config requires {JOINT_CONFIG_FILE}")
        with protocol_file.open() as file:
            protocol = json.load(file)
        expected = {
            "version": JOINT_PROTOCOL_VERSION,
            "template": JOINT_TEMPLATE,
            "marker": JOINT_MARKER,
            "readout_offset": JOINT_READOUT_OFFSET,
            "max_questions": None,
            "max_length": None,
            "usage": JOINT_USAGE,
        }
        if not isinstance(protocol, dict) or protocol != expected:
            raise ValueError(
                f"{JOINT_CONFIG_FILE} must declare the joint inline-readout "
                f"protocol {expected}"
            )
        return cls(
            marker=JOINT_MARKER,
            readout_offset=JOINT_READOUT_OFFSET,
            max_questions=None,
            max_pixels=None,
        )


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
    joint: D3JointConfig | None = None
    max_pixels: int | None = None

    @classmethod
    def load(cls, checkpoint: str | Path) -> D3Config:
        """Read and validate ``decision_config.json`` and the backbone config.

        Raises ValueError (including JSONDecodeError) for a malformed or
        incompatible checkpoint; filesystem errors propagate. Version-1
        sidecars follow the official d3 runtime; version-2 sidecars follow the
        official joint inline-readout protocol (pplx_decider), including
        their defaults.
        """
        root = Path(checkpoint)
        with (root / SIDE_FILE).open() as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"{SIDE_FILE} must contain a JSON object")
        version = data.get("format_version")
        if type(version) is not int:
            raise ValueError(f"{SIDE_FILE} requires an integer format_version")
        if version not in (FORMAT_VERSION, JOINT_FORMAT_VERSION):
            raise ValueError(
                f"{SIDE_FILE} format_version {version} is not supported "
                f"(this runtime reads {FORMAT_VERSION} and {JOINT_FORMAT_VERSION})"
            )
        joint: D3JointConfig | None = None
        if version == FORMAT_VERSION:
            for key, expected in (
                ("format_version", FORMAT_VERSION),
                ("format_id", FORMAT_ID),
                ("prompt", PROMPTS[0]),
            ):
                if type(data.get(key)) is not type(expected) or data[key] != expected:
                    raise ValueError(f"{SIDE_FILE} requires {key}={expected!r}")
        else:
            # A joint sidecar identifies itself through joint_config.json
            # instead of the version-1 format_id/prompt fields.
            joint = D3JointConfig.load(root)
        attention_mode = data.get("attention_mode", "causal")
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"{SIDE_FILE}: attention_mode must be one of "
                f"{', '.join(ATTENTION_MODES)}, got {attention_mode!r}"
            )
        if version == JOINT_FORMAT_VERSION and attention_mode != "noncausal_full_attention":
            raise ValueError(
                f"{SIDE_FILE}: the joint protocol requires "
                f"attention_mode=noncausal_full_attention, got {attention_mode!r}"
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
        if version == JOINT_FORMAT_VERSION:
            if max_length is not None:
                raise ValueError(
                    f"{SIDE_FILE}: the joint protocol carries its length limit "
                    f"in {JOINT_CONFIG_FILE}, not max_length"
                )
            max_pixels = data.get("max_pixels")
            if max_pixels is not None:
                max_pixels = _integer(max_pixels, f"{SIDE_FILE}: max_pixels")
        else:
            max_pixels = None
        codes = _strings(data.get("codes"), f"{SIDE_FILE}: codes")
        token_ids = _integers(data.get("token_ids"), f"{SIDE_FILE}: token_ids")
        if len(codes) != MAX_OPTIONS or len(token_ids) != MAX_OPTIONS:
            raise ValueError(
                f"{SIDE_FILE}: codes and token_ids must each hold {MAX_OPTIONS} entries"
            )
        hidden_size = cls._load_hidden_size(root, attention_mode)
        if version == JOINT_FORMAT_VERSION:
            # The joint protocol's own length limit is unbounded (null); the
            # backbone's position limit is the default per-sequence bound.
            max_length = min(
                cls._load_position_limit(root), JOINT_BACKBONE_LIMIT
            )
        return cls(
            format_version=version,
            format_id=(
                FORMAT_ID if version == FORMAT_VERSION else JOINT_TEMPLATE
            ),
            codes=codes,
            token_ids=token_ids,
            temperature=temperature,
            attention_mode=attention_mode,
            pooling=pooling,
            max_length=max_length,
            readout_dtype=readout_dtype,
            hidden_size=hidden_size,
            joint=joint,
            max_pixels=max_pixels,
        )

    @staticmethod
    def _load_position_limit(root: Path) -> int:
        """The backbone's max_position_embeddings, the official joint default bound."""
        config_file = root / "config.json"
        if not config_file.is_file():
            raise ValueError(f"D3 checkpoints require a config.json beside {SIDE_FILE}")
        with config_file.open() as file:
            backbone = json.load(file)
        text = backbone.get("text_config") if isinstance(backbone, dict) else None
        text_config = backbone if not isinstance(text, dict) else text
        position_limit = text_config.get("max_position_embeddings")
        if position_limit is None:
            return JOINT_BACKBONE_LIMIT
        return _integer(
            position_limit, "config.json text_config.max_position_embeddings"
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
