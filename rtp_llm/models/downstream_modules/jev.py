"""JEV System 1 checkpoint contract and prefill-only label readout.

The backbone supplies final-normalized activations, including expanded image
positions. This module projects only the final token of each causal row. Task
selection, calibration and normalization belong to the frontend: the engine
returns raw logits for every exported label, in manifest order.

Manifest inspection imports neither PyTorch nor the native RTP-LLM runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


KINDS = ("noul", "score", "choice")
HEAD_FILE = "jev_head.safetensors"
HEAD_WEIGHT_KEY = "projection.weight"


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
class JevKindConfig:
    row_indices: tuple[int, ...]
    label_names: tuple[str, ...]
    token_ids: tuple[int, ...]
    bias: tuple[float, ...]
    temperature: float
    native_slots: int
    max_options: int


@dataclass(frozen=True)
class JevConfig:
    hidden_size: int
    num_labels: int
    labels: tuple[str, ...]
    token_ids: tuple[int, ...]
    kinds: Mapping[str, JevKindConfig]
    max_context_tokens: int
    backbone_architecture: str
    head_file: str = HEAD_FILE
    head_weight_key: str = HEAD_WEIGHT_KEY
    protocol: str = "jev27-bare-v1"
    template_version: str = "bare-v1"
    prefix: str = ""
    default_strategy: str = "single"

    @classmethod
    def from_dict(cls, data: Any) -> JevConfig:
        if not isinstance(data, dict):
            raise ValueError("jev_config.json must contain an object")
        for key, expected in (
            ("schema_version", 1),
            ("model_type", "jev"),
            ("mode", "system1"),
            ("head_file", HEAD_FILE),
            ("head_weight_key", HEAD_WEIGHT_KEY),
            ("template_version", "bare-v1"),
            ("protocol", "jev27-bare-v1"),
        ):
            if type(data.get(key)) is not type(expected) or data[key] != expected:
                raise ValueError(f"jev_config.json requires {key}={expected!r}")
        for key, expected in (
            ("head_dtype", "float32"),
            ("head_includes_bias", False),
            ("thinking", "off"),
        ):
            value = data.get(key, expected)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"JEV System 1 requires {key}={expected!r}")
        hidden_size = _integer(data.get("hidden_size"), "hidden_size")
        num_labels = _integer(data.get("num_labels"), "num_labels")
        labels = _strings(data.get("labels"), "labels")
        token_ids = _integers(data.get("token_ids"), "token_ids")
        if (
            num_labels != len(labels)
            or num_labels != len(token_ids)
            or num_labels > 264
        ):
            raise ValueError("num_labels must match labels/token_ids and be <= 264")
        max_context = _integer(data.get("max_context_tokens"), "max_context_tokens")
        architecture = data.get("backbone_architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("backbone_architecture must be a nonempty string")
        prefix = data.get("prefix", "")
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")
        strategy = data.get("default_strategy", "single")
        if strategy not in ("single", "permute", "tournament"):
            raise ValueError("unsupported default_strategy")
        kinds = data.get("kinds")
        if not isinstance(kinds, dict) or set(kinds) != set(KINDS):
            raise ValueError("kinds must contain exactly noul, score and choice")
        parsed = {}
        cursor = 0
        for name in KINDS:
            kind = kinds[name]
            if not isinstance(kind, dict):
                raise ValueError(f"kinds.{name} must be an object")
            rows = _integers(kind.get("row_indices"), f"kinds.{name}.row_indices")
            names = _strings(kind.get("label_names"), f"kinds.{name}.label_names")
            ids = _integers(kind.get("token_ids"), f"kinds.{name}.token_ids")
            count = len(rows)
            if name == "noul" and names != ("false", "true"):
                raise ValueError("noul label_names must be false, true")
            if name == "score" and names != tuple(str(i) for i in range(6)):
                raise ValueError("score label_names must be 0 through 5")
            if (
                rows != tuple(range(cursor, cursor + count))
                or cursor + count > num_labels
                or names != labels[cursor : cursor + count]
                or ids != token_ids[cursor : cursor + count]
            ):
                raise ValueError(f"kinds.{name} does not match ordered label rows")
            bias = kind.get("bias")
            if not isinstance(bias, list) or len(bias) != count:
                raise ValueError(f"kinds.{name}.bias must have one value per row")
            bias = tuple(_number(value, f"kinds.{name}.bias") for value in bias)
            temperature = _number(
                kind.get("temperature"), f"kinds.{name}.temperature", positive=True
            )
            native_slots = _integer(kind.get("native_slots", count), "native_slots")
            max_options = _integer(kind.get("max_options", count), "max_options")
            if name == "choice":
                if native_slots != 16 or not 16 <= count <= 256 or max_options != count:
                    raise ValueError(
                        "choice requires 16 native slots and 16..256 label rows"
                    )
                if any(bias[16:]):
                    raise ValueError("untrained choice labels must have zero bias")
            elif native_slots != count or max_options != count:
                raise ValueError(f"kinds.{name} slot counts must match its labels")
            parsed[name] = JevKindConfig(
                rows, names, ids, bias, temperature, native_slots, max_options
            )
            cursor += count
        if cursor != num_labels:
            raise ValueError("kind rows must cover every exported label")
        return cls(
            hidden_size=hidden_size,
            num_labels=num_labels,
            labels=labels,
            token_ids=token_ids,
            kinds=parsed,
            max_context_tokens=max_context,
            backbone_architecture=architecture,
            prefix=prefix,
            default_strategy=strategy,
        )

    @classmethod
    def load(cls, checkpoint: str | Path) -> JevConfig:
        with (Path(checkpoint) / "jev_config.json").open() as file:
            return cls.from_dict(json.load(file))


def load_jev_config(checkpoint: str | Path) -> dict[str, Any] | None:
    """Read validated metadata, preserving converter provenance fields."""
    path = Path(checkpoint) / "jev_config.json"
    if not path.exists():
        return None
    with path.open() as file:
        data = json.load(file)
    JevConfig.from_dict(data)
    return data


class JevHandler:
    prefill_only = True
    supports_prefix_cache = True
    need_post_process = False

    def __init__(self, config: JevConfig):
        self.config = config
        self._weight: torch.Tensor | None = None
        self._device: torch.device | None = None

    def custom_weight_info(self) -> list[Any]:
        # The generic weight loader casts backbone weights; the readout stays FP32.
        return []

    def init_cpp_handler(self) -> None:
        pass

    def init(self, tensors: Mapping[str, torch.Tensor]) -> None:
        import torch

        if set(tensors) != {self.config.head_weight_key}:
            raise ValueError("JEV head must contain exactly projection.weight")
        weight = tensors[self.config.head_weight_key]
        shape = (self.config.num_labels, self.config.hidden_size)
        if weight.dtype != torch.float32 or tuple(weight.shape) != shape:
            raise ValueError(f"JEV projection.weight must be fp32 with shape {shape}")
        if not torch.isfinite(weight).all().item():
            raise ValueError("JEV projection.weight contains nonfinite weights")
        self._weight = weight.detach()
        self._device = None

    def extend_forward_args(self) -> list[str]:
        return ["hidden_states", "input_ids", "input_lengths", "prefix_lengths"]

    def select_token_position(
        self,
        input_ids: torch.Tensor,
        text_tokens_mask: torch.Tensor | None = None,
    ) -> int:
        """Select after multimodal expansion; image tokens may precede readout."""
        import torch

        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.device.type != "cpu"
            or input_ids.ndim != 1
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.numel() == 0
        ):
            raise ValueError(
                "JEV admission requires a nonempty one-dimensional CPU token row"
            )
        if input_ids.numel() > self.config.max_context_tokens:
            raise ValueError("JEV expanded row exceeds max_context_tokens")
        if text_tokens_mask is not None:
            if (
                not isinstance(text_tokens_mask, torch.Tensor)
                or text_tokens_mask.device.type != "cpu"
                or text_tokens_mask.ndim != 1
                or text_tokens_mask.numel() != input_ids.numel()
                or text_tokens_mask.dtype not in (torch.bool, torch.int32, torch.int64)
                or not bool(text_tokens_mask[-1].item())
            ):
                raise ValueError(
                    "JEV text_tokens_mask must align with expanded tokens and end in text"
                )
        return input_ids.numel() - 1

    def cache_prefix_length(self, input_ids: torch.Tensor) -> int:
        # Only the final hidden state is read. Keep that token in the suffix;
        # cache block rounding and multimodal interval handling belong to engine.
        return self.select_token_position(input_ids)

    def extend_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as functional

        if self._weight is None:
            raise RuntimeError("JEV head has not been initialized")
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.config.hidden_size
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                "JEV hidden_states must have shape [tokens, hidden_size] and floating dtype"
            )
        for name, vector in (
            ("input_ids", input_ids),
            ("input_lengths", input_lengths),
        ):
            if (
                not isinstance(vector, torch.Tensor)
                or vector.ndim != 1
                or vector.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(f"JEV {name} must be a one-dimensional integer tensor")
        if not input_lengths.numel() or input_ids.numel() != hidden_states.shape[0]:
            raise ValueError("JEV requires complete, unpadded prefill suffix rows")
        if prefix_lengths is not None and (
            not isinstance(prefix_lengths, torch.Tensor)
            or prefix_lengths.ndim != 1
            or prefix_lengths.dtype not in (torch.int32, torch.int64)
            or prefix_lengths.numel() not in (0, input_lengths.numel())
        ):
            raise ValueError("JEV prefix_lengths must contain one integer per row")

        def require(condition: torch.Tensor, message: str) -> None:
            if condition.device.type == "cpu":
                if not condition.item():
                    raise ValueError(message)
            else:
                # Do not synchronize or copy GPU IDs/lengths back to the CPU.
                torch._assert_async(condition, message)

        require((input_lengths > 0).all(), "JEV suffix lengths must be positive")
        require(
            input_lengths.sum() == hidden_states.shape[0],
            "JEV requires complete, unpadded prefill suffix rows",
        )
        lengths = input_lengths.to(device=hidden_states.device, dtype=torch.int64)
        if prefix_lengths is not None and prefix_lengths.numel():
            require(
                (prefix_lengths >= 0).all(), "JEV prefix_lengths must be nonnegative"
            )
            prefixes = prefix_lengths.to(device=hidden_states.device, dtype=torch.int64)
        else:
            prefixes = torch.zeros_like(lengths)
        require(
            (lengths + prefixes <= self.config.max_context_tokens).all(),
            "JEV expanded row exceeds max_context_tokens",
        )
        device = hidden_states.device
        if self._device != device:
            self._weight = self._weight.to(device=device)
            self._device = device
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            # Lengths describe expanded, executed suffixes, never raw prompt
            # tokenizer lengths. Prefix reuse cannot change their final token.
            last = hidden_states.index_select(0, lengths.cumsum(0) - 1).float()
            return functional.linear(last, self._weight)


class JevModule:
    """RTP custom module with a small, separately loaded FP32 label matrix."""

    def __init__(self, config: Any, tokenizer: Any):
        import torch

        if config.compute_dtype != torch.bfloat16:
            raise ValueError(
                "JEV requires BF16 backbone activations; set --act_type bf16"
            )
        quant = getattr(config, "quant_config", None)
        algo = config.quant_algo
        requested = getattr(config, "quantization", None)
        has_quant = (
            algo.isQuant()
            or quant is not None
            or str(requested or "").lower() not in ("", "none")
        )
        fp8_block = (
            quant is not None
            and getattr(quant, "get_method", lambda: None)() == "FP8_PER_BLOCK"
            and algo.isQuant()
            and getattr(algo, "isFp8", lambda: False)()
            and getattr(algo, "getWeightBits", lambda: None)() == 8
            and getattr(algo, "getGroupSize", lambda: None)() == 128
        )
        layered = (
            getattr(config, "enable_nvfp4_dense_ffn", False)
            or getattr(config, "enable_w4a16_sm120_dense_ffn", False)
            or getattr(quant, "enable_nvfp4_dense_ffn", False)
        )
        if (has_quant and not fp8_block) or layered:
            raise ValueError(
                "JEV supports only BF16 or FP8_PER_BLOCK backbone weights without layered quantization"
            )
        self.config_ = config
        self.tokenizer_ = tokenizer
        self.jev_config = JevConfig.load(config.ckpt_path)
        if config.hidden_size != self.jev_config.hidden_size:
            raise ValueError("JEV head hidden_size does not match the backbone")
        if tokenizer is not None:
            if hasattr(tokenizer, "get_real_tokenizer"):
                tokenizer = tokenizer.get_real_tokenizer()
            for label, token_id in zip(
                self.jev_config.labels, self.jev_config.token_ids
            ):
                if list(tokenizer.encode(label, add_special_tokens=False)) != [
                    token_id
                ]:
                    raise ValueError(
                        f"JEV label {label!r} does not match the checkpoint tokenizer"
                    )
        self.handler = JevHandler(self.jev_config)

    def get_handler(self) -> JevHandler:
        return self.handler

    def get_custom_weight_info(self) -> list[Any]:
        return []

    def init(self, weight: Any) -> None:
        from safetensors.torch import load_file

        tensors = load_file(
            str(Path(self.config_.ckpt_path) / self.jev_config.head_file)
        )
        self.handler.init(tensors)
