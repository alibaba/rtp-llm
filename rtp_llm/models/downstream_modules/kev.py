"""Kev's independent-row codec and fp32 pointer head.

The serving contract is intentionally small: a complete causal row per question,
optional state-prefix reuse, no chunked prefill, and probabilities rather than
generated tokens.
Importing the codec does not load RTP-LLM's native runtime or PyTorch.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import torch


SPECIAL = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _special_ids(values: Any) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != len(SPECIAL):
        raise ValueError("special_token_ids must contain state, q, opt, endopt, decide")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("special_token_ids must be nonnegative integers")
    if len(set(values)) != len(SPECIAL):
        raise ValueError("special_token_ids must be distinct")
    return tuple(values)


def kev_special_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Resolve the five delimiters from the real Hugging Face tokenizer."""
    if hasattr(tokenizer, "get_real_tokenizer"):
        tokenizer = tokenizer.get_real_tokenizer()
    return _special_ids([tokenizer.convert_tokens_to_ids(token) for token in SPECIAL])


@dataclass(frozen=True)
class KevConfig:
    hidden_size: int
    head_dim: int
    temperature: float
    special_token_ids: tuple[int, ...]
    max_options: int
    max_state_tokens: int
    max_row_tokens: int
    base_model: str
    base_revision: str
    source_revision: str

    @classmethod
    def load(cls, checkpoint: str | Path) -> KevConfig:
        with (Path(checkpoint) / "kev_config.json").open() as file:
            metadata = json.load(file)
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("format_version")) is not int
            or metadata["format_version"] != 1
        ):
            raise ValueError("kev_config.json requires format_version=1")
        hidden_size = _positive_int(metadata.get("hidden_size"), "hidden_size")
        head_dim = _positive_int(metadata.get("head_dim"), "head_dim")
        max_options = _positive_int(metadata.get("max_options", 32), "max_options")
        max_state_tokens = _positive_int(
            metadata.get("max_state_tokens", 65536), "max_state_tokens"
        )
        max_row_tokens = _positive_int(
            metadata.get("max_row_tokens", 73728), "max_row_tokens"
        )
        temperature = metadata.get("temperature")
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive")
        revisions = {}
        for key in ("base_model", "base_revision", "source_revision"):
            value = metadata.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty string")
            revisions[key] = value
        if metadata.get("option_isolation", False):
            raise ValueError("Kev row execution does not support option_isolation")
        return cls(
            hidden_size=hidden_size,
            head_dim=head_dim,
            temperature=float(temperature),
            special_token_ids=_special_ids(metadata.get("special_token_ids")),
            max_options=max_options,
            max_state_tokens=max_state_tokens,
            max_row_tokens=max_row_tokens,
            **revisions,
        )


def load_kev_config(checkpoint: str | Path) -> dict[str, Any] | None:
    """Read and validate a Kev manifest, or return None for an ordinary model.

    Invalid metadata raises ValueError (including JSONDecodeError for malformed
    JSON); filesystem errors propagate. This function needs no native runtime.
    """
    if not (Path(checkpoint) / "kev_config.json").exists():
        return None
    metadata = asdict(KevConfig.load(checkpoint))
    metadata["format_version"] = 1
    metadata["special_token_ids"] = list(metadata["special_token_ids"])
    return metadata


def encode_kev_rows(
    tokenizer: Any,
    state: str,
    questions: list[dict[str, Any]],
    *,
    max_state_tokens: int,
    max_row_tokens: int,
    max_options: int = 32,
) -> list[list[int]]:
    """Encode official Kev causal rows, rejecting overflow without truncation.

    Limits include delimiters. Each row starts at position zero, so its branch
    positions directly follow the state, as in ``kev.model.rows_of(encode(...))``.
    """
    _positive_int(max_state_tokens, "max_state_tokens")
    _positive_int(max_row_tokens, "max_row_tokens")
    _positive_int(max_options, "max_options")
    if not isinstance(state, str):
        raise ValueError("state must be a string")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    if hasattr(tokenizer, "get_real_tokenizer"):
        tokenizer = tokenizer.get_real_tokenizer()
    state_id, question_id, option_id, end_id, decide_id = kev_special_token_ids(
        tokenizer
    )

    def user_tokens(text: str) -> list[int]:
        escaped = _SPECIAL_RE.sub(r"<¦\1¦>", text)
        return list(tokenizer(escaped, add_special_tokens=False).input_ids)

    prefix = [state_id] + user_tokens(state)
    if len(prefix) > max_state_tokens:
        raise ValueError(
            f"state has {len(prefix)} tokens, exceeding limit {max_state_tokens}"
        )
    rows = []
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not isinstance(question.get("instr"), str):
            raise ValueError(f"questions[{index}].instr must be a string")
        options = question.get("options")
        if not isinstance(options, list) or not 1 <= len(options) <= max_options:
            raise ValueError(
                f"questions[{index}].options must contain 1..{max_options} options"
            )
        if any(not isinstance(option, str) or not option for option in options):
            raise ValueError(f"questions[{index}].options must be nonempty strings")
        row = prefix + [question_id] + user_tokens(question["instr"])
        for option in options:
            row.extend([option_id] + user_tokens(option) + [end_id])
        row.append(decide_id)
        if len(row) > max_row_tokens:
            raise ValueError(
                f"question {index} row has {len(row)} tokens, "
                f"exceeding limit {max_row_tokens}"
            )
        # Fail closed if a tokenizer unexpectedly emits a reserved token from text.
        validate_kev_row(
            row, (state_id, question_id, option_id, end_id, decide_id), max_options
        )
        rows.append(row)
    return rows


def validate_kev_row(
    token_ids: Sequence[int], special_token_ids: Sequence[int], max_options: int
) -> tuple[list[int], int]:
    """Validate one complete row and return option-end and decide positions."""
    state_id, question_id, option_id, end_id, decide_id = _special_ids(
        special_token_ids
    )
    _positive_int(max_options, "max_options")
    if not token_ids or token_ids[0] != state_id or token_ids[-1] != decide_id:
        raise ValueError("Kev row must start with state and end with decide")
    phase = "state"
    ends = []
    for position, token in enumerate(token_ids[1:], start=1):
        if token == state_id:
            raise ValueError("Kev row must contain exactly one state delimiter")
        if token == question_id:
            if phase != "state":
                raise ValueError("Kev row must contain exactly one question")
            phase = "instruction"
        elif token == option_id:
            if phase not in ("instruction", "between_options"):
                raise ValueError("unexpected or nested option delimiter")
            phase = "option"
        elif token == end_id:
            if phase != "option":
                raise ValueError("unmatched option end delimiter")
            ends.append(position)
            if len(ends) > max_options:
                raise ValueError(f"Kev row exceeds {max_options} options")
            phase = "between_options"
        elif token == decide_id:
            if position != len(token_ids) - 1 or phase != "between_options":
                raise ValueError("decide must follow at least one complete option")
        elif phase == "between_options":
            raise ValueError("unexpected text between options or before decide")
    return ends, len(token_ids) - 1


class KevHandler:
    """Deployment-local prefill pointer head; independent of native ops."""

    prefill_only = True
    supports_prefix_cache = True
    need_post_process = False

    def __init__(self, config: KevConfig):
        self.config = config
        self._weights: dict[str, torch.Tensor] = {}
        self._device: torch.device | None = None

    def custom_weight_info(self) -> list[Any]:
        # Loading through CustomAtomicWeight would cast the fp32 pointer head.
        return []

    def init_cpp_handler(self) -> None:
        pass

    def init(self, tensors: Mapping[str, torch.Tensor]) -> None:
        import torch

        shapes = {
            "q.weight": (self.config.head_dim, self.config.hidden_size),
            "q.bias": (self.config.head_dim,),
            "k.weight": (self.config.head_dim, self.config.hidden_size),
            "k.bias": (self.config.head_dim,),
        }
        if set(tensors) != set(shapes):
            raise ValueError(f"Kev head must contain exactly {list(shapes)}")
        for name, shape in shapes.items():
            tensor = tensors[name]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be fp32 with shape {shape}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} contains nonfinite weights")
        self._weights = {name: tensor.detach() for name, tensor in tensors.items()}
        self._device = None

    def extend_forward_args(self) -> list[str]:
        return ["hidden_states", "input_ids", "input_lengths", "prefix_lengths"]

    def cache_prefix_length(self, input_ids: torch.Tensor) -> int:
        """Return the state length; every question/option anchor must be computed."""
        self.select_token_position(input_ids)
        return input_ids.tolist().index(self.config.special_token_ids[1])

    def select_token_position(
        self,
        input_ids: torch.Tensor,
        text_tokens_mask: torch.Tensor | None = None,
    ) -> int:
        import torch

        if (
            input_ids.device.type != "cpu"
            or input_ids.ndim != 1
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Kev admission requires a one-dimensional CPU token row")
        if text_tokens_mask is not None:
            if (
                text_tokens_mask.ndim != 1
                or text_tokens_mask.numel() != input_ids.numel()
                or not text_tokens_mask.bool().all().item()
            ):
                raise ValueError("Kev supports text-only rows")
        _, decide = validate_kev_row(
            input_ids.tolist(), self.config.special_token_ids, self.config.max_options
        )
        return decide

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

        if not self._weights:
            raise RuntimeError("Kev head has not been initialized")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.config.hidden_size:
            raise ValueError("Kev hidden_states must have shape [tokens, hidden_size]")
        if (
            input_ids.ndim != 1
            or input_lengths.ndim != 1
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "Kev token IDs and lengths must be one-dimensional integers"
            )
        lengths = input_lengths.tolist()
        if not lengths or any(
            type(length) is not int or length <= 0 for length in lengths
        ):
            raise ValueError("Kev input_lengths must contain positive integers")
        if prefix_lengths is None:
            prefixes = [0] * len(lengths)
        else:
            if (
                not isinstance(prefix_lengths, torch.Tensor)
                or prefix_lengths.ndim != 1
                or prefix_lengths.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Kev prefix_lengths must be one-dimensional integers")
            prefixes = prefix_lengths.tolist()
            if len(prefixes) != len(lengths) or any(
                type(prefix) is not int or prefix < 0 for prefix in prefixes
            ):
                raise ValueError(
                    "Kev prefix_lengths must contain one nonnegative integer per row"
                )
        if sum(lengths) != input_ids.numel() or sum(lengths) != hidden_states.shape[0]:
            raise ValueError("Kev requires complete, unpadded prefill suffix rows")
        tokens = input_ids.tolist()
        option_positions, decide_positions, owners, option_counts = [], [], [], []
        offset = 0
        for owner, length in enumerate(lengths):
            row = tokens[offset : offset + length]
            synthetic_state = int(prefixes[owner] > 0)
            if synthetic_state:
                # Only ordinary state text may precede the still-present q.
                # The complete-row validator rejects a cached question, extra
                # delimiters in the state tail, and incomplete option anchors.
                row = [self.config.special_token_ids[0]] + row
            ends, decide = validate_kev_row(
                row,
                self.config.special_token_ids,
                self.config.max_options,
            )
            option_positions.extend(offset + end - synthetic_state for end in ends)
            decide_positions.append(offset + decide - synthetic_state)
            owners.extend([owner] * len(ends))
            option_counts.append(len(ends))
            offset += length
        device = hidden_states.device
        if self._device != device:
            self._weights = {
                name: tensor.to(device=device) for name, tensor in self._weights.items()
            }
            self._device = device
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            # The engine supplies normalized activations in the backbone dtype.
            # Keep that rounding, then perform both biased projections in fp32,
            # matching the official PointerHead's many() path.
            options = hidden_states[option_positions].float()
            decisions = hidden_states[decide_positions].float()
            keys = functional.linear(
                options, self._weights["k.weight"], self._weights["k.bias"]
            )
            queries = functional.linear(
                decisions, self._weights["q.weight"], self._weights["q.bias"]
            )
            scores = (keys * queries[owners]).sum(dim=-1)
            scores = scores * (1 / math.sqrt(self.config.head_dim))
            scores = scores / self.config.temperature
            probabilities = torch.zeros(
                (len(lengths), self.config.max_options),
                dtype=torch.float32,
                device=device,
            )
            offset = 0
            for row, count in enumerate(option_counts):
                probabilities[row, :count] = scores[offset : offset + count].softmax(-1)
                offset += count
        return probabilities


class KevModule:
    """The CustomModule interface, without importing model-loader/native code."""

    def __init__(self, config: Any, tokenizer: Any):
        import torch

        if config.compute_dtype != torch.bfloat16:
            raise ValueError(
                "Kev-4B requires BF16 backbone activations; set --act_type bf16"
            )
        quant_config = getattr(config, "quant_config", None)
        quant_algo = config.quant_algo
        requested_quantization = getattr(config, "quantization", None)
        has_quantization = (
            quant_algo.isQuant()
            or quant_config is not None
            or str(requested_quantization or "").lower() not in ("", "none")
        )
        # Use the resolved checkpoint configuration, not just the CLI string:
        # checkpoint quantization takes precedence in ModelConfig.
        fp8_per_block = (
            quant_config is not None
            and getattr(quant_config, "get_method", lambda: None)() == "FP8_PER_BLOCK"
            and quant_algo.isQuant()
            and getattr(quant_algo, "isFp8", lambda: False)()
            and getattr(quant_algo, "getWeightBits", lambda: None)() == 8
            and getattr(quant_algo, "getGroupSize", lambda: None)() == 128
        )
        layered_quantization = (
            getattr(config, "enable_nvfp4_dense_ffn", False)
            or getattr(config, "enable_w4a16_sm120_dense_ffn", False)
            or getattr(quant_config, "enable_nvfp4_dense_ffn", False)
        )
        if (has_quantization and not fp8_per_block) or layered_quantization:
            raise ValueError(
                "Kev-4B supports only BF16 or FP8_PER_BLOCK backbone weights "
                "(8-bit, group_size=128), without layered quantization"
            )
        self.config_ = config
        self.tokenizer_ = tokenizer
        self.kev_config = KevConfig.load(config.ckpt_path)
        if config.hidden_size != self.kev_config.hidden_size:
            raise ValueError("Kev head hidden_size does not match the backbone")
        if (
            tokenizer is not None
            and kev_special_token_ids(tokenizer) != self.kev_config.special_token_ids
        ):
            raise ValueError("Kev delimiter IDs do not match the checkpoint tokenizer")
        self.handler = KevHandler(self.kev_config)

    def get_handler(self) -> KevHandler:
        return self.handler

    def get_custom_weight_info(self) -> list[Any]:
        return []

    def init(self, weight: Any) -> None:
        from safetensors.torch import load_file

        tensors = load_file(str(Path(self.config_.ckpt_path) / "kev_head.safetensors"))
        self.handler.init(tensors)
