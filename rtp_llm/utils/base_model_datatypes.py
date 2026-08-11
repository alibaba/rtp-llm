import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional

import torch

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.ops import MultimodalInput


class EmbeddingOutput:
    text_embedding: torch.Tensor
    extra_input: Optional[torch.Tensor]

    def __init__(
        self, text_embedding: torch.Tensor, extra_input: Optional[List[torch.Tensor]]
    ):
        self.text_embedding = text_embedding
        if extra_input:
            try:
                self.extra_input = torch.concat(extra_input)
                self.extra_input = torch.Tensor(self.extra_input.shape[1:])
            except:
                raise Exception("Extra input must have same shape except dim 0")
        else:
            self.extra_input = None


class MMUrlType(IntEnum):
    DEFAULT = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3
    TENSOR = 4
    IGRAPH = 5


class VitParameters:
    """Vit parameters for multimodal models."""

    # config includes origin vit config in ckpt/config.json
    config: Dict[str, Any] = {}
    special_token_ids: Dict[str, Any] = {}
    special_tokens: Dict[str, Any] = {}
    vit_weights: Any = None
    preprocess_batch_size: int = 1
    eval_param_count = None
    eval_model_size = None


@dataclass(frozen=True)
class RequestDeadlineAnchor:
    monotonic_s: float
    unix_ms: int

    @classmethod
    def now(cls) -> "RequestDeadlineAnchor":
        return cls(
            monotonic_s=current_monotonic_time_s(),
            unix_ms=current_unix_time_ms(),
        )


# single batch prompt input
@dataclass
class GenerateInput:
    request_id: int
    token_ids: torch.Tensor
    mm_inputs: List[MultimodalInput]
    generate_config: GenerateConfig
    tokenizer: Any = None  # TODO: remove this
    prefix_length: int = 0
    token_type_ids: List[int] = field(default_factory=list)
    batch_group_size: int = 1
    batch_group_id: int = (
        -1
    )  # Batch group ID for force batch grouping, -1 means not set
    request_deadline_monotonic_s: Optional[float] = field(
        default=None, repr=False, compare=False
    )
    request_deadline_unix_ms: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    ttft_deadline_monotonic_s: Optional[float] = field(
        default=None, repr=False, compare=False
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def input_length(self):
        return self.token_ids.shape[-1]

    @property
    def prompt_length(self):
        return self.token_ids.shape[-1] - self.prefix_length

    def update_prefix(self, prefix_tokens: torch.Tensor):
        self.token_ids = torch.concat([prefix_tokens, self.token_ids], dim=0)
        self.prefix_length = prefix_tokens.nelement()


def _positive_timeout_ms(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        timeout_ms = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return timeout_ms if timeout_ms > 0 else None


def current_monotonic_time_s() -> float:
    return time.monotonic()


def current_unix_time_ms() -> int:
    return time.time_ns() // 1_000_000


def initialize_request_deadlines(
    generate_input: GenerateInput,
    monotonic_now_s: Optional[float] = None,
    unix_now_ms: Optional[int] = None,
) -> None:
    """Initialize one immutable total/TTFT budget at backend ingress."""
    if monotonic_now_s is None:
        monotonic_now_s = current_monotonic_time_s()

    config = generate_input.generate_config
    total_timeout_ms = _positive_timeout_ms(getattr(config, "timeout_ms", None))
    request_deadline = getattr(
        generate_input, "request_deadline_monotonic_s", None
    )
    request_deadline_unix_ms = getattr(
        generate_input, "request_deadline_unix_ms", None
    )

    if total_timeout_ms is not None and request_deadline is None:
        request_deadline = monotonic_now_s + total_timeout_ms / 1000.0
        setattr(generate_input, "request_deadline_monotonic_s", request_deadline)

    if total_timeout_ms is not None and request_deadline_unix_ms is None:
        if unix_now_ms is None:
            unix_now_ms = current_unix_time_ms()
        if request_deadline is None:
            remaining_ms = total_timeout_ms
        else:
            remaining_ms = remaining_deadline_ms(
                request_deadline, monotonic_now_s
            )
        setattr(
            generate_input,
            "request_deadline_unix_ms",
            unix_now_ms + (remaining_ms or 0),
        )

    if getattr(generate_input, "ttft_deadline_monotonic_s", None) is not None:
        return

    ttft_timeout_ms = _positive_timeout_ms(
        getattr(config, "ttft_timeout_ms", None)
    )
    ttft_deadline = (
        monotonic_now_s + ttft_timeout_ms / 1000.0
        if ttft_timeout_ms is not None
        else request_deadline
    )
    if ttft_deadline is not None and request_deadline is not None:
        ttft_deadline = min(ttft_deadline, request_deadline)
    if ttft_deadline is not None:
        setattr(generate_input, "ttft_deadline_monotonic_s", ttft_deadline)


def remaining_deadline_ms(
    deadline_monotonic_s: Optional[float],
    monotonic_now_s: Optional[float] = None,
) -> Optional[int]:
    if deadline_monotonic_s is None:
        return None
    if monotonic_now_s is None:
        monotonic_now_s = current_monotonic_time_s()
    remaining_ms = (deadline_monotonic_s - monotonic_now_s) * 1000.0
    if remaining_ms <= 0:
        return 0
    return max(1, math.ceil(remaining_ms - 1e-9))


@dataclass
class AuxInfo:
    cost_time: float = 0
    iter_count: int = 0
    prefix_len: int = 0
    input_len: int = 0
    output_len: int = 0
    step_output_len: int = 0
    first_token_cost_time: float = 0
    wait_time: float = 0
    pd_sep: bool = False
    cum_log_probs: List[float] = field(default_factory=list)
    beam_responses: List[str] = field(default_factory=list)
    softmax_probs: List[float] = field(default_factory=list)

    reuse_len: int = 0
    local_reuse_len: int = 0
    remote_reuse_len: int = 0
    memory_reuse_len: int = 0

    prefill_total_reuse_len: int = 0
    prefill_local_reuse_len: int = 0
    prefill_remote_reuse_len: int = 0
    prefill_memory_reuse_len: int = 0

    decode_total_reuse_len: int = 0
    decode_local_reuse_len: int = 0
    decode_remote_reuse_len: int = 0
    decode_memory_reuse_len: int = 0

    multimodal_lengths: Dict[int, int] = field(default_factory=dict)

    role_addrs: List[RoleAddr] = field(default_factory=list)
    aux_string: str = ""


@dataclass
class GenerateOutput:
    hidden_states: Optional[torch.Tensor] = None
    all_hidden_states: Optional[torch.Tensor] = None
    output_ids: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    finished: bool = False
    aux_info: Optional[AuxInfo] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    all_probs: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True


@dataclass
class GenerateOutputs:
    generate_outputs: List[GenerateOutput] = field(default_factory=list)


@dataclass
class GenerateResponse:
    generate_outputs: GenerateOutputs
    generate_texts: List[str]


class GenerateContext(NamedTuple):
    inputs: Any
    input_embeds: Any
    attention_mask: Any
    pad_lengths: Any
    input_lengths: Any
    memory_length: Any
    sampler: Any
    batch_size: Any
    beam_width: Any
    max_input_length: Any
    finished: Any
    sequence_lengths: Any
    gen_length: Any
    cum_log_probs: Any
    extra_args: Any
    all_start_time: Any
    cache_indirection: Any
    output_token_ids: Any
