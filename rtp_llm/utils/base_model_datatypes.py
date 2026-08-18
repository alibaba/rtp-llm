from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, NamedTuple, Optional

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


# single batch prompt input
@dataclass
class RequestInfo:
    frontend_ip: str = ""
    dash_ip: str = ""
    trace_id: str = ""
    request_id: str = ""
    source_role: str = ""


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
    headers: Dict[str, str] = field(default_factory=dict, repr=False)
    frontend_metric_tags: Dict[str, str] = field(default_factory=dict, repr=False)
    frontend_metric_observer: Optional[Callable[[Any, int], None]] = field(
        default=None, repr=False, compare=False
    )
    request_info: RequestInfo = field(default_factory=RequestInfo, repr=False)

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
    prompt_logits: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


@dataclass
class GenerateOutputs:
    generate_outputs: List[GenerateOutput] = field(default_factory=list)
    # Internal transport marker. Unlike AuxInfo fields, this wrapper-level
    # value is consumed by BackendRPCServerVisitor and is never serialized in
    # an outward inference response.
    frontend_metric_only: bool = field(default=False, repr=False, compare=False)
    frontend_input_len: Optional[int] = field(default=None, repr=False, compare=False)
    frontend_output_len: Optional[int] = field(default=None, repr=False, compare=False)
    frontend_context_token_num: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_context_token_num_with_cache: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_context_execute_time_us: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_context_execute_time_with_cache_us: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_generate_token_num: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_generate_execute_time_us: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_speculative_verify_rounds: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_speculative_accepted_token_num: Optional[int] = field(
        default=None, repr=False, compare=False
    )
    frontend_speculative_proposed_draft_tokens: Optional[int] = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class FrontendMetricFrame:
    """Typed private projection shared by frontend metric producers."""

    aux_info: List[Optional[AuxInfo]]
    _frontend_metric_attempt: int
    _frontend_context_batch_size: int
    _frontend_output_batch_size: int
    _frontend_metric_unit_id: Optional[int]
    frontend_input_len: Optional[int]
    frontend_output_len: Optional[int]
    context_token_num: Optional[int]
    context_token_num_with_cache: Optional[int]
    context_execute_time_us: Optional[int]
    context_execute_time_with_cache_us: Optional[int]
    generate_token_num: Optional[int]
    generate_execute_time_us: Optional[int]
    speculative_verify_rounds: Optional[int]
    speculative_accepted_token_num: Optional[int]
    speculative_proposed_draft_tokens: Optional[int]

    @classmethod
    def from_output(
        cls,
        output: GenerateOutputs,
        *,
        attempt: int,
        context_batch_size: int = 1,
        unit_id: Optional[int] = None,
    ) -> "FrontendMetricFrame":
        return cls(
            aux_info=[item.aux_info for item in output.generate_outputs],
            _frontend_metric_attempt=attempt,
            _frontend_context_batch_size=context_batch_size,
            _frontend_output_batch_size=len(output.generate_outputs),
            _frontend_metric_unit_id=unit_id,
            frontend_input_len=output.frontend_input_len,
            frontend_output_len=output.frontend_output_len,
            context_token_num=output.frontend_context_token_num,
            context_token_num_with_cache=output.frontend_context_token_num_with_cache,
            context_execute_time_us=output.frontend_context_execute_time_us,
            context_execute_time_with_cache_us=output.frontend_context_execute_time_with_cache_us,
            generate_token_num=output.frontend_generate_token_num,
            generate_execute_time_us=output.frontend_generate_execute_time_us,
            speculative_verify_rounds=output.frontend_speculative_verify_rounds,
            speculative_accepted_token_num=output.frontend_speculative_accepted_token_num,
            speculative_proposed_draft_tokens=output.frontend_speculative_proposed_draft_tokens,
        )


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
