from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional

import torch

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr


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
    CUSTOM = 6


@dataclass
class MMPreprocessConfig:
    width: int = -1
    height: int = -1
    min_pixels: int = -1
    max_pixels: int = -1
    fps: int = -1
    min_frames: int = -1
    max_frames: int = -1
    crop_positions: List[float] = field(default_factory=list)
    mm_timeout_ms: int = 30000

    def to_string(self):
        return f"{self.width}_{self.height}_{self.min_pixels}_{self.max_pixels}_{self.fps}_{self.min_frames}_{self.max_frames}_[{'_'.join([str(x) for x in self.crop_positions])}]_{self.mm_timeout_ms}"


class MultimodalInput:
    url: str
    mm_type: MMUrlType
    tensor: torch.Tensor
    config: MMPreprocessConfig

    def __init__(
        self,
        url: str,
        mm_type: MMUrlType = MMUrlType.DEFAULT,
        tensor: Optional[torch.Tensor] = None,
        config: MMPreprocessConfig = MMPreprocessConfig(),
    ):
        self.url = url
        self.mm_type = mm_type
        self.config = config
        self.tensor = tensor

    def to_string(self):
        return f"{self.url}_{self.mm_type}_{self.config.to_string()}"


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

    prefill_total_reuse_len: int = 0
    prefill_local_reuse_len: int = 0
    prefill_remote_reuse_len: int = 0

    decode_total_reuse_len: int = 0
    decode_local_reuse_len: int = 0
    decode_remote_reuse_len: int = 0

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
