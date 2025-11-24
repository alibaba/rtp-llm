from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

import torch

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.weight_type import WEIGHT_TYPE


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


class ModelConfig:
    KV_CACHE_DTYPE = "KV_CACHE_DTYPE"
    QUANTIZATION_KEY = "QUANTIZATION"
    ACT_TYPE = "ACT_TYPE"
    WEIGHT_TYPE = "WEIGHT_TYPE"  # Compatible for old config
    INT8_MODE = "INT8_MODE"  # Compatible for old config

    SP_KV_CACHE_DTYPE = "SP_KV_CACHE_DTYPE"
    SP_QUANTIZATION_KEY = "SP_QUANTIZATION"
    SP_ACT_TYPE = "SP_ACT_TYPE"
    SP_WEIGHT_TYPE = "SP_WEIGHT_TYPE"  # Compatible for old config

    def __init__(
        self,
        model_type: str = "",
        ckpt_path: str = "",
        tokenizer_path: str = "",
        act_type: str = None,
        kv_cache_type: str = None,
        max_seq_len: int = 0,
        seq_size_per_block: int = 8,
        gen_num_per_circle: int = 1,
        ptuning_path: Optional[str] = None,
        lora_infos: Optional[Dict[str, str]] = None,
        ref_module: Optional[torch.nn.Module] = None,
        ref_dict: Dict[str, torch.Tensor] = {},
        sp_type: str = "",
        quantization: str = "",
    ):
        self.model_type: str = model_type
        self.ckpt_path: str = ckpt_path
        self.tokenizer_path: str = tokenizer_path
        self.act_type: str = act_type
        self.kv_cache_type: str = kv_cache_type
        self.quantization = quantization
        self.max_seq_len: int = max_seq_len
        self.seq_size_per_block: int = seq_size_per_block
        self.gen_num_per_circle: int = gen_num_per_circle
        self.ptuning_path: Optional[str] = ptuning_path
        self.lora_infos: Optional[Dict[str, str]] = lora_infos
        self.ref_module: Optional[torch.nn.Module] = ref_module
        self.ref_dict: Dict[str, torch.Tensor] = ref_dict
        self.sp_type: str = sp_type

    def add_ref_module(self, ref_module: Optional[torch.nn.Module]):
        self.ref_module = ref_module

    def add_ref_dict(self, ref_dict: Dict[str, torch.Tensor]):
        self.ref_dict = ref_dict

    def _replace(self, **kwargs: Any):
        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
        return self

    @staticmethod
    def get_quantization_from_params(env_params: Dict[str, str]):
        if (not env_params.get(ModelConfig.QUANTIZATION_KEY)) and (
            env_params.get(ModelConfig.WEIGHT_TYPE, "").upper() == "INT8"
            or int(env_params.get(ModelConfig.INT8_MODE, "0")) == 1
        ):
            quantization = "INT8"
        else:
            quantization = env_params.get(ModelConfig.QUANTIZATION_KEY)
        return quantization

    @staticmethod
    def get_sp_quantization_from_params(env_params: Dict[str, str]):
        if not env_params.get(ModelConfig.SP_QUANTIZATION_KEY) and (
            env_params.get(ModelConfig.SP_WEIGHT_TYPE, "").upper() == "INT8"
            or int(env_params.get(ModelConfig.INT8_MODE, "0")) == 1
        ):
            quantization = "INT8"
        else:
            quantization = env_params.get(ModelConfig.SP_QUANTIZATION_KEY)
        return quantization
