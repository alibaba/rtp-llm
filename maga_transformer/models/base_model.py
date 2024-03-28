import os
import torch
from dataclasses import dataclass
from pydantic import BaseModel as PyBaseModel
from typing import Any, Dict, List, Optional, Union, NamedTuple

from transformers import PreTrainedTokenizerBase

from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.utils.sample_utils import HuggingfaceSampler, FtSampler, BaseSampler, DynamicDecodeOp, BeamSearchSampler
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, ModelType

FT_DEFAULT_MAX_NEW_TOKENS = 2048

# single batch prompt input
class GenerateInput(PyBaseModel):
    token_ids: torch.Tensor
    images: Union[torch.Tensor, List[Any]] = []
    generate_config: GenerateConfig
    tokenizer: Any = None # TODO: remove this
    lora_id: int = -1
    prefix_length: int = 0

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

class AuxInfo(PyBaseModel):
    cost_time: float = 0
    iter_count: int = 0
    prefix_len: int = 0
    input_len: int = 0
    reuse_len: int = 0
    output_len: int = 0
    cum_log_probs: List[float] = []
    beam_responses: List[str] = []

class GenerateOutput(PyBaseModel):
    hidden_states: Optional[torch.Tensor] = None
    output_ids: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    finished: bool = False
    aux_info: AuxInfo = AuxInfo()
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

class GenerateResponse(PyBaseModel):
    generate_output: GenerateOutput
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

@dataclass
class ModelConfigBase:
    model_type: str = ""
    ckpt_path: str = ""
    tokenizer_path: str = ""
    weight_type: WEIGHT_TYPE = WEIGHT_TYPE.FP16
    act_type: WEIGHT_TYPE = WEIGHT_TYPE.FP16
    max_seq_len: int = 0
    seq_size_per_block: int = 8
    gen_num_per_circle: int = 1
    ptuning_path: Optional[str] = None
    lora_infos: Optional[Dict[str, str]] = None
    ref_model: Optional[torch.nn.Module] = None

class ModelConfig(ModelConfigBase):
    @property
    def int8_mode(self):
        return True if self.weight_type == WEIGHT_TYPE.INT8 else False

    def add_ref_model(self, ref_model: Optional[torch.nn.Module]):
        self.ref_model = ref_model

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
        return self

class BaseModel(object):

    config: GptInitModelParameters
    vocab_size_padded: int

    @classmethod
    def create_config(cls, model_config: ModelConfig) -> GptInitModelParameters:
        config: GptInitModelParameters = cls._create_config(model_config.ckpt_path)
        if config.hidden_size == 0:
            config.hidden_size = config.size_per_head * config.head_num
        int8_mode = model_config.int8_mode
        if config.quant_algo.int4_mode:
            int8_mode = False
        config.update_common(
            ckpt_path=model_config.ckpt_path,
            tokenizer_path=model_config.tokenizer_path,
            int8_mode=int8_mode,
            data_type=model_config.act_type,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.seq_size_per_block,
            tp_size=g_parallel_info.tp_size,
            gen_num_per_circle=model_config.gen_num_per_circle,
            lora_infos=model_config.lora_infos,
            ptuning_path=model_config.ptuning_path,
            ref_model=model_config.ref_model
        )
        return config

    @staticmethod
    def _create_config(ckpt_path: str) -> GptInitModelParameters:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: Any) -> 'BaseModel':
        raise NotImplementedError()

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters) -> PreTrainedTokenizerBase:
        raise NotImplementedError()

    @classmethod
    def is_multimodal(cls) -> bool:
        return False

    def __init__(self) -> None:
        self.weight = None
        self.word_embedding: Optional[torch.nn.Module] = None
        self.prefix_encoder: Optional[torch.nn.Module] = None
        self.position_encoding: Optional[torch.nn.Module] = None
        self.token_type_embeddings: Optional[torch.nn.Module] = None
        self.pre_decoder_layernorm: Optional[torch.nn.Module] = None
        self.post_decoder_layernorm: Optional[torch.nn.Module] = None

        self.lm_head: Optional[torch.nn.Module] = None
        self.config: GptInitModelParameters = None
        self.context_decoder: Optional[FTOPBase] = None
        self.decoder: Optional[FTOPBase] = None
        self.dynamic_decoder = None
        self.use_fp32_to_compute_logit = False
        self.linear_bias_slopes: Optional[torch.Tensor] = None

        self.medusa_head: Optional[torch.nn.ModuleList] = None

        self.prefix_tokens: Optional[torch.Tensor] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.max_input_buffer_len: int = 0

        self.default_generate_config: GenerateConfig = GenerateConfig()

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype
    
    @property
    def model_type(self) -> ModelType:
        return self.config.model_type

    @property
    def device(self) -> Union[str, torch.device]:
        assert self.weight is not None
        return 'cuda:0'

    def dup_dim0_for_beam_search(self, t: torch.Tensor, beam_width: int) -> torch.Tensor:
        shape = list(t.shape)
        return t.unsqueeze(1).repeat([1, beam_width] + [1] * len(shape[1:])).reshape([-1] + shape[1:]).contiguous()

    def async_input_word_embedding(self, inputs: torch.Tensor, images: List[List[Any]]):
        return self.word_embedding(inputs)

    def create_context_position_ids(self, input_lengths: List[int]):
        return torch.concat([torch.arange(input_length) for input_length in input_lengths], dim=0)

    def create_context_decoder_mask(self, input_lengths: List[int]):
        batch_size = len(input_lengths)
        max_input_length = max(input_lengths)
        attention_mask = torch.ones(
            (max_input_length, max_input_length), dtype=torch.bool, device=self.device)
        if self.config.is_causal:
            attention_mask = attention_mask.tril()
        attention_mask = attention_mask.unsqueeze_(0).tile(batch_size, 1, 1).to(self.dtype)
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
            if not self.config.is_causal:
                attention_mask[b, :, input_length: ]= 0
        return attention_mask

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        return config.eval_model_size()

    def _create_hf_sampler(self, generate_config: GenerateConfig) -> HuggingfaceSampler:
        return HuggingfaceSampler(generate_config)

    def _create_ft_sampler(self, generate_config: GenerateConfig) -> FtSampler:
        dynamic_decoder = DynamicDecodeOp(self.config.vocab_size, self.vocab_size_padded)
        return FtSampler(config=generate_config, dynamic_decoder=dynamic_decoder)

    def _create_beam_search_sampler(self, generate_config: GenerateConfig) -> BeamSearchSampler:
        dynamic_decoder = DynamicDecodeOp(self.config.vocab_size, self.vocab_size_padded)
        return BeamSearchSampler(generate_config, dynamic_decoder)

    def create_sampler(self, generate_config: GenerateConfig) -> BaseSampler:
        using_hf_sampling = generate_config.using_hf_sampling or self.config.using_hf_sampling
        if generate_config.num_beams > 1:
            return self._create_beam_search_sampler(generate_config)
        elif using_hf_sampling:
            return self._create_hf_sampler(generate_config)
        else:
            return self._create_ft_sampler(generate_config)
