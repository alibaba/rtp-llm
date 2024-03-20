import os
import time
import torch
import pynvml
import logging
import json
import pydantic
from dataclasses import dataclass
from pydantic import BaseModel as PyBaseModel
from typing import Any, Dict, List, Optional, Union, Tuple, NamedTuple, AsyncGenerator
from PIL import Image

import torch.distributed as dist
from transformers import PreTrainedTokenizerBase

from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.utils.util import WEIGHT_TYPE
from maga_transformer.utils.sample_utils import HuggingfaceSampler, FtSampler, BaseSampler, \
    SamplerSetupParams, SamplingParams, DynamicDecodeOp, BeamSearchSampler
from maga_transformer.utils.stop_utils import create_stop_criteria_list
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.generate_config import GenerateConfig, RequestFormat
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, ModelType
from maga_transformer.utils.model_weight import LoRAMap
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase

FT_DEFAULT_MAX_NEW_TOKENS = 2048

def debug_print_hidden(name: str, t: torch.Tensor):
    if not (os.environ.get('FT_DEBUG_PRINT_LEVEL') == 'DEBUG'):
        return
    if not (os.environ.get('WORLD_RANK') == '0'):
        return
    torch.set_printoptions(profile="full")
    print(name, t.shape)
    if len(t.shape) == 2 or len(t.shape) == 1:
        if t.dtype == torch.int32:
            print(t)
        else:
            print(t.reshape([-1])[:20])
        return
    for b_idx in range(t.shape[0]):
        for s_idx in range(t.shape[1]):
            print(b_idx, s_idx, t[b_idx,s_idx,:8])

# single batch prompt input
class GenerateInput(PyBaseModel):
    token_ids: torch.Tensor
    images: Union[torch.Tensor, List[torch.Tensor]] = []
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
    async_mode: bool = True
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
    
    @property
    def int4_mode(self):
        return True if self.weight_type == WEIGHT_TYPE.INT4 else False

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
        config.update_common(
            ckpt_path=model_config.ckpt_path,
            tokenizer_path=model_config.tokenizer_path,
            int8_mode=model_config.int8_mode,
            int4_mode=model_config.int4_mode,
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

    @torch.no_grad()
    def prepare_context(self, input: GenerateInput)  -> GenerateContext:
        all_start_time = time.time()
        assert self.config is not None, "Config should not be None"
        assert self.decoder is not None, "Decoder should not be None"
        assert self.context_decoder is not None, "Context Decoder should not be None"
        assert self.weight is not None, 'Please call load() first to initialize weights.'

        input.token_ids = input.token_ids.type(torch.int32).to(self.device)
        input_embeds = self._do_pipeline_multimodal_embed(input.token_ids, input.images)

        inputs_np = input.token_ids.cpu().numpy()
        batch_size = len(inputs_np)

        eos_token_id = input.generate_config.eos_token_id if input.generate_config.eos_token_id is not None \
            else self.config.special_tokens.eos_token_id
        assert eos_token_id is not None, 'eos_token-id must be specified in generation.'
        input.generate_config.eos_token_id = eos_token_id
        sampler = self.create_sampler(input.generate_config)

        max_new_tokens = FT_DEFAULT_MAX_NEW_TOKENS
        if input.generate_config.max_new_tokens != None:
            max_new_tokens = input.generate_config.max_new_tokens

        max_input_length = input.token_ids.shape[-1]
        if self.max_input_buffer_len < max_input_length * batch_size:
            if self.max_input_buffer_len != 0:
                torch.cuda.empty_cache()
            self.max_input_buffer_len = max_input_length * batch_size

        # Setup decoder_op prior to calling the forward function.
        sampler.setup(SamplerSetupParams(batch_size, eos_token_id, max_input_length, input.token_ids))
        beam_width = input.generate_config.num_beams

        pre_seq_len = 0
        if getattr(self.config, "pre_seq_len", None) is not None:
            pre_seq_len = self.config.pre_seq_len
        gen_length = min(self.config.max_seq_len - max_input_length, max_new_tokens)
        if gen_length < 0:
            raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR, f"model max tokens is {self.config.max_seq_len}, request length is {max_input_length}， max_new_tokens is {max_new_tokens}")
        max_seq_length = max_input_length + gen_length + pre_seq_len
        memory_length = max_seq_length
        device = self.device

        # Prepare input and output arguments.
        pad_lengths = max_input_length - input.input_length
        # Since tril() doesn't support bf16 dtype, we create of bool type and then cast it to dtype.
        attention_mask = self.create_context_decoder_mask([max_input_length])
        # concat attention_mask
        if self.config.pre_seq_len is not None and self.config.pre_seq_len > 0:
            prefix_mask = torch.ones((batch_size, max_input_length, self.config.pre_seq_len), dtype=torch.bool, device=self.device)
            attention_mask = torch.concat([prefix_mask, attention_mask], dim=-1)

        finished = torch.tensor([False])
        # sequence_lengths = (max_input_length - 1) * torch.ones_like(input_lengths)
        sequence_lengths = torch.tensor([input.token_ids.shape[-1] - 1])

        # Contiguous buffer for each decode_op step, it will be transposed tensor for the final output.
        output_token_ids: torch.Tensor = torch.ones(
            (max_seq_length, batch_size), dtype=torch.int32, device=device) * eos_token_id
        output_token_ids[:max_input_length, ...] = input.token_ids.T

        position_ids = torch.arange(0, max_input_length, dtype=torch.int, device=device) \
                            .unsqueeze(0).view(-1, max_input_length)

        input_embeds = self._do_pipeline_embed_post_process(batch_size, max_input_length, input_embeds, position_ids)

        prefix_prompt: Optional[torch.Tensor] = None
        prefix_lengths: Optional[torch.Tensor] = None

        # 这里是存储各个模型特有的参数，这样不需要为context_decoder和decoder的接口适配所有参数
        extra_args = {}

        if self.prefix_encoder is not None:
            # 这里ft对dim0==1会做broadcast， 所以batchsize写1
            prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(1, -1).to(device)
            prefix_prompt = self.prefix_encoder(prefix_tokens)
            prefix_lengths = torch.IntTensor([self.prefix_tokens.size(0)] * batch_size).to(torch.int32).to(device)
            # cpu
            max_prefix_lengths = torch.IntTensor([self.prefix_tokens.size(0)]).to(torch.int32)

            extra_args['prefix_prompt'] = prefix_prompt
            extra_args['prefix_lengths'] = prefix_lengths
            extra_args['max_prefix_lengths'] = max_prefix_lengths

        if self.linear_bias_slopes is not None:
            extra_args['linear_bias_slopes'] = self.linear_bias_slopes

        # set lora
         # lora
        lora_names = []
        lora_ids = []
        if input.generate_config.adapter_name is not None:
            if isinstance(input.generate_config.adapter_name, str):
                lora_names = [input.generate_config.adapter_name]
            else:
                lora_names = input.generate_config.adapter_name
        if len(lora_names) != 0:
            for lora_name in lora_names:
                lora_ids.append(self.weight.lora_resource.get_id(lora_name))

        lora_ids += [-1] * (batch_size - len(lora_names))
        extra_args['lora_ids'] = torch.IntTensor(lora_ids)

        cache_indirection = None
        cum_log_probs = None

        if beam_width > 1:
            input_embeds = self.dup_dim0_for_beam_search(input_embeds, beam_width)
            attention_mask = self.dup_dim0_for_beam_search(attention_mask, beam_width)
            input.input_lengths = self.dup_dim0_for_beam_search(input.input_length, beam_width)
            pad_lengths = self.dup_dim0_for_beam_search(pad_lengths, beam_width)
            sequence_lengths = self.dup_dim0_for_beam_search(sequence_lengths, beam_width)
            finished = self.dup_dim0_for_beam_search(finished, beam_width)
            output_token_ids = output_token_ids.unsqueeze(2).repeat(1, 1, beam_width) \
                .reshape([max_seq_length, -1])
            # src/tgt cache indirections.
            cache_indirection = torch.zeros(
                (2, batch_size, beam_width, memory_length), dtype=torch.int32, device=device)
            cum_log_probs = torch.zeros(batch_size * beam_width, device=device)

        return GenerateContext(input.token_ids, input_embeds, attention_mask, pad_lengths,
                               torch.tensor([input.input_length], dtype=torch.int32).cuda(),
                               memory_length, sampler,
                               batch_size, beam_width, max_input_length, finished,
                               sequence_lengths, gen_length, cum_log_probs,
                               extra_args, all_start_time, cache_indirection, output_token_ids)

    @torch.no_grad()
    def generate_loss(self,
                 generate_context: GenerateContext,
                 context_decoder_output,
                 calculate_loss: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        context_decoder_output_logits = self._get_logits(context_decoder_output)

        shift_labels = generate_context.inputs[..., 1:].type(torch.int64).contiguous()
        shift_logits = context_decoder_output_logits[..., :-1, :].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss = loss.view(generate_context.batch_size, generate_context.max_input_length - 1)

        for idx in range(generate_context.batch_size):
            loss[idx, generate_context.sequence_lengths[idx]:] = 0

        if calculate_loss == 1:
            loss_mean = loss.sum(dim=1) / generate_context.sequence_lengths
            loss_mean = loss_mean.exp()
            return loss_mean
        elif calculate_loss == 2:
            return [loss[idx, :generate_context.sequence_lengths[idx]] for idx in range(generate_context.batch_size)]
        else:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "calculate_loss in generate_config can only be 0, 1 or 2")

    def acquire_resource(self, generate_config: GenerateConfig) -> None:
        if generate_config.adapter_name != None:
            for name in generate_config.adapter_name:
                self.weight.lora_resource.read_acquire(name)

    def release_resource(self, generate_config: GenerateConfig) -> None:
        if generate_config.adapter_name != None:
            for name in generate_config.adapter_name:
                self.weight.lora_resource.read_release(name)

    def enqueue(self, input: GenerateInput) -> AsyncGenerator[GenerateOutput, None]:
        return self.generate_stream(input)

    @torch.no_grad()
    async def generate_stream(self, input: GenerateInput) -> AsyncGenerator[GenerateOutput, None]:
        self.acquire_resource(input.generate_config)
        try:
            for element in self.generate_stream_wrapper(input):
                yield element
        finally:
            self.release_resource(input.generate_config)

    @torch.no_grad()
    def generate_stream_wrapper(self, input: GenerateInput):
        """

        # Args.
            inputs: IntTensor, (batch_size, max_input_length),
                input hidden state to decoder.
            input_lengths: IntTensor, (batch_size),
                the lengths of input context sequences.
        # Returns
            Iterator[GenerateOutput]
        """
        generate_context = self.prepare_context(input)

        context_decoder_output, k_cache, v_cache, hidden_states = self.context_decoder.forward(
            input_embeds=generate_context.input_embeds,
            attention_mask=generate_context.attention_mask,
            input_lengths=generate_context.input_lengths,
            memory_length=generate_context.memory_length,
            **generate_context.extra_args)

        loss = None
        logits = None
        if input.generate_config.calculate_loss:
            loss = self.generate_loss(generate_context, context_decoder_output, input.generate_config.calculate_loss)
            batch_size = generate_context.batch_size
            yield GenerateOutput(hidden_states,
                                 generate_context.output_token_ids.view(-1, generate_context.batch_size,
                                        generate_context.beam_width)[:1, ...].permute(1, 2, 0),
                                 generate_context.finished, [{}] * batch_size, loss)


        src_cache_indirection = None
        tgt_cache_indirection = None

        for step in range(generate_context.max_input_length, generate_context.max_input_length + generate_context.gen_length):
            if generate_context.cache_indirection != None:
                src_indir_idx = (step - generate_context.max_input_length) % 2
                tgt_indir_idx = 1 - src_indir_idx
                src_cache_indirection = generate_context.cache_indirection[src_indir_idx, ...]
                tgt_cache_indirection = generate_context.cache_indirection[tgt_indir_idx, ...]

            if step != generate_context.max_input_length:
                input_embeds = self._do_pipline_first_token_emb(
                    generate_context.batch_size,
                    1,
                    generate_context.output_token_ids[step - 1, :],
                    (step - 1) * torch.ones_like(generate_context.pad_lengths))
                forward_start_time = time.time()
                hidden_states = self.decoder.forward(
                    max_input_length=generate_context.max_input_length,
                    step=step,
                    ite=0,
                    input_embeds=input_embeds,
                    sequence_lengths=generate_context.sequence_lengths,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    finished=finished,
                    input_lengths=generate_context.input_lengths,
                    masked_tokens=None,
                    **generate_context.extra_args)

            # TODO(xinfei.sxf) create_stop_criteria_list 不要多次调用啊。
            finished, sequence_lengths = self._do_pipline_last_sample(
                generate_context.sampler,
                step,
                generate_context.batch_size,
                generate_context.beam_width,
                generate_context.max_input_length,
                hidden_states,
                generate_context.input_lengths,
                generate_context.finished,
                generate_context.sequence_lengths,
                generate_context.output_token_ids,
                k_cache,
                v_cache,
                generate_context.cum_log_probs,
                create_stop_criteria_list(input.generate_config.stop_words_list, input.generate_config.stop_words_str, self.tokenizer)
            )
            aux_info = None
            if generate_context.cum_log_probs != None:
                aux_info = [
                    {'cum_log_probs': prob} for prob in generate_context.cum_log_probs.view(
                                generate_context.batch_size, generate_context.beam_width).tolist()
                ]

            if input.generate_config.return_logits:
                logits = self._get_logits(hidden_states)

            yield GenerateOutput(hidden_states.view(generate_context.batch_size, generate_context.beam_width, -1),
                                 generate_context.output_token_ids.view(-1, generate_context.batch_size,
                                        generate_context.beam_width)[:step + 1, ...].permute(1, 2, 0),
                                 finished,
                                 aux_info, loss, logits)



    def _do_pipline_first_token_emb(self, batch_size: int, max_input_length: int,
                                inputs: torch.Tensor, position_ids: Optional[torch.Tensor]):
        if g_parallel_info.is_pp_first:
            input_embeds = self.word_embedding(inputs)
            debug_print_hidden('ids', inputs)
            debug_print_hidden('embds', input_embeds)
            if self.position_encoding is not None:
                input_embeds += self.position_encoding(position_ids)
            if self.pre_decoder_layernorm is not None:
                input_embeds = self.pre_decoder_layernorm(input_embeds)
            debug_print_hidden('attn_in', input_embeds)
        else:
            # Dummy input_embeds
            input_embeds = torch.empty(
                size=(batch_size, max_input_length, self.context_decoder.hidden_size),
                dtype=self.dtype,
                device=self.device)
        return input_embeds

    def async_input_word_embedding(self, inputs: torch.Tensor, images: List[List[Any]]):
        return self.word_embedding(inputs)

    def input_word_embedding(self, inputs: torch.Tensor, images: List[List[Any]]):
        return self.word_embedding(inputs)

    def _do_pipeline_multimodal_embed(self, inputs: torch.Tensor, images: List[List[Any]] = [[]]):
        if g_parallel_info.is_pp_first:
            input_embeds = self.input_word_embedding(inputs, images)
            debug_print_hidden('ids', inputs)
            debug_print_hidden('embds', input_embeds)
            return input_embeds
        else:
            return None

    def _do_pipeline_embed_post_process(self, batch_size: int, max_input_length: int,
                                input_embeds: torch.Tensor, position_ids: Optional[torch.Tensor]):
        if g_parallel_info.is_pp_first:
            if self.position_encoding is not None:
                input_embeds += self.position_encoding(position_ids)
            if self.token_type_embeddings is not None:
                input_embeds += self.token_type_embeddings(torch.zeros_like(position_ids))
            if self.pre_decoder_layernorm is not None:
                input_embeds = self.pre_decoder_layernorm(input_embeds)
            debug_print_hidden('attn_in', input_embeds)
        else:
            # Dummy input_embeds
            input_embeds = torch.empty(
                size=(batch_size, max_input_length, self.context_decoder.hidden_size),
                dtype=self.dtype,
                device=self.device)
        return input_embeds

    def do_broadcast(self, sampler: BaseSampler, output_token_ids: torch.Tensor, step: int, finished: torch.Tensor, sequence_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        def broadcast_from_last_index(tensor: torch.Tensor) -> torch.Tensor:
            dist.broadcast(tensor, g_parallel_info.world_size - 1)
            return tensor

        if isinstance(sampler, FtSampler):
                # broadcast and synchronize error
                sampler.dynamic_decoder.broadcast_from_last_pipeline([output_token_ids, finished, sequence_lengths])
        else:
            # TODO@miji add work to check timeout
            if g_parallel_info.world_size > 1:
                output_token_ids[step, ...] = broadcast_from_last_index(output_token_ids[step, ...])
                finished = broadcast_from_last_index(finished)
                sequence_lengths = broadcast_from_last_index(sequence_lengths)
        return finished, sequence_lengths

    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        debug_print_hidden('decoder out', hidden_states)
        if self.post_decoder_layernorm is not None:
            hidden_states = self.post_decoder_layernorm(hidden_states)
            debug_print_hidden('after norm', hidden_states)
        # We use logits of fp32 type to avoid overflow issue.
        if self.use_fp32_to_compute_logit:
            # The FT GPT op internally uses FP32 compute type for matrix multiplication.
            # This will produce the same result with the end-to-end FT's GPT op.
            logits = torch.nn.functional.linear(hidden_states.float(), self.lm_head.weight)
        else:
            logits = self.lm_head(hidden_states).float()

        debug_print_hidden('logits', logits)

        return logits

    def _do_pipline_last_sample(self,
                                sampler: BaseSampler,
                                step: int,
                                batch_size: int,
                                beam_width: int,
                                max_input_length: int,
                                hidden_states: torch.Tensor,
                                input_lengths: torch.Tensor,
                                finished: torch.Tensor,
                                sequence_lengths: torch.Tensor,
                                output_token_ids: torch.Tensor,
                                key_cache: torch.Tensor,
                                value_cache: torch.Tensor,
                                cum_log_probs: Optional[torch.Tensor] = None,
                                criteria_list: List[Any] = [],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not g_parallel_info.is_pp_last:
            return self.do_broadcast(sampler, output_token_ids, step, finished, sequence_lengths)

        logits = self._get_logits(hidden_states)

        sampler.do_sampling(SamplingParams(
            step,
            batch_size,
            beam_width,
            max_input_length,
            output_token_ids, # [max_seq_len, batch_size * beam_width]
            logits, # [batch_size * beam_width, vocab_size]
            finished, # [batch_size * beam_width]
            input_lengths, # [batch_size * beam_width]
            sequence_lengths, # [batch_size * beam_width]
            key_cache,
            value_cache,
            cum_log_probs, # [batch_size * beam_width]
            None,
            None,
        ))
        token_ids = output_token_ids.view(-1, batch_size, beam_width).permute(1, 2, 0)

        for i in range(batch_size):
            for stop_criteria in criteria_list:
                # NOTE: for beam search, generation should stop iff first beam is finished
                if stop_criteria(token_ids[i, 0, :step + 1].tolist(), int(input_lengths[i])):
                    finished.view([batch_size, beam_width])[i] = True
                    break

        return self.do_broadcast(sampler, output_token_ids, step, finished, sequence_lengths)
    
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
        return attention_mask

    def get_per_query_gpu_mem(self,
                          max_input_len: Optional[int] = None,
                          max_new_token: Optional[int] = None) -> int:
        """ caculate per query GPU memory (batch size 1) cost

        Args:
            max_input_len (int): max length of input tokens
            max_new_token (int): max length of generate tokens

        Returns:
            int: GPU memory cost
        """

        logging.info(f'max_seq_len {self.config.max_seq_len}')
        max_input_len = self.config.max_seq_len if max_input_len is None else max_input_len
        max_new_token = self.config.max_seq_len if max_new_token is None else max_new_token
        # magic num 10 for temp var: 1.2 * (decoder_normed_input_,self_attn_output_,decoder_layer_output_,padding_offset_,unormed_input_...)
        return 4 * self.config.head_num * self.config.size_per_head * (self.config.layer_num + 10) \
                * min(self.config.max_seq_len, max_input_len + max_new_token) * 1.2

    def get_max_query_batch_size(self,
                                 max_input_len: Optional[int] = None,
                                 max_new_token: Optional[int] = None,
                                 buffer_length: Optional[int] = 1024*1024*1024) -> int:
        """ caculate current query max batch size

        Args:
            max_input_len (int): max length of input tokens
            max_new_token (int): max length of generate tokens
            buffer_length (int): left buffer size on purpose. Defaults to 1024*1024*1024(1G).

        Returns:
            int: batch size
        """

        # get current available GPU memory
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = meminfo.free - buffer_length
        size_per_query = self.get_per_query_gpu_mem(max_input_len, max_new_token)
        logging.info(f'free_mem {free_mem}')
        logging.info(f'size_per_query {size_per_query}')
        return int(free_mem / size_per_query)

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
