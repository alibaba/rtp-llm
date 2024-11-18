import os
import torch
import json
import logging
import math


import torch.nn.functional as F
from dataclasses import dataclass, field
from pydantic import BaseModel as PyBaseModel
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.task_type import TaskType
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.models.downstream_modules.utils import create_custom_module
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.utils.model_weight import W, ModelDeployWeightInfo
from maga_transformer.utils.model_weights_loader import get_model_weights_loader, estimate_load_parallel_num, ModelWeightsLoader
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.utils.multimodal_util import MultimodalInput
from maga_transformer.utils.database import CkptDatabase, ModuleDatabase, DictDatabase
from maga_transformer.utils.time_util import Timer
from maga_transformer.ops.comm.parallel_op import ParallelEmbedding, ParallelLinear

FT_DEFAULT_MAX_NEW_TOKENS = 2048

class EmbeddingOutput:
    text_embedding: torch.Tensor
    extra_input: Optional[torch.Tensor]

    def __init__(self, text_embedding: torch.Tensor, extra_input: Optional[List[torch.Tensor]]):
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
class GenerateInput(PyBaseModel):
    request_id: int
    token_ids: torch.Tensor
    mm_inputs: List[MultimodalInput]
    generate_config: GenerateConfig
    tokenizer: Any = None # TODO: remove this
    prefix_length: int = 0
    token_type_ids: List[int] = []

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
    step_output_len: int = 0
    fallback_tokens: int = 0
    fallback_times: int = 0
    pd_sep: bool = False
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
    all_probs: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

class GenerateOutputs(PyBaseModel):
    generate_outputs: List[GenerateOutput] = []

class GenerateResponse(PyBaseModel):
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
    def __init__(
            self,
            model_type: str = "",
            ckpt_path: str = "",
            tokenizer_path: str = "",
            weight_type: WEIGHT_TYPE = WEIGHT_TYPE.FP16,
            act_type: WEIGHT_TYPE = WEIGHT_TYPE.FP16,
            max_seq_len: int = 0,
            seq_size_per_block: int = 8,
            gen_num_per_circle: int = 1,
            ptuning_path: Optional[str] = None,
            lora_infos: Optional[Dict[str, str]] = None,
            ref_module: Optional[torch.nn.Module] = None,
            ref_dict: Dict[str, torch.Tensor] = {},
            sp_type: str = "",
        ):
        self.model_type: str = model_type
        self.ckpt_path: str = ckpt_path
        self.tokenizer_path: str = tokenizer_path
        self.weight_type: WEIGHT_TYPE = weight_type
        self.act_type: WEIGHT_TYPE = act_type
        self.max_seq_len: int = max_seq_len
        self.seq_size_per_block: int = seq_size_per_block
        self.gen_num_per_circle: int = gen_num_per_circle
        self.ptuning_path: Optional[str] = ptuning_path
        self.lora_infos: Optional[Dict[str, str]] = lora_infos
        self.ref_module: Optional[torch.nn.Module] = ref_module
        self.ref_dict: Dict[str, torch.Tensor] = ref_dict
        self.sp_type: str = sp_type

    @property
    def int8_mode(self):
        return True if self.weight_type == WEIGHT_TYPE.INT8 else False

    def add_ref_module(self, ref_module: Optional[torch.nn.Module]):
        self.ref_module = ref_module

    def add_ref_dict(self, ref_dict: Dict[str, torch.Tensor]):
        self.ref_dict = ref_dict

    def _replace(self, **kwargs: Any):
        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
        return self

def get_slopes(n: int) -> List[float]:
    def get_slopes_power_of_2(n: int) -> List[float]:
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))

        return get_slopes_power_of_2(closest_power_of_2) + \
            get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

class BaseModel(object):

    config: GptInitModelParameters
    vocab_size_padded: int
    device: str

    def __init__(self, config: GptInitModelParameters) -> None:
        self.config = config
        self.weight = None

        self.linear_bias_slopes: Optional[torch.Tensor] = None
        self.prefix_tokens: Optional[torch.Tensor] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.max_input_buffer_len: int = 0

        self.task_type: TaskType = TaskType.LANGUAGE_MODEL
        self.custom_module: Optional[CustomModule] = None

        self.default_generate_config: GenerateConfig = GenerateConfig()
        self.device = g_parallel_info.device

        self.load_tokenizer()
        self.init_misc()
        self.may_init_multimodal()
        self.load(self.device)

    @classmethod
    def create_config(cls, model_config: ModelConfig) -> GptInitModelParameters:
        config: GptInitModelParameters = cls._create_config(model_config.ckpt_path)
        cls._load_quant_config(model_config.ckpt_path, config)
        if config.hidden_size == 0:
            config.hidden_size = config.size_per_head * config.head_num
        config.update_common(
            ckpt_path=model_config.ckpt_path,
            tokenizer_path=model_config.tokenizer_path,
            int8_mode=model_config.int8_mode,
            data_type=model_config.act_type,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.seq_size_per_block,
            tp_size=g_parallel_info.tp_size,
            gen_num_per_circle=model_config.gen_num_per_circle,
            lora_infos=model_config.lora_infos,
            ptuning_path=model_config.ptuning_path,
            ref_module=model_config.ref_module,
            ref_dict=model_config.ref_dict
        )
        return config

    @staticmethod
    def _create_config(ckpt_path: str) -> GptInitModelParameters:
        raise NotImplementedError()

    @staticmethod
    def _load_quant_config(ckpt_path: str,  config: GptInitModelParameters):
        quant_config_path = os.path.join(ckpt_path, 'smoothquant.ini')
        if os.path.exists(quant_config_path):
            config.quant_algo.setQuantAlgo('smooth_quant', 0, 0)
        per_tensor_config_path = os.path.join(ckpt_path, "pertensorquant.ini")

        if os.path.exists(per_tensor_config_path):
            config.quant_algo.setQuantAlgo('pertensor_quant', 0, 0)

        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            config_json = json.load(open(config_path))
            quant_config = config_json.get("quantization_config", None)
            if quant_config is not None:
                config.quant_algo.setQuantAlgo(quant_config['quant_method'], quant_config["bits"], quant_config.get("group_size", 0))

    @classmethod
    def from_config(cls, config: Any) -> 'BaseModel':
        return cls(config)

    @staticmethod
    def get_weight_cls() -> ModelDeployWeightInfo:
        raise NotImplementedError

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype

    def may_init_multimodal(self):
        if self.is_multimodal():
            assert isinstance(self, MultiModalMixin) # for syntax check
            self.config.is_multimodal = True
            if g_parallel_info.tp_rank == 0:
                self.init_multimodal(self.config)

    def init_misc(self):
        self.task_type = self.config.task_type
        self.custom_module = self.load_custom_module()
        self.compute_dtype = to_torch_dtype(self.config.data_type)

    def split_slopes_tp(self, slopes: torch.Tensor):
        local_head_num = 1 if self.config.head_num == 1 else self.config.head_num // g_parallel_info.tp_size
        start_pos = local_head_num * g_parallel_info.tp_rank
        return slopes[start_pos: start_pos + local_head_num]

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        assert config.tokenizer_path
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

    def load_tokenizer(self):
        if self.config.tokenizer_path:
            self.tokenizer = self.get_tokenizer(self.config)
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
                self.config.special_tokens.eos_token_id = self.tokenizer.eos_token_id
            self.config.update_task_prompt_tokens_id(self.tokenizer)

    def is_multimodal(self) -> bool:
        return isinstance(self, MultiModalMixin)

    def init_database(self):
        if self.config.ref_module is not None:
            self.database = ModuleDatabase(self.config.ref_module)
        elif len(self.config.ref_dict) != 0:
            self.database = DictDatabase(self.config.ref_dict)
        else:
            self.database = CkptDatabase(self.config.ckpt_path, self.config.ptuning_path)

    def load_static_lora(self):
        # static lora load
        self.static_lora: bool = self.config.lora_infos is not None and len(self.config.lora_infos) == 1
        if self.static_lora:
            for name, path in self.config.lora_infos.items():
                self.database.load_lora(name, path)
            self.database.dump_lora_info()

    def load_model_weight(self):
        load_parallel_num = estimate_load_parallel_num(
            self.config, g_parallel_info.tp_size)
        weights_info = self.get_weight_cls()(self.config, g_parallel_info.tp_size, g_parallel_info.tp_rank)
        self.model_weights_loader = get_model_weights_loader(weights_info, self.database, compute_dtype=self.compute_dtype)
        self.weight = self.model_weights_loader.load_weights_from_scratch(num_process=load_parallel_num, device=self.device)
        self._load_custom_module_weights(self.model_weights_loader)
        if self.static_lora:
            lora_name = list(self.config.lora_infos.keys())[0]
            self.model_weights_loader.show_warns(lora_name=lora_name)
        else:
            self.model_weights_loader.show_warns()

    def _load_weights(self,
                      ref_dict: Dict[str, torch.Tensor] = {}):
        with Timer() as timer:
            self.init_database()
            self.load_static_lora()
            self.load_model_weight()
        logging.info(f'load weights time: {timer.cost_ms() / 1000 :.2f} s')

    def load_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.task_type, self.config, self.tokenizer)

    def _load_custom_module_weights(self, model_weights_loader: ModelWeightsLoader):
        if self.custom_module is not None:
            tensor_names = self.custom_module.handler.tensor_info()
            tensor_map: Dict[str, torch.Tensor] = {}
            for name in tensor_names:
                tensors = model_weights_loader.load_tensor(name)
                if len(tensors) != 1:
                    raise Exception(f"load tensor {name} failed, get len=={len(tensors)}")
                loaded_tensor = tensors[0].to(self.device)
                tensor_map[name] = loaded_tensor
                self.weight.set_global_weight(name, loaded_tensor)
            self.custom_module.handler.init(tensor_map)

    def _initialize_weights(self):
        assert (self.weight is not None)

        if self.task_type == TaskType.LANGUAGE_MODEL:
            lm_head_w = self.weight.steal_global_weight(W.lm_head)
            if lm_head_w == None:
                lm_head_w = self.weight.global_weights[W.embedding]
            if self.config.normalize_lm_head_weight:
                lm_head_w = F.normalize(lm_head_w)
            if self.config.logit_scale != 1.0:
                lm_head_w = self.config.scale_logit * lm_head_w
            self.weight.set_global_weight(W.lm_head, lm_head_w)
        else:
            # Some LLM can be used for other tasks, e.g. classification, in which case lm_head is not needed
            self.weight.steal_global_weight(W.lm_head)

        pos_weight = self.weight.global_weights.get(W.positional_embedding, None)
        if pos_weight != None:
            if pos_weight.shape[0] < self.config.max_seq_len:
                raise Exception(f"positon_weight has shape: {pos_weight.shape}, but max_seq_len is: {self.config.max_seq_len} > {pos_weight.shape[0]}")
            pos_weight = pos_weight[:self.config.max_seq_len].to(self.device)
            self.weight.set_global_weight(W.positional_embedding, pos_weight)

        if self.config.use_attention_linear_bias:
            slopes = torch.Tensor(get_slopes(self.config.head_num))
            slopes = self.split_slopes_tp(slopes)
            self.linear_bias_slopes = slopes.to(torch.float).to(self.device)
            self.weight.set_global_weight(W.linear_bias_slopes, self.linear_bias_slopes)

        if self.config.quant_algo.isPerTensorQuant() and \
            (self.weight.global_weights.get(W.pre_decoder_ln_static_quant, None) == None or \
            self.weight.global_weights.get(W.pre_decoder_ln_static_quant_reciprocal, None) == None):
                raise Exception("pre_decoder_ln_static_quant and pre_decoder_ln_static_quant_reciprocal \
                                are quired for per tensor quantization")

        if self.is_multimodal():
            self.load_mm_weight(self.compute_dtype, self.device)

        torch.cuda.empty_cache()

    def load(self, device: str):
        self._load_weights()
        self._initialize_weights()

    def dup_dim0_for_beam_search(self, t: torch.Tensor, beam_width: int) -> torch.Tensor:
        shape = list(t.shape)
        return t.unsqueeze(1).repeat([1, beam_width] + [1] * len(shape[1:])).reshape([-1] + shape[1:]).contiguous()

    def extend_context_combo_token_types(self, token_types: List[int]) -> List[int]:
        return []

    def extend_generate_combo_token_types(self, combo_tokens: List[int]) -> List[int]:
        return []

    def create_context_position_ids(self, input_lengths: Union[List[int], torch.Tensor]):
        return torch.concat([torch.arange(int(input_length), dtype=torch.int32) for input_length in input_lengths], dim=0)

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

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        return config.model_param_count
