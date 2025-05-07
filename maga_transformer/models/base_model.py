import os

import torch
import json
import logging
import math


import torch.nn.functional as F
from pydantic import BaseModel as PyBaseModel
from typing import Any, Dict, List, Optional, Union, NamedTuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, ConfigMode
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.task_type import TaskType
from maga_transformer.distribute.gang_info import get_gang_info
from maga_transformer.distribute.worker_info import ParallelInfo, g_parallel_info
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.models.downstream_modules.utils import create_custom_module
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.utils.fuser import fetch_remote_file_to_local
from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.utils.model_weight import W
from maga_transformer.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeights
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.loader import ModelLoader, get_model_loader
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.utils.multimodal_util import MultimodalInput
from maga_transformer.utils.database import CkptDatabase
from maga_transformer.utils.time_util import Timer
from maga_transformer.eplb.ep_balancer import ExpertBalancer

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
    first_token_cost_time: float = 0
    wait_time: float = 0
    pd_sep: bool = False
    cum_log_probs: List[float] = []
    beam_responses: List[str] = []
    softmax_probs: List[float] = []

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
        self.load_tokenizer()

    def _load_to_device(self, parallel_info: ParallelInfo=g_parallel_info):
        self.parallel_info = parallel_info
        self.device = self.parallel_info.device

        self.may_init_multimodal()
        self.init_misc()
        self.load(self.device)

    @classmethod
    def create_config(cls, model_config: ModelConfig,
                      parallel_info:ParallelInfo=g_parallel_info,
                      config_mode: ConfigMode = ConfigMode.ComplexMode) -> GptInitModelParameters:
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
            gen_num_per_circle=model_config.gen_num_per_circle,
            lora_infos=model_config.lora_infos,
            ptuning_path=model_config.ptuning_path,
            ref_module=model_config.ref_module,
            ref_dict=model_config.ref_dict,
            parallel_info=parallel_info,
            gang_info=get_gang_info(),
            config_mode=config_mode
        )
        cls._update_config(config)
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str) -> GptInitModelParameters:
        raise NotImplementedError()

    @classmethod
    def _update_config(cls, config: GptInitModelParameters):
        pass

    @staticmethod
    def _load_quant_config(ckpt_path: str,  config: GptInitModelParameters):
        quant_config_path = os.path.join(ckpt_path, 'smoothquant.ini')
        if os.path.exists(quant_config_path):
            config.quant_algo.setQuantAlgo('smooth_quant', 0, 0)
        per_tensor_config_path = os.path.join(ckpt_path, "pertensorquant.ini")

        if os.path.exists(per_tensor_config_path):
            config.quant_algo.setQuantAlgo('pertensor_quant', 0, 0)

        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return

        config_json = json.load(open(config_path))
        quant_config = None
        quant_method = None
        if config_json.get("quantization_config", None):
            quant_config = config_json["quantization_config"]
            quant_method = quant_config['quant_method'].lower()

        if config_json.get("quantization", None):
            quant_config = config_json["quantization"]
            quant_method = quant_config['quant_algo'].lower()
        if quant_config is None:
            return

        group_size = quant_config['group_size'] if 'group_size' in quant_config else 0
        bits = quant_config['bits'] if 'bits' in quant_config else 0
        if quant_method == 'fp8':
            bits = 8
            if 'weight_block_size' in quant_config:
                weight_block = quant_config.get("weight_block_size")
                assert isinstance(weight_block, list) and all(element == weight_block[0] for element in weight_block), f"weight_block_size: {weight_block} must be same"
                group_size = weight_block[0]

        config.quant_algo.setQuantAlgo(quant_method, bits, group_size)

    @classmethod
    def from_config(cls, config: Any, parallel_info:ParallelInfo=g_parallel_info) -> 'BaseModel':
        model = cls(config)
        model._load_to_device(parallel_info)
        return model

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
            if self.parallel_info.tp_rank == 0:
                self.init_multimodal(self.config)

    def init_misc(self):
        self.task_type = self.config.task_type
        self.custom_module = self.load_custom_module()
        self.compute_dtype: torch.dtype = to_torch_dtype(self.config.data_type)

    def split_slopes_tp(self, slopes: torch.Tensor):
        local_head_num = 1 if self.config.head_num == 1 else self.config.head_num // self.parallel_info.tp_size
        start_pos = local_head_num * self.parallel_info.tp_rank
        return slopes[start_pos: start_pos + local_head_num]

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        assert config.tokenizer_path
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

    def load_tokenizer(self):
        if not self.config.tokenizer_path:
            self.tokenizer = None
            return
        def error_handler(func: Any):
            def wrapper(*args: Any, **kwargs: Any):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    method_name = func.__name__
                    raise RuntimeError(f"{method_name} failed, with input args: {args}, kwargs: {kwargs}")
            return wrapper

        self.tokenizer = self.get_tokenizer(self.config)
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
            self.config.special_tokens.eos_token_id = self.tokenizer.eos_token_id
            self.config.update_task_prompt_tokens_id(self.tokenizer)
        if getattr(self.tokenizer, 'encode', None):
            self.tokenizer.encode = error_handler(self.tokenizer.encode)
        if getattr(self.tokenizer, 'decode', None):
            self.tokenizer.decode = error_handler(self.tokenizer.decode)

    def is_multimodal(self) -> bool:
        return isinstance(self, MultiModalMixin)

    def init_database(self):
        self.database = CkptDatabase(self.config.ckpt_path, self.config.ptuning_path)

    def load_static_lora(self):
        # static lora load
        self.static_lora: bool = self.config.lora_infos is not None and len(self.config.lora_infos) == 1
        if self.static_lora:
            for name, path in self.config.lora_infos.items():
                self.database.load_lora(name, path)
            self.database.dump_lora_info()

    def load_model_weight(self):
        weights_info = self.get_weight_cls()(self.config, self.parallel_info.tp_size, self.parallel_info.tp_rank)
        self.model_weights_loader = ModelLoader(weights_info, self.compute_dtype, self.database)
        self.weight: ModelWeights = self.model_weights_loader.load_weights(device=self.device)
        self._load_custom_module_weights(self.model_weights_loader)

    def _load_weights(self,
                      ref_dict: Dict[str, torch.Tensor] = {}):
        with Timer() as timer:
            self.init_database()
            self.load_static_lora()
            self.load_model_weight()
        logging.info(f'load weights time: {timer.cost_ms() / 1000 :.2f} s')

    def load_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.task_type, self.config, self.tokenizer)

    def _load_custom_module_weights(self, model_weights_loader: ModelLoader):
        if self.custom_module is not None:
            tensor_names = self.custom_module.handler.tensor_info()
            tensor_map: Dict[str, torch.Tensor] = {}
            for name in tensor_names:
                loaded_tensor = model_weights_loader.load_raw_tensor(name, device=self.device)
                tensor_map[name] = loaded_tensor
                self.weight.set_global_weight(name, loaded_tensor)
            self.custom_module.handler.init(tensor_map)

    def _initialize_weights(self):
        assert (self.weight is not None)

        embedding_weight = self.weight.global_weights.get(W.embedding, None)
        if embedding_weight != None:
            self.config.embedding_size = embedding_weight.shape[0]
            logging.info(f"embedding_size is {self.config.embedding_size}, vocab size is {self.config.vocab_size}")

        if self.config.vit_separation != 2 and self.is_multimodal():
            self.load_mm_weight(self.compute_dtype, self.device)

        if self.config.vit_separation != 1:
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

        ModelLoader.force_clean_cuda_memory()

    def _initialize_rope(self):
        pass

    def init_redundant_expert(self):
        if self.config.expert_num == 0:
            return

        expert_num = self.config.expert_num
        ep_size = self.parallel_info.ep_size
        layer_num = self.config.layer_num
        phy_exp_num = self.config.phy_exp_num

        phy2log = LoadConfig.create_redundant_expert(layer_num=layer_num, 
                                                           expert_num=expert_num, 
                                                           phy_exp_num=phy_exp_num, 
                                                           ep_size=ep_size, 
                                                           num_nodes=self.config.num_nodes)
        self.config.phy2log = phy2log

    def init_eplb_weight(self, weight: ModelWeights):
        expert_num = self.config.expert_num
        redundant_expert = self.config.phy_exp_num - expert_num
        layer_num = self.config.layer_num
        phy2log = self.config.phy2log

        if expert_num == 0 or redundant_expert == 0:
            return

        # init logic_expert_cnt and log2phy
        for layer_id in range(layer_num):
            logic_expert_cnt = torch.zeros((expert_num,), dtype=torch.int32)
            log2phy = torch.empty((expert_num, redundant_expert + 1), dtype=torch.int32).fill_(-1)
            layer_phy2log = phy2log[layer_id]

            for phy_exp_id, expert_id in enumerate(layer_phy2log):
                cnt = logic_expert_cnt[expert_id]
                log2phy[expert_id, cnt] = phy_exp_id
                logic_expert_cnt[expert_id] += 1

            weight.weights[layer_id][W.logic_expert_cnt] = logic_expert_cnt.contiguous().to(self.device)
            weight.weights[layer_id][W.log2phy] = log2phy.contiguous().to(self.device)

    def init_eplb_config(self, compute_dtype: torch.dtype):
        self.init_redundant_expert()
        if self.config.enable_eplb:
            model_path = None
            if self.config.is_mtp:
                model_path = self.config.ckpt_path
            else:
                model_path = fetch_remote_file_to_local(
                    os.environ.get(
                        "ORIGINAL_CHECKPOINT_PATH", self.config.ckpt_path
                    )
                )
            weights_info: ModelDeployWeightInfo = self.get_weight_cls()(self.config, self.parallel_info.tp_size, self.parallel_info.tp_rank)

            ep_lb_database = CkptDatabase(model_path)
            self.ep_balancer = ExpertBalancer(
                weights_info=weights_info,
                compute_dtype=compute_dtype,
                database=ep_lb_database
            )
            self.config.py_eplb = self.ep_balancer

    def load(self, device: str):
        self.init_eplb_config(compute_dtype=self.compute_dtype)
        self._load_weights()
        self.init_eplb_weight(self.weight)
        self._initialize_rope()
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

    def create_model_loader(self, parallel_info: ParallelInfo) -> ModelLoader:
        self.parallel_info = parallel_info
        self.device = self.parallel_info.device

        self.may_init_multimodal()
        self.init_misc()
        self.init_database()
        self.load_static_lora()
        self.init_eplb_config(compute_dtype=self.compute_dtype)

        tp_rank = self.parallel_info.tp_rank
        tp_size = self.parallel_info.tp_size

        weights_info: ModelDeployWeightInfo = self.get_weight_cls()(self.config, tp_size, tp_rank)
        return get_model_loader(weights_info, self.compute_dtype, self.database)

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        return config.eval_model_size()

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        return config.model_param_count
