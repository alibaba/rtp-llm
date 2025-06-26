
import torch

from pydantic import BaseModel as PyBaseModel
from typing import Any, Dict, List, Optional, Union, NamedTuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters, ConfigMode
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.task_type import TaskType
from rtp_llm.distribute.gang_info import get_gang_info
from rtp_llm.distribute.worker_info import ParallelInfo, g_parallel_info
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.utils.util import to_torch_dtype
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeights
from rtp_llm.model_loader.loader import ModelLoader, get_model_loader
from rtp_llm.utils.weight_type import WEIGHT_TYPE
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model

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
    QUANTIZATION_KEY = 'QUANTIZATION'
    SP_QUANTIZATION_KEY = 'SP_QUANTIZATION'
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
            quantization: str = ""
        ):
        self.model_type: str = model_type
        self.ckpt_path: str = ckpt_path
        self.tokenizer_path: str = tokenizer_path
        self.weight_type: WEIGHT_TYPE = weight_type
        self.act_type: WEIGHT_TYPE = act_type
        if self.weight_type == WEIGHT_TYPE.INT8 and not quantization:
            self.quantization = "INT8"
        else:
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

        self.py_model: Optional[GptModelBase] = None

    @timer_wrapper(description="load model")
    def load(self, parallel_info: ParallelInfo=g_parallel_info):
        self.model_weights_loader = self.create_model_loader(parallel_info)
        self._load(self.device)

        if True:  # TODO: add a config option to disable this
            self._create_python_model()

    def _create_python_model(self) -> Optional[GptModelBase]:
        # TODO(wangyin): in base_model this function should only return None
        # and the actual create method should be implemented in each derived class of specific models
        # There should also be a option to disable this python model creation
        self.py_model = Qwen3Model(self.config, self.weight)
        # self.py_model = GptModelExample(self.config, self.weight)

    def _load(self, device: str):
        self.weight: ModelWeights = self.model_weights_loader.load_weights(device=self.device)
        self._load_custom_module()
        self._load_multimodal()
        self.model_weights_loader.force_clean_cuda_memory()

    @classmethod
    def create_config(cls, model_config: ModelConfig,
                      parallel_info:ParallelInfo=g_parallel_info,
                      config_mode: ConfigMode = ConfigMode.ComplexMode) -> GptInitModelParameters:
        config: GptInitModelParameters = cls._create_config(model_config.ckpt_path)
        if config.hidden_size == 0:
            config.hidden_size = config.size_per_head * config.head_num
        config.update_common(
            ckpt_path=model_config.ckpt_path,
            tokenizer_path=model_config.tokenizer_path,
            quantization=model_config.quantization,
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

    @classmethod
    def from_config(cls, config: Any, parallel_info:ParallelInfo=g_parallel_info) -> 'BaseModel':
        model = cls(config)
        model.load(parallel_info)
        return model

    @staticmethod
    def get_weight_cls() -> ModelDeployWeightInfo:
        raise NotImplementedError

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype


    @timer_wrapper(description="init mutlimodal")
    def _may_init_multimodal(self):
        if self.is_multimodal():
            assert isinstance(self, MultiModalMixin) # for syntax check
            self.config.is_multimodal = True
            if self.parallel_info.tp_rank == 0:
                self.init_multimodal(self.config, self.device)

    @timer_wrapper(description="init custom_module")
    def _init_misc(self):
        self._may_init_multimodal()
        self.task_type = self.config.task_type
        self.custom_module = self._init_custom_module()
        self.compute_dtype: torch.dtype = to_torch_dtype(self.config.data_type)

    def _init_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.task_type, self.config, self.tokenizer)

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        assert config.tokenizer_path
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

    def load_tokenizer(self) -> None:
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

    def _init_database(self):
        self.database = CkptDatabase(self.config.ckpt_path, self.config.ptuning_path)
        # static lora load
        self.static_lora: bool = self.config.lora_infos is not None and len(self.config.lora_infos) == 1
        if self.static_lora:
            for name, path in self.config.lora_infos.items():
                self.database.load_lora(name, path)
            self.database.dump_lora_info()

    def _load_model_weights(self):
        self.weight: ModelWeights = self.model_weights_loader.load_weights(device=self.device)

    @timer_wrapper(description="load custom module")
    def _load_custom_module(self):
         if self.custom_module is not None:
                self.custom_module.init(self.weight)

    @timer_wrapper(description="load multimodal")
    def _load_multimodal(self):
        if self.config.vit_separation != 2 and self.is_multimodal():
            self.load_mm_weight(self.compute_dtype, self.config.tp_size, self.config.tp_rank, self.device)


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

        self._init_misc()
        self._init_database()

        tp_rank = self.parallel_info.tp_rank
        tp_size = self.parallel_info.tp_size

        weights_info: ModelDeployWeightInfo = self.get_weight_cls()(self.config, tp_size, tp_rank)
        misc_weights_info = self.custom_module.get_custom_weight_info() if self.custom_module else []
        return get_model_loader(self.task_type, weights_info, misc_weights_info, self.compute_dtype, self.database)

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        return config.eval_model_size()

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        return config.model_param_count
