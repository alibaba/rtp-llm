from typing import Any, Dict, List, NamedTuple, Optional
from dataclasses import dataclass, field, asdict
import json
import torch
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.weight_type import WEIGHT_TYPE


# Pre-import for performance
from dataclasses import asdict, is_dataclass
from enum import Enum

def _asdict_with_enum_handling(obj, _cache=None):
    """处理包含枚举的 dataclass 字典转换"""
    # 使用弱引用缓存来避免循环引用和重复计算
    if _cache is None:
        _cache = set()

    obj_id = id(obj)
    if obj_id in _cache:
        return obj  # 避免循环引用
    _cache.add(obj_id)

    try:
        def convert_enum(value):
            # 处理各种枚举类型
            if isinstance(value, Enum):
                return value.value
            # 处理嵌套的 dataclass with optimized type check
            elif is_dataclass(value) and not isinstance(value, (type, type(open))):
                return _asdict_with_enum_handling(value, _cache)
            # 递归处理字典 - optimize
            elif isinstance(value, dict):
                if not value:
                    return value
                return {k: convert_enum(v) for k, v in value.items()}
            # 递归处理列表 - check for empty
            elif isinstance(value, list):
                if not value:
                    return value
                return [convert_enum(item) for item in value]
            elif isinstance(value, tuple):
                if not value:
                    return value
                return tuple(convert_enum(item) for item in value)
            elif isinstance(value, set):
                if not value:
                    return value
                return {convert_enum(item) for item in value}
            else:
                return value

        result = asdict(obj)
        return {k: convert_enum(v) for k, v in result.items()}
    finally:
        _cache.discard(obj_id)


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

    def __post_init__(self):
        # 处理 None 值
        if self.token_type_ids is None:
            self.token_type_ids = []

    @property
    def input_length(self):
        return self.token_ids.shape[-1]

    @property
    def prompt_length(self):
        return self.token_ids.shape[-1] - self.prefix_length

    def update_prefix(self, prefix_tokens: torch.Tensor):
        self.token_ids = torch.concat([prefix_tokens, self.token_ids], dim=0)
        self.prefix_length = prefix_tokens.nelement()

    # 兼容性方法
    def dict(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def model_dump(self, *args, **kwargs):
        return self.dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
            # 过滤掉 exclude_none 参数，避免传递给 json.dumps
            json_kwargs = {k: v for k, v in kwargs.items() if k != 'exclude_none'}
            return json.dumps(result, *args, **json_kwargs)
        return json.dumps(result, *args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        return self.json(*args, **kwargs)


@dataclass
class AuxInfo:
    cost_time: float = 0
    iter_count: int = 0
    prefix_len: int = 0
    input_len: int = 0
    output_len: int = 0
    step_output_len: int = 0
    fallback_tokens: int = 0
    fallback_times: int = 0
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

    def __post_init__(self):
        # 处理 None 值
        if self.cum_log_probs is None:
            self.cum_log_probs = []
        if self.beam_responses is None:
            self.beam_responses = []
        if self.softmax_probs is None:
            self.softmax_probs = []
        if self.role_addrs is None:
            self.role_addrs = []

    # 兼容性方法
    def dict(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def model_dump(self, *args, **kwargs):
        return self.dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
            # 过滤掉 exclude_none 参数，避免传递给 json.dumps
            json_kwargs = {k: v for k, v in kwargs.items() if k != 'exclude_none'}
            return json.dumps(result, *args, **json_kwargs)
        return json.dumps(result, *args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        return self.json(*args, **kwargs)


@dataclass
class GenerateOutput:
    hidden_states: Optional[torch.Tensor] = None
    all_hidden_states: Optional[torch.Tensor] = None
    output_ids: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    finished: bool = False
    aux_info: AuxInfo = field(default_factory=AuxInfo)
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    all_probs: Optional[torch.Tensor] = None

    def __post_init__(self):
        # 处理 None 值
        if self.aux_info is None:
            self.aux_info = AuxInfo()

    # 兼容性方法
    def dict(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def model_dump(self, *args, **kwargs):
        return self.dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
            # 过滤掉 exclude_none 参数，避免传递给 json.dumps
            json_kwargs = {k: v for k, v in kwargs.items() if k != 'exclude_none'}
            return json.dumps(result, *args, **json_kwargs)
        return json.dumps(result, *args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        return self.json(*args, **kwargs)


@dataclass
class GenerateOutputs:
    generate_outputs: List[GenerateOutput] = field(default_factory=list)

    def __post_init__(self):
        # 处理 None 值
        if self.generate_outputs is None:
            self.generate_outputs = []

    # 兼容性方法
    def dict(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def model_dump(self, *args, **kwargs):
        return self.dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
            # 过滤掉 exclude_none 参数，避免传递给 json.dumps
            json_kwargs = {k: v for k, v in kwargs.items() if k != 'exclude_none'}
            return json.dumps(result, *args, **json_kwargs)
        return json.dumps(result, *args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        return self.json(*args, **kwargs)


@dataclass
class GenerateResponse:
    generate_outputs: GenerateOutputs
    generate_texts: List[str] = field(default_factory=list)

    def __post_init__(self):
        # 处理 None 值
        if self.generate_texts is None:
            self.generate_texts = []

    # 兼容性方法
    def dict(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
        return result

    def model_dump(self, *args, **kwargs):
        return self.dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        result = _asdict_with_enum_handling(self)
        if kwargs.get('exclude_none'):
            result = {k: v for k, v in result.items() if v is not None}
            # 过滤掉 exclude_none 参数，避免传递给 json.dumps
            json_kwargs = {k: v for k, v in kwargs.items() if k != 'exclude_none'}
            return json.dumps(result, *args, **json_kwargs)
        return json.dumps(result, *args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        return self.json(*args, **kwargs)


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

    model_type: str = ""
    ckpt_path: str = ""
    tokenizer_path: str = ""
    act_type: str = None
    kv_cache_type: str = None
    quantization: str = ""
    max_seq_len: int = 0
    seq_size_per_block: int = 8
    gen_num_per_circle: int = 1
    ptuning_path: Optional[str] = None
    lora_infos: Optional[Dict[str, str]] = None
    ref_module: Optional[torch.nn.Module] = None
    ref_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    sp_type: str = ""

    def __post_init__(self):
        # 处理 None 值
        if self.lora_infos is None:
            self.lora_infos = {}
        if self.ref_dict is None:
            self.ref_dict = {}

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
