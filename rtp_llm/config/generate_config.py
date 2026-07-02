import copy
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.ops import RoleType
from rtp_llm.utils.check_util import *
from rtp_llm.utils.util import check_with_info


class RequestFormat:
    RAW = "raw"
    CHAT_API = "chatapi"


class ReturnAllProbsMode:
    NONE = 0
    DEFAULT = 1
    ORIGINAL = 2


class RoleAddr(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: RoleType
    ip: str
    http_port: int
    grpc_port: int

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v):
        """Convert string to RoleType enum for deserialization."""
        if isinstance(v, str):
            return getattr(RoleType, v)
        elif isinstance(v, RoleType):
            return v
        else:
            raise ValueError(
                f"RoleType must be a string or RoleType enum, got {type(v)}"
            )

    @field_serializer("role")
    def serialize_role(self, role: RoleType, _info) -> str:
        """Serialize RoleType enum to its name string for JSON serialization."""
        return role.name


class GenerateConfig(BaseModel):
    max_new_tokens: int = 32000
    # only for qwen agent fncall check max input tokens
    max_input_tokens: int = 32000
    max_thinking_tokens: int = 32000
    in_think_mode: bool = (
        False  # same as `enable_thinking` in chat_template_kwargs, discard one in the future
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    end_think_token_ids: List[int] = []
    num_beams: int = 1
    variable_num_beams: List[int] = []
    do_sample: bool = True
    # 0 mean not use num_return_sequences,
    # whether to enable num_return_sequences, the output format of the results is inconsistent.
    num_return_sequences: int = 0
    top_k: Union[List[int], int] = 0
    top_p: Union[List[float], float] = 1.0
    temperature: Union[List[float], float] = 1.0
    repetition_penalty: Union[List[float], float] = 1.0
    presence_penalty: Union[List[float], float] = 0.0
    frequency_penalty: Union[List[float], float] = 0.0
    min_new_tokens: Union[List[int], int] = 0
    no_repeat_ngram_size: Optional[Union[List[int], int]] = None
    # 生成式推荐：组合 token 粒度去重与曝光过滤。
    # combo_token_size 表示一个商品由多少个连续 token 组成（例如三层语义 ID = 3），0 表示关闭该功能。
    # banned_combo_token_ids 是禁止生成的商品 token 组合列表，每项长度必须等于 combo_token_size。
    # auto_parse_banned_combo 为 True 时，frontend 会自动从 prompt 中解析「已推荐曝光的商品序列」并
    # 填充到 banned_combo_token_ids；默认关闭，避免影响非推荐场景。
    combo_token_size: int = 0
    banned_combo_token_ids: List[List[int]] = []
    auto_parse_banned_combo: bool = False
    # 跨序列 combo 去重：当 num_return_sequences > 1 时，任一序列生成完整 combo 后自动广播到
    # 其他序列的 banned_combos，确保多条序列输出互不重复。默认关闭。
    # 跨序列去重开关（primary-protected 非对称模式）：开启后，主序列（序列 0）仅保留自身 ban、
    # 不受其他序列影响；补充序列接收所有序列 banned_combos 并集，保证彼此不重复。
    enable_cross_sequence_ban: bool = False
    # 跨序列分叉起始商品位置：前 N 个商品所有序列保持 greedy 一致，
    # 从第 N+1 个商品开始对非主序列施加 top-K 遮蔽制造分叉。默认 0（立即分叉）。
    cross_seq_diverge_start_combo: int = 0

    @field_validator("cross_seq_diverge_start_combo", mode="before")
    @classmethod
    def _clamp_diverge_start_combo(cls, v):
        """确保分叉起始位置非负，负值 clamp 到 0 并输出 warning。"""
        if v is None:
            return 0
        try:
            val = int(v)
        except (TypeError, ValueError) as e:
            logging.getLogger(__name__).warning(
                "cross_seq_diverge_start_combo received non-integer value %r, defaulting to 0: %s", v, e)
            return 0
        if val < 0:
            logging.getLogger(__name__).warning(
                "cross_seq_diverge_start_combo is negative (%d), clamped to 0", val)
            return 0
        if val > 100:
            logging.getLogger(__name__).warning(
                "cross_seq_diverge_start_combo=%d is very large, top-K diverge masking may never activate", val)
        return val

    @model_validator(mode='after')
    def _check_cross_seq_ban_compatibility(self):
        """cross_sequence_ban 与 beam search / combo_token_size<2 互斥，不匹配时输出 warning。"""
        if self.enable_cross_sequence_ban:
            if self.has_num_beams():
                logging.getLogger(__name__).warning(
                    "enable_cross_sequence_ban is incompatible with beam search "
                    "(max_num_beams=%d), cross_sequence_ban will be disabled at runtime",
                    self.max_num_beams())
            elif self.combo_token_size < 2:
                logging.getLogger(__name__).warning(
                    "enable_cross_sequence_ban requires combo_token_size>=2, got %d, "
                    "cross_sequence_ban will be disabled at runtime", self.combo_token_size)
        return self

    random_seed: Optional[Union[List[int], int]] = None
    top_p_decay: Optional[Union[List[float], float]] = None
    top_p_min: Optional[Union[List[float], float]] = None
    top_p_reset_ids: Optional[Union[List[int], int]] = None
    stop_words_str: List[str] = []
    stop_words_list: List[List[int]] = []
    bad_words_list: Optional[Union[List[List[List[int]]], List[List[int]]]] = None
    eos_token_id: Optional[Union[List[int], int]] = None
    pad_token_id: Optional[Union[List[int], int]] = None
    bos_token_id: Optional[Union[List[int], int]] = None
    using_hf_sampling: bool = False
    print_stop_words: bool = False
    timeout_ms: Optional[int] = -1
    ttft_timeout_ms: Optional[int] = -1
    traffic_reject_priority: Optional[int] = 100
    chat_id: Optional[str] = None
    task_id: Optional[Union[str, int]] = None
    request_format: str = RequestFormat.RAW
    # calculate_loss style: 0 for not calculate; 1 for sum; 2 for each token
    calculate_loss: int = 0
    return_logits: bool = False
    logits_index: Optional[int] = None
    return_incremental: bool = False
    return_hidden_states: bool = False
    return_all_hidden_states: bool = False
    hidden_states_cut_dim: int = 0
    normalized_hidden_states: bool = False
    select_tokens_str: List[str] = []
    select_tokens_id: List[int] = []
    return_input_ids: bool = False
    return_output_ids: bool = False
    md5_value: str = ""
    custom_prop: str = "{}"
    sp_advice_prompt: str = ""
    sp_advice_prompt_token_ids: List[int] = []
    sp_edit: bool = False
    force_disable_sp_run: bool = False
    force_sp_accept: bool = False
    return_cum_log_probs: bool = False
    return_all_probs: int = ReturnAllProbsMode.NONE
    return_softmax_probs: bool = False
    aux_info: bool = True
    can_use_pd_separation: bool = True
    gen_timeline: bool = False
    profile_step: int = 3
    profile_trace_name: str = ""
    out_prefix: str = ""
    # for load balance
    role_addrs: List[RoleAddr] = []
    trace_id: str = ""

    ignore_eos: bool = False
    skip_special_tokens: bool = False
    # lora
    adapter_name: Optional[Union[str, List[str]]] = None
    is_streaming: bool = False

    # multimodal preprocess
    resized_shape: Optional[List[int]] = None
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
    fps: Optional[int] = None
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None
    crop_positions: Optional[List[float]] = None
    mm_timeout_ms: Optional[int] = None

    # whether add vision id in chat template; only use in frontend
    add_vision_id: bool = True

    # 是否允许tool_call专用的标签如<tool_call>作为content传出, 优化tool_call失败时的用户体验
    tool_call_message_extract_strategy: str = "default"  # default/skip_on_failure

    global_request_id: int = -1

    # 只有开启环境变量 REUSE_CACHE 时才生效
    reuse_cache: bool = True

    enable_device_cache: bool = True

    enable_memory_cache: bool = True

    enable_remote_cache: bool = True
    # 是否强制相同 request_id 的 stream 在一批中调度
    force_batch: bool = False
    batch_group_timeout: Optional[int] = None  # ms

    unique_key: str = ""

    @field_validator("return_all_probs", mode="before")
    @classmethod
    def _coerce_return_all_probs(cls, v):
        """Legacy bool → ReturnAllProbsMode int. True → DEFAULT, False → NONE.

        return_all_probs was a `bool` field before the ORIGINAL/DEFAULT/NONE three-state
        refactor. Old callers may still pass `True/False`; pydantic strict mode would
        otherwise reject the bool→int implicit conversion.
        """
        if isinstance(v, bool):
            return ReturnAllProbsMode.DEFAULT if v else ReturnAllProbsMode.NONE
        return v

    def gen_hash_value(self):
        cp = copy.copy(self)
        cp.max_new_tokens = 0
        cp.chat_id = None
        cp.random_seed = None
        cp.md5_value = ""
        cp.timeout_ms = -1
        self.md5_value = hashlib.md5(cp.__str__().encode()).hexdigest()

    def max_num_beams(self):
        return (
            self.num_beams
            if len(self.variable_num_beams) == 0
            else max(self.variable_num_beams)
        )

    def has_num_beams(self):
        return self.max_num_beams() > 1

    def is_same(self, config: "GenerateConfig") -> bool:
        return self.md5_value == config.md5_value

    def update(self, new: Dict[str, Any]):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def update_and_pop(self, new: Dict[str, Any]):
        to_remove: List[str] = []
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
        return {k: v for k, v in new.items() if k not in to_remove}

    @staticmethod
    def create_generate_config(
        generate_config: Dict[str, Any], **kwargs: Any
    ) -> "GenerateConfig":
        generate_config.update(kwargs)
        try:
            config = GenerateConfig(**generate_config)
        except Exception as e:
            raise FtRuntimeException(
                ExceptionType.ERROR_GENERATE_CONFIG_FORMAT,
                f"generate_config validate failed: {str(e)}",
            )
        config.validate()
        return config

    def convert_select_tokens(self, vocab_size, tokenizer):
        for token_str in self.select_tokens_str:
            self.select_tokens_id += tokenizer.encode(token_str)
        if not all(
            token_id < vocab_size and token_id >= 0
            for token_id in self.select_tokens_id
        ):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"token_id in select_tokens_id {self.select_tokens_id} should be less than vocab_size {vocab_size}, and shoud not be negative",
            )

    def add_special_tokens(self, special_tokens: Any):
        # 这里假设外部传进来的stop_word_list和stop_word_str都不包含batch维度
        self.stop_words_list += special_tokens.stop_words_id_list
        self.stop_words_str += special_tokens.stop_words_str_list

    def add_thinking_params(self, tokenizer, generate_env_config):
        """Add thinking parameters from generate_env_config.

        Args:
            tokenizer: Tokenizer instance.
            generate_env_config: GenerateEnvConfig object.
        """

        end_think_token_id = generate_env_config.think_end_token_id
        self.end_think_token_ids = (
            [end_think_token_id] if end_think_token_id != -1 else []
        )
        if (
            bool(generate_env_config.think_mode)
            and tokenizer
            and end_think_token_id == -1
        ):
            think_end_tag: str = generate_env_config.think_end_tag.encode(
                "utf-8"
            ).decode("unicode_escape")
            tokenized_result: List[int] = tokenizer.encode(
                think_end_tag, add_special_tokens=False
            )
            self.end_think_token_ids = tokenized_result
        self.in_think_mode = (
            bool(generate_env_config.think_mode) and len(self.end_think_token_ids) >= 0
        )

    def add_stop_ids_from_str(self, tokenizer):
        ids_list = []
        for word in self.stop_words_str:
            if tokenizer is None:
                return
            else:
                token_id = tokenizer.convert_tokens_to_ids(word)
                if isinstance(token_id, int):
                    ids_list.append([token_id])
                elif isinstance(token_id, list):
                    ids_list.append(token_id)
                else:
                    ids_list.append(tokenizer.encode(word, add_special_tokens=True))

        # remove duplicate element
        for item in ids_list:
            if item not in self.stop_words_list:
                self.stop_words_list.append(item)

    def validate(self):
        try:
            check_with_info(
                is_union_positive_integer(self.top_k),
                f"top_k {self.top_k} is wrong data type",
            )
            check_with_info(
                is_union_positive_number(self.top_p),
                f"top_p {self.top_p} is wrong data type",
            )
            check_with_info(
                is_union_positive_integer(self.min_new_tokens),
                f"min_new_tokens {self.min_new_tokens} is wrong data type",
            )
            check_with_info(
                is_union_positive_number(self.repetition_penalty),
                f"repetition_penalty {self.repetition_penalty} is wrong data type",
            )
            check_with_info(
                is_union_number(self.presence_penalty),
                f"presence_penalty {self.presence_penalty} is wrong data type",
            )
            check_with_info(
                is_union_number(self.frequency_penalty),
                f"frequency_penalty {self.frequency_penalty} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.max_new_tokens),
                f"max_new_tokens {self.max_new_tokens} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.num_beams),
                f"num_beams {self.num_beams} is wrong data type",
            )
            check_with_info(
                is_list_positive_integer(self.variable_num_beams),
                f"variable_num_beams {self.variable_num_beams} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.num_return_sequences),
                f"num_return_sequences {self.num_return_sequences} is wrong data type",
            )
            check_with_info(
                is_union_positive_number(self.temperature),
                f"temperature {self.temperature} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.no_repeat_ngram_size),
                f"no_repeat_ngram_size {self.no_repeat_ngram_size} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.random_seed),
                f"random_seed {self.random_seed} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_number, self.top_p_decay),
                f"top_p_decay {self.top_p_decay} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_number, self.top_p_min),
                f"top_p_min {self.top_p_min} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.top_p_reset_ids),
                f"top_p_reset_ids {self.top_p_reset_ids} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.eos_token_id),
                f"eos_token_id {self.eos_token_id} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.pad_token_id),
                f"pad_token_id {self.pad_token_id} is wrong data type",
            )
            check_with_info(
                check_optional(is_union_positive_integer, self.bos_token_id),
                f"bos_token_id {self.bos_token_id} is wrong data type",
            )
            check_with_info(
                is_list_positive_integer_list(self.stop_words_list),
                f"stop_words_list {self.stop_words_list} is wrong data type",
            )
            check_with_info(
                is_positive_integer(self.combo_token_size),
                f"combo_token_size {self.combo_token_size} is wrong data type",
            )
            check_with_info(
                is_list_positive_integer_list(self.banned_combo_token_ids),
                f"banned_combo_token_ids {self.banned_combo_token_ids} is wrong data type",
            )
            if self.combo_token_size > 0:
                check_with_info(
                    all(
                        len(combo) == self.combo_token_size
                        for combo in self.banned_combo_token_ids
                    ),
                    f"every item in banned_combo_token_ids must have length == combo_token_size {self.combo_token_size}",
                )
            check_with_info(
                is_union_positive_integer(self.sp_advice_prompt_token_ids),
                f"sp_advice_prompt_token_ids {self.sp_advice_prompt_token_ids} is wrong data type",
            )
            if self.in_think_mode:
                check_with_info(
                    is_positive_integer(self.max_thinking_tokens),
                    f"max_thinking_tokens {self.max_thinking_tokens} is wrong data type",
                )
                check_with_info(
                    is_list_positive_integer(self.end_think_token_ids),
                    f"end_think_token_ids {self.end_think_token_ids} is wrong data type",
                )
            calculate_loss_list = [0, 1, 2]
            check_with_info(
                self.calculate_loss in calculate_loss_list,
                f"calculate_loss {self.top_k} in generate_config can only be in {calculate_loss_list},"
                " but it's {self.calculate_loss}",
            )
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, str(e))
