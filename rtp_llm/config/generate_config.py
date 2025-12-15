import copy
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.utils.check_util import *
from rtp_llm.utils.util import check_with_info


class RequestFormat:
    RAW = "raw"
    CHAT_API = "chatapi"


class RoleType(Enum):
    PDFUSION = 0
    PREFILL = 1
    DECODE = 2
    VIT = 3
    FRONTEND = 4


class RoleAddr(BaseModel):
    role: RoleType
    ip: str
    http_port: int
    grpc_port: int


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
    return_all_probs: bool = False
    return_softmax_probs: bool = False
    can_use_pd_separation: bool = True
    gen_timeline: bool = False
    profile_step: int = 3
    out_prefix: str = ""
    # for load balance
    role_addrs: List[RoleAddr] = []
    trace_id: str = ""

    # inter request id, from master
    inter_request_id: int = -1

    ignore_eos: bool = False
    skip_special_tokens: bool = False
    # lora
    adapter_name: Optional[Union[str, List[str]]] = None
    is_streaming: bool = False

    # multimodal preprocess
    resized_shape: Optional[List[int]] = None

    # whether add vision id in chat template; only use in frontend
    add_vision_id: bool = True

    # 是否允许tool_call专用的标签如<tool_call>作为content传出, 优化tool_call失败时的用户体验
    tool_call_message_extract_strategy: str = "default"  # default/skip_on_failure

    global_request_id: int = -1

    # 只有开启环境变量 REUSE_CACHE 时才生效
    reuse_cache: bool = True

    # 是否启用 memory block cache
    enable_memory_block_cache: bool = True

    enable_remote_cache: bool = True

    # close device cache manually, only use memory_cache or remote_cache(only for debug and test)
    enable_device_cache: bool = True

    # 是否开启同步写入
    sync_wait_write: bool = True

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

    def add_thinking_params(self, tokenizer):
        end_think_token_id = StaticConfig.generate_env_config.think_end_token_id
        self.end_think_token_ids = (
            [end_think_token_id] if end_think_token_id != -1 else []
        )
        if (
            bool(StaticConfig.generate_env_config.think_mode)
            and tokenizer
            and end_think_token_id == -1
        ):
            think_end_tag: str = StaticConfig.generate_env_config.think_end_tag.encode(
                "utf-8"
            ).decode("unicode_escape")
            tokenized_result: List[int] = tokenizer.encode(
                think_end_tag, add_special_tokens=False
            )
            self.end_think_token_ids = tokenized_result
        self.in_think_mode = (
            bool(StaticConfig.generate_env_config.think_mode)
            and len(self.end_think_token_ids) >= 0
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
