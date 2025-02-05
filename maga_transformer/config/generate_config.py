import copy
import hashlib
from pydantic import BaseModel
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union
from maga_transformer.utils.util import check_with_info
from maga_transformer.utils.check_util import *
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType

class RequestFormat:
    RAW = 'raw'
    CHAT_API = 'chatapi'

class GenerateConfig(BaseModel):
    max_new_tokens: int = 1000
    # only for qwen agent fncall check max input tokens
    max_input_tokens: int = 32000
    num_beams: int = 1
    # 0 mean not use num_return_sequences,
    # whether to enable num_return_sequences, the output format of the results is inconsistent.
    num_return_sequences: int = 0
    top_k: Union[List[int], int] = 0
    top_p: Union[List[float], float] = 0.95
    temperature: Union[List[float], float] = 1.0
    repetition_penalty: Union[List[float], float] = 1.0
    min_new_tokens: Union[List[int], int] = 0
    no_repeat_ngram_size: Optional[Union[List[int], int]] = None
    random_seed: Optional[Union[List[int], int]] = None
    top_p_decay: Optional[Union[List[float], float]] = None
    top_p_min: Optional[Union[List[float], float]] = None
    top_p_reset_ids: Optional[Union[List[int],int]] = None
    stop_words_str: List[str] = []
    stop_words_list: List[List[int]] = []
    bad_words_list: Optional[Union[List[List[List[int]]], List[List[int]]]] = None
    eos_token_id: Optional[Union[List[int],int]] = None
    pad_token_id: Optional[Union[List[int],int]] = None
    bos_token_id: Optional[Union[List[int],int]] = None
    using_hf_sampling: bool = False
    print_stop_words: bool = False
    timeout_ms: Optional[int] = -1
    chat_id: Optional[str] = None
    task_id: Optional[Union[str, int]] = None
    request_format: str = RequestFormat.RAW
    # calculate_loss style: 0 for not calculate; 1 for sum; 2 for each token
    calculate_loss: int = 0
    return_logits: bool = False
    return_incremental: bool = False
    return_hidden_states: bool = False
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
    return_all_probs: bool = False
    return_softmax_probs: bool = False
    can_use_pd_separation: bool = True

    # lora
    adapter_name: Optional[Union[str, List[str]]] = None
    is_streaming: bool = False

    def gen_hash_value(self):
        cp = copy.copy(self)
        cp.max_new_tokens = 0
        cp.chat_id = None
        cp.random_seed = None
        cp.md5_value = ""
        cp.timeout_ms = -1
        self.md5_value = hashlib.md5(cp.__str__().encode()).hexdigest()

    def is_same(self, config: 'GenerateConfig') -> bool:
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
    def create_generate_config(generate_config: Dict[str, Any], **kwargs: Any) -> 'GenerateConfig':
        generate_config.update(kwargs)
        try:
            config = GenerateConfig(**generate_config)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_GENERATE_CONFIG_FORMAT, f"generate_config validate failed: {str(e)}")
        config.validate()
        return config

    def convert_select_tokens(self, vocab_size, tokenizer):
        for token_str in self.select_tokens_str:
            self.select_tokens_id += tokenizer.encode(token_str)
        if not all(token_id < vocab_size and token_id >= 0 for token_id in self.select_tokens_id):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                                    f"token_id in select_tokens_id {self.select_tokens_id} should be less than vocab_size {vocab_size}, and shoud not be negative")

    def add_special_tokens(self, special_tokens: Any):
        # 这里假设外部传进来的stop_word_list和stop_word_str都不包含batch维度
        self.stop_words_list += special_tokens.stop_words_id_list
        self.stop_words_str += special_tokens.stop_words_str_list

    def validate(self):
        try:
            check_with_info(is_union_positive_integer(self.top_k), \
                f"top_k {self.top_k} is wrong data type")
            check_with_info(is_union_positive_number(self.top_p), \
                f"top_p {self.top_p} is wrong data type")
            check_with_info(is_union_positive_integer(self.min_new_tokens), \
                f"min_new_tokens {self.min_new_tokens} is wrong data type")
            check_with_info(is_union_positive_number(self.repetition_penalty), \
                f"repetition_penalty {self.repetition_penalty} is wrong data type")
            check_with_info(is_positive_integer(self.max_new_tokens), \
                f"max_new_tokens {self.max_new_tokens} is wrong data type")
            check_with_info(is_positive_integer(self.num_beams), \
                f"num_beams {self.num_beams} is wrong data type")
            check_with_info(is_positive_integer(self.num_return_sequences), \
                f"num_return_sequences {self.num_return_sequences} is wrong data type")
            check_with_info(is_union_positive_number(self.temperature), \
                f"temperature {self.temperature} is wrong data type")
            check_with_info(check_optional(is_union_positive_integer, self.no_repeat_ngram_size), \
                f"no_repeat_ngram_size {self.no_repeat_ngram_size} is wrong data type")
            check_with_info(check_optional(is_union_positive_integer, self.random_seed), \
                f"random_seed {self.random_seed} is wrong data type")
            check_with_info(check_optional(is_union_positive_number, self.top_p_decay),
                f"top_p_decay {self.top_p_decay} is wrong data type")
            check_with_info(check_optional(is_union_positive_number, self.top_p_min), \
                f"top_p_min {self.top_p_min} is wrong data type")
            check_with_info(check_optional(is_union_positive_integer, self.top_p_reset_ids), \
                f"top_p_reset_ids {self.top_p_reset_ids} is wrong data type")
            check_with_info(check_optional(is_union_positive_integer, self.eos_token_id), \
                f"eos_token_id {self.eos_token_id} is wrong data type")
            check_with_info(check_optional(is_union_positive_integer, self.pad_token_id), \
                f"pad_token_id {self.pad_token_id} is wrong data type")
            check_with_info(check_optional(is_union_positive_integer, self.bos_token_id), \
                f"bos_token_id {self.bos_token_id} is wrong data type")
            check_with_info(is_list_positive_integer_list(self.stop_words_list), \
                f"stop_words_list {self.stop_words_list} is wrong data type")
            check_with_info(is_union_positive_integer(self.sp_advice_prompt_token_ids),
                f"sp_advice_prompt_token_ids {self.sp_advice_prompt_token_ids} is wrong data type")
            calculate_loss_list = [0, 1, 2]
            check_with_info(self.calculate_loss in calculate_loss_list, \
                f"calculate_loss {self.top_k} in generate_config can only be in {calculate_loss_list}," \
                " but it's {self.calculate_loss}")
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, str(e))
