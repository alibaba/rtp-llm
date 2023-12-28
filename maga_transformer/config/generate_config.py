import copy
import hashlib
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union
from transformers.generation.stopping_criteria import StoppingCriteria
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType

class RequestFormat:
    RAW = 'raw'
    CHAT_API = 'chatapi'

class StopWordIdsCriteria(StoppingCriteria):
    def __init__(self, stop_word_ids_list: List[List[int]]):
        self.stop_word_ids_list = stop_word_ids_list

    def __call__(self, token_ids: List[int], input_length: int, **kwargs: Any) -> bool:
        if len(self.stop_word_ids_list) == 0:
            return False
        output_tokens_ids = token_ids[input_length: ]
        for stop_word_ids in self.stop_word_ids_list:
            if len(output_tokens_ids) >= len(stop_word_ids) and output_tokens_ids[-len(stop_word_ids):] == stop_word_ids:
                return True
        return False

class StopWordStrsCriteria(StoppingCriteria):
    def __init__(self, stop_word_str_list: List[str], decode_func: Any, tokenizer: Any):
        self.stop_word_str_list = stop_word_str_list
        self.decode_func = decode_func
        self.tokenizer = tokenizer

    def __call__(self, token_ids: List[int], input_length: int, **kwargs: Any) -> bool:
        if len(self.stop_word_str_list) == 0:
            return False
        output_str = self.decode_func(0, token_ids[input_length:], tokenizer=self.tokenizer)
        if not isinstance(output_str, str):
            return False
        for stop_word_str in self.stop_word_str_list:

            if len(output_str) >= len(stop_word_str) and output_str[-len(stop_word_str):] == stop_word_str:
                return True

@dataclass
class GenerateConfig:
    max_new_tokens: int = 1000
    num_beams: int = 1
    top_k: Union[List[int], int] = 0
    top_p: Union[List[float], float] = 0.95
    temperature: Union[List[float], float] = 1.0
    repetition_penalty: Union[List[float], float] = 1.0
    min_new_tokens: Union[List[int], int] = 0
    random_seed: Optional[Union[List[int], int]] = None
    top_p_decay: Optional[Union[List[float], float]] = None
    top_p_min: Optional[Union[List[float], float]] = None
    top_p_reset_ids: Optional[Union[List[int],int]] = None
    stop_words_list: List[List[int]] = field(default_factory=list)
    stop_words_str: List[str] = field(default_factory=list)
    bad_words_list: Optional[Union[List[List[List[int]]], List[List[int]]]] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    using_hf_sampling: bool = False
    print_stop_words: bool = False
    timeout_ms: int = -1
    criteria_list: List[StoppingCriteria] = field(default_factory=list)
    chat_id: Optional[str] = None
    task_id: Optional[int] = None
    request_format: str = RequestFormat.RAW
    # calculate_loss style: 0 for not calculate; 1 for sum; 2 for each token
    calculate_loss: int = 0
    return_logits: bool = False
    return_incremental: bool = False
    md5_value: str = ""
    custom_prop: str = "{}"

    # lora
    adapter_name: Optional[List[str]] = None

    def gen_hash_value(self):
        cp = copy.copy(self)
        cp.max_new_tokens = 0
        cp.chat_id = None
        cp.random_seed = None
        cp.md5_value = ""
        cp.criteria_list = []
        cp.timeout_ms = -1
        self.md5_value = hashlib.md5(cp.__str__().encode()).hexdigest()

    def is_same(self, config: 'GenerateConfig') -> bool:
        return self.md5_value == config.md5_value

    def update(self, new: Dict[str, Any]):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        # avoid use as_dict since it deepcopy everything include tokenizer
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def create_stop_word_criteria(self, decode_func: Any, tokenizer: Any) -> List[StoppingCriteria]:
        self.criteria_list.append(StopWordIdsCriteria(self.stop_words_list))
        self.criteria_list.append(StopWordStrsCriteria(self.stop_words_str, decode_func, tokenizer))

    def check_data_type(self):
        try:
            assert isinstance(self.top_k, int) or \
             (isinstance(self.top_k, list) and all([isinstance(i, int) for i in self.top_k]))
            assert isinstance(self.top_p, (float, int)) or \
             (isinstance(self.top_p, list) and all([isinstance(i, (int, float)) for i in self.top_p]))
            assert isinstance(self.min_new_tokens, int) or \
             (isinstance(self.min_new_tokens, list) and all([isinstance(i, int) for i in self.min_new_tokens]))
            assert isinstance(self.repetition_penalty, (float, int)) or \
             (isinstance(self.repetition_penalty, list) and all([isinstance(i, (int, float)) for i in self.repetition_penalty]))
        except:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "wrong data type in generate config")
