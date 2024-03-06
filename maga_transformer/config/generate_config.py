import copy
import hashlib
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from pydantic import BaseModel

class RequestFormat:
    RAW = 'raw'
    CHAT_API = 'chatapi'

class GenerateConfig(BaseModel):
    max_new_tokens: int = 1000
    num_beams: int = 1
    # 0 mean not use num_return_sequences
    num_return_sequences: int = 0
    top_k: Union[List[int], int] = 0
    top_p: Union[List[float], float] = 0.95
    temperature: Union[List[float], float] = 1.0
    repetition_penalty: Union[List[float], float] = 1.0
    min_new_tokens: Union[List[int], int] = 0
    random_seed: Optional[Union[List[int], int]] = None
    top_p_decay: Optional[Union[List[float], float]] = None
    top_p_min: Optional[Union[List[float], float]] = None
    top_p_reset_ids: Optional[Union[List[int],int]] = None
    stop_words_str: List[str] = []
    stop_words_list: List[List[int]] = []
    bad_words_list: Optional[Union[List[List[List[int]]], List[List[int]]]] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    using_hf_sampling: bool = False
    print_stop_words: bool = False
    timeout_ms: int = -1
    chat_id: Optional[str] = None
    task_id: Optional[int] = None
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

    # lora
    adapter_name: Optional[Union[str, List[str]]] = None

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

    # generate config for sample
    # TODO: do not gen generate config, gen sample config
    @staticmethod
    def merge_generate_config(configs: List['GenerateConfig']) -> 'GenerateConfig':
        top_k: List[int] = []
        top_p: List[float] = []
        min_new_tokens: List[int] = []
        repetition_penalty: List[float] = []
        for config in configs:
            top_k.append(config.top_k)
            top_p.append(config.top_p)
            min_new_tokens.append(config.min_new_tokens)
            repetition_penalty.append(config.repetition_penalty)

        res = GenerateConfig(
            top_k=top_k,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            eos_token_id=configs[0].eos_token_id,
            num_beams=configs[0].num_beams,
        )
        res.gen_hash_value()
        return res

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
        # 如果同时在外面和里面都有设置采样参数，选择使用外面的
        # 这里假设外部传进来的stop_word_list和stop_word_str都不包含batch维度
        self.stop_words_list += special_tokens.stop_words_list
        self.stop_words_str += special_tokens.stop_words_str

    def validate(self):
        try:
            assert isinstance(self.top_k, int) or \
             (isinstance(self.top_k, list) and all([isinstance(i, int) for i in self.top_k]))
            assert isinstance(self.top_p, (float, int)) or \
             (isinstance(self.top_p, list) and all([isinstance(i, (int, float)) for i in self.top_p]))
            assert isinstance(self.min_new_tokens, int) or \
             (isinstance(self.min_new_tokens, list) and all([isinstance(i, int) for i in self.min_new_tokens]))
            assert isinstance(self.repetition_penalty, (float, int)) or \
             (isinstance(self.repetition_penalty, list) and all([isinstance(i, (int, float)) for i in self.repetition_penalty]))
             
            calculate_loss_list = [0, 1, 2]
            assert self.calculate_loss in calculate_loss_list, \
                f"calculate_loss in generate_config can only be in {calculate_loss_list}, but it's {self.calculate_loss}"
        except:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "wrong data type in generate config")
