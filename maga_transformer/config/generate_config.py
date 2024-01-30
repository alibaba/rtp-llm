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
    top_k: Union[List[int], int] = 0
    top_p: Union[List[float], float] = 0.95
    temperature: Union[List[float], float] = 1.0
    repetition_penalty: Union[List[float], float] = 1.0
    min_new_tokens: Union[List[int], int] = 0
    random_seed: Optional[Union[List[int], int]] = None
    top_p_decay: Optional[Union[List[float], float]] = None
    top_p_min: Optional[Union[List[float], float]] = None
    top_p_reset_ids: Optional[Union[List[int],int]] = None
    stop_words_list: List[List[int]] = []
    stop_words_str: List[str] = []
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
    return_input_ids: bool = False
    md5_value: str = ""
    custom_prop: str = "{}"

    # lora
    adapter_name: Optional[Union[str,List[str]]] = None

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

    # generate config for sample
    # TODO: do not gen generate config, gen sample config
    @staticmethod
    def merge_generate_config(configs: List['GenerateConfig']):
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
        
        calculate_loss_list = [0, 1, 2]
        assert self.calculate_loss in calculate_loss_list, \
                f"calculate_loss in generate_config can only be in {calculate_loss_list}, but it's {self.calculate_loss}"