from typing import Any, Dict, List, Optional, Union
from transformers.generation.stopping_criteria import StoppingCriteria

class StopWordIdsCriteria(StoppingCriteria):
    def __init__(self, stop_word_ids_list: List[List[int]]):
        self.stop_word_ids_list = stop_word_ids_list

    def __call__(self, token_ids: List[int], **kwargs: Any) -> bool:
        if len(self.stop_word_ids_list) == 0:
            return False
        for stop_word_ids in self.stop_word_ids_list:
            if len(token_ids) >= len(stop_word_ids) and token_ids[-len(stop_word_ids):] == stop_word_ids:
                return True
        return False

class StopWordStrsCriteria(StoppingCriteria):
    def __init__(self, stop_word_str_list: List[str], tokenizer: Any):
        self.stop_word_str_list = stop_word_str_list
        self.tokenizer = tokenizer

    def __call__(self, token_ids: List[int], **kwargs: Any) -> bool:
        if len(self.stop_word_str_list) == 0:
            return False
        output_str = self.tokenizer.decode(token_ids)
        if not isinstance(output_str, str):
            return False
        for stop_word_str in self.stop_word_str_list:
            if len(stop_word_str) == 0:
                continue
            if len(output_str) >= len(stop_word_str) and output_str[-len(stop_word_str):] == stop_word_str:
                return True

def create_stop_criteria_list(stop_word_ids: List[List[int]], stop_word_strs: List[str], tokenizer: Optional[Any]):
    lst: List[StoppingCriteria] = []
    if len(stop_word_ids) > 0:
        lst.append(StopWordIdsCriteria(stop_word_ids))
    if len(stop_word_strs) > 0 and tokenizer is not None:
        lst.append(StopWordStrsCriteria(stop_word_strs, tokenizer))
    return lst
