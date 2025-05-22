import numpy as np
import torch
from typing import List, Union, Any

def remove_padding_eos(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    # token_ids shape: [max_length]
    out_token_ids = token_ids.cpu().numpy()
    out_token_ids = out_token_ids[out_token_ids != eos_token_id].tolist()
    return torch.IntTensor(out_token_ids)

def remove_padding_eos_for_list(token_ids_list: List[torch.Tensor], eos_token_id: int) -> List[torch.Tensor]:
    # token_ids shape: [sub batch of stream, max_length]
    return [remove_padding_eos(token_ids, eos_token_id) for token_ids in token_ids_list]

def get_list_dim(origin: Any) -> int:
    def _get_dim_internal(x: Any) -> int:
        if not isinstance(x, list):
            return 0
        if len(x) == 0:
            return 1
        else:
            return _get_dim_internal(x[0]) + 1
    return _get_dim_internal(origin)

'''
input:
words_list shape: [batch_size, word_num, word_token_size]
output:
[batch_size, 2, max_seq_length]
'''
def to_word_list_format(words_list: List[List[List[int]]]):
    flat_ids = []
    offsets = []

    for words in words_list:
        item_flat_ids = []
        item_offsets = []

        for ids in words:
            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    result = np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
    # if result.shape[0] == 1:
    #   result = result.squeeze(0)
    return np.ascontiguousarray(result)

def get_stop_word_slices(stop_word_list: List[Union[str, List[int]]]) -> List[Union[str, List[int]]]:
    result: List[Union[str, List[int]]] = []
    for stop_word in stop_word_list:
        result.append(stop_word)
        for i in range(1, len(stop_word)):
            result.append(stop_word[:-i])
    return result

def is_truncated(input_str: str, trunc_strs: List[str], is_streaming: bool):
    if len(input_str) > 0 and len(truncate_response_with_stop_words(input_str, trunc_strs, is_streaming)) != len(input_str):
        return True
    return False

def truncate_response_with_stop_words(response: str, stop_word_strs: List[str], is_streaming: bool = True):
    if is_streaming:
        for stop_word in stop_word_strs:
            if stop_word and response.endswith(stop_word):
                response = response[:(-len(stop_word))]
                break
    else:
        min_index = len(response)
        for stop_word in stop_word_strs:
            if stop_word:
                index = response.find(stop_word)
                if index != -1 and index < min_index:
                    min_index = index
        if min_index != len(response):
            response = response[:min_index]
    return response

def truncate_token_with_stop_word_id(tokens: List[int], stop_word_ids: List[int]):
    for stop_word_id in stop_word_ids:
        if stop_word_id and tokens[-len(stop_word_id):] == stop_word_id:
            tokens = tokens[:(-len(stop_word_id))]
            break
    return tokens

def match_stop_words(response: str, stop_word_strs: List[str]) -> bool:
    for stop_word in stop_word_strs:
        if stop_word and response.endswith(stop_word):
            return True
    return False

# main
if __name__ == "__main__":
    # word_list = [[20490, 25]]
    # stop_list = to_word_list_format([word_list])
    # print(stop_list, stop_list.shape)
    
    stop_words = ['abc', '11123']
    print(get_stop_word_slices(stop_words))
