import numpy as np
import torch
from typing import List, Any

def remove_padding_eos(token_ids: torch.Tensor, eos_token_id: int) -> List[torch.Tensor]:
    # token_ids shape: [beam_width, max_length]
    out_token_ids = [tokens.cpu().numpy() for tokens in token_ids]
    out_token_ids = [tokens[tokens != eos_token_id].tolist() for tokens in out_token_ids]
    return [torch.IntTensor(x) for x in out_token_ids]

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

def get_stop_word_slice_list(stop_word_strs: List[str]) -> List[str]:
    result: List[str] = []
    for stop_word_str in stop_word_strs:
        result.append(stop_word_str)
        for i in range(1, len(stop_word_str)):
            result.append(stop_word_str[:-i])
    return result

def truncate_response_with_stop_words(response: str, stop_word_strs: List[str]):
    for stop_word in stop_word_strs:
        if stop_word and response.endswith(stop_word):
            response = response[:(-len(stop_word))]
    return response

# main
if __name__ == "__main__":
    # word_list = [[20490, 25]]
    # stop_list = to_word_list_format([word_list])
    # print(stop_list, stop_list.shape)
    
    stop_words = ['abc', '11123']
    print(get_stop_word_slice_list(stop_words))