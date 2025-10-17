from typing import Any, List, Union

import numpy as np
import torch


def remove_padding_eos_with_numpy(token_ids: np.ndarray, eos_token_id: int) -> np.ndarray:
    # token_ids shape: [max_length]
    return token_ids[token_ids != eos_token_id]

def remove_padding_eos(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    return torch.IntTensor(remove_padding_eos_with_numpy(token_ids.cpu().numpy(), eos_token_id).tolist())


def remove_padding_eos_for_list(
    token_ids_list: List[torch.Tensor], eos_token_id: int
) -> List[torch.Tensor]:
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


"""
input:
words_list shape: [batch_size, word_num, word_token_size]
output:
[batch_size, 2, max_seq_length]
"""


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


def get_stop_word_slices(
    stop_word_list: List[Union[str, List[int]]]
) -> List[Union[str, List[int]]]:
    result: List[Union[str, List[int]]] = []
    for stop_word in stop_word_list:
        result.append(stop_word)
        for i in range(1, len(stop_word)):
            result.append(stop_word[:-i])
    return result


def is_truncated(
    input_str: str, trunc_strs: List[str], is_streaming: bool, slice: bool = False
):
    if len(input_str) > 0 and len(
        truncate_response_with_stop_words(input_str, trunc_strs, is_streaming, slice)
    ) != len(input_str):
        return True
    return False


def truncate_response_with_stop_words(
    response: str,
    stop_word_strs: List[str],
    is_streaming: bool = True,
    slice: bool = False,
):
    # 优化：预先过滤空的stop words，避免重复检查
    valid_stop_words = [sw for sw in stop_word_strs if sw]
    if not valid_stop_words or not response:
        return response

    if is_streaming:
        if slice:
            # 优化：对于slice模式，从最长的stop word开始检查
            for stop_word in sorted(valid_stop_words, key=len, reverse=True):
                if response.endswith(stop_word):
                    return response[:-len(stop_word)]
            return response
        else:
            # 优化：使用更高效的单次遍历查找最早位置
            first_pos = len(response)
            for stop_word in valid_stop_words:
                pos = response.find(stop_word)
                if pos == 0:
                    return ""  # 立即返回，避免后续处理
                if pos != -1 and pos < first_pos:
                    first_pos = pos
            return response[:first_pos]
    else:
        # 优化：非流式模式，找到最小位置
        min_index = len(response)
        for stop_word in valid_stop_words:
            index = response.find(stop_word)
            if index != -1 and index < min_index:
                min_index = index
                # 优化：如果找到了开头就立即返回
                if min_index == 0:
                    return ""
        return response[:min_index] if min_index != len(response) else response


def truncate_token_with_stop_word_id(tokens: List[int], stop_word_ids: List[int]):
    for stop_word_id in stop_word_ids:
        if stop_word_id and tokens[-len(stop_word_id) :] == stop_word_id:
            tokens = tokens[: (-len(stop_word_id))]
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

    stop_words = ["abc", "11123"]
    print(get_stop_word_slices(stop_words))
