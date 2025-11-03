import logging
from typing import Any, List, Tuple, Union

import numpy as np
import torch


def remove_padding_eos_with_numpy(
    token_ids: np.ndarray, eos_token_id: int
) -> np.ndarray:
    # token_ids shape: [max_length]
    return token_ids[token_ids != eos_token_id]


def remove_padding_eos(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    return torch.IntTensor(
        remove_padding_eos_with_numpy(token_ids.cpu().numpy(), eos_token_id).tolist()
    )


import numpy as np


def batch_remove_padding_eos(
    batched_tokens: np.ndarray, eos_token_id: int
) -> List[np.ndarray]:
    eos_mask = batched_tokens == eos_token_id
    first_eos_indices = np.argmax(eos_mask, axis=1)
    seq_len = batched_tokens.shape[1]
    end_indices = np.where(np.any(eos_mask, axis=1), first_eos_indices, seq_len)
    truncated_tokens_list = [
        batched_tokens[i, : end_indices[i]] for i in range(len(batched_tokens))
    ]
    return truncated_tokens_list


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
    stop_word_list: List[Union[str, List[int]]],
) -> List[Union[str, List[int]]]:
    result: List[Union[str, List[int]]] = []
    for stop_word in stop_word_list:
        result.append(stop_word)
        for i in range(1, len(stop_word)):
            result.append(stop_word[:-i])
    return result


def is_truncated(
    input_str: str, trunc_strs: List[str], is_streaming: bool, slice: bool = False
) -> bool:
    """Check if input_str would be truncated by stop words.

    This function delegates to truncate_response_with_stop_words and checks
    if the result differs from the input.

    When using slice=True, pass pre-computed slices from get_stop_word_slices()
    as trunc_strs to avoid constructing temporary variables on each call.

    Args:
        input_str: The string to check
        trunc_strs: List of stop words (or pre-computed slices when slice=True)
        is_streaming: Whether in streaming mode (affects truncation behavior)
        slice: If True, check if response ends with any of trunc_strs
               If False, check if any trunc_str appears anywhere

    Returns:
        True if input_str would be truncated
    """
    if not input_str:
        return False
    return len(
        truncate_response_with_stop_words(input_str, trunc_strs, is_streaming, slice)
    ) != len(input_str)


def truncate_response_with_stop_words(
    response: str,
    stop_word_strs: List[str],
    is_streaming: bool = True,
    slice: bool = False,
):

    if is_streaming:
        first_pos = len(response)
        for stop_word in stop_word_strs:
            if stop_word:
                if slice:
                    # When slice=True, stop_word_strs should be pre-computed slices
                    # Just check endswith directly - no temporary construction
                    if response.endswith(stop_word):
                        response = response[: (-len(stop_word))]
                        break
                else:
                    pos = response.find(stop_word)
                    if pos == 0:
                        first_pos = 0
                        break
                    if pos != -1 and pos < first_pos:
                        first_pos = pos
        response = response[:first_pos]
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


def truncate_token_with_stop_word_id(tokens: List[int], stop_word_ids: List[List[int]]):
    for stop_word_id in stop_word_ids:
        if stop_word_id and np.array_equal(tokens[-len(stop_word_id) :], stop_word_id):
            tokens = tokens[: (-len(stop_word_id))]
            break
    return tokens


def match_stop_words(response: str, stop_word_strs: List[str]) -> Tuple[int, int]:
    """
    Finds the first occurrence of any stop word in the response string.

    Args:
        response (str): The string to search for stop words.
        stop_word_strs (List[str]): A list of stop word strings to search for.

    Returns:
        Tuple[int, int]: A tuple (position, length) where:
            - position is the index of the first matching stop word in the response,
              or -1 if no stop word is found.
            - length is the length of the matched stop word, or 0 if none is found.
    """
    min_idx = len(response)
    stop_len = 0
    for stop_word in stop_word_strs:
        if stop_word:
            stop_idx = response.find(stop_word)
            if stop_idx != -1 and stop_idx < min_idx:
                min_idx = stop_idx
                stop_len = len(stop_word)
    if min_idx == len(response):
        return -1, 0
    return min_idx, stop_len


# main
if __name__ == "__main__":
    # word_list = [[20490, 25]]
    # stop_list = to_word_list_format([word_list])
    # print(stop_list, stop_list.shape)

    stop_words = ["abc", "11123"]
    print(get_stop_word_slices(stop_words))
