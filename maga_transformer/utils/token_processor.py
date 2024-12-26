import os
import asyncio
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Optional, List

from maga_transformer.utils.tokenizer_utils import DecodingState, IncrementDecodingUtils
from maga_transformer.utils.word_util import remove_padding_eos, get_stop_word_slices, \
            truncate_response_with_stop_words, truncate_token_with_stop_word_id, match_stop_words

class TokenProcessor:
    def __init__(self, tokenizer, special_tokens):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

    def __call__(self, prompt: str) -> Any:
        return self.tokenizer(prompt, return_offsets_mapping=True, return_attention_mask=False)

    def decode(self, token_id: List[int]) -> str:
        return self.tokenizer.decode(token_id)

    def encode(self, prompt: str) -> torch.tensor:
        th = self.tokenizer.encode(prompt)
        return th

class TokenProcessorPerStream:
    decoding_states: List[DecodingState]
    ouput_tokens_list: List[torch.Tensor]
    token_buffers: List[str]
    num_beams: int

    def __init__(self, num_beams: int, size: int, token_processor: TokenProcessor):
        self.num_beams = num_beams;
        self.tokenizer = token_processor.tokenizer
        self.special_tokens = token_processor.special_tokens
        if num_beams == 1:
            self.decoding_states = [DecodingState() for _ in range(size)]
        else:
            # num_beams不等于1的情况下，不能进行增量decode，因为过去的token id会变化
            self.decoding_states = [None] * size
        self.token_buffers = [""] * size
        self.ouput_tokens_list = [torch.empty(0, dtype=torch.int32) for _ in range(size)]

    def decode_tokens(self,
                      i: int,
                      tokens: torch.Tensor,
                      finished: bool,
                      print_stop_words: bool,
                      stop_word_str_list: List[str],
                      stop_word_ids: List[List[int]],
                      return_incremental: bool = False):
        if self.num_beams == 1:
            self.ouput_tokens_list[i] = torch.cat((self.ouput_tokens_list[i], tokens), dim=1)
        tokens = self.ouput_tokens_list[i]
        tokens = remove_padding_eos(tokens, self.special_tokens.eos_token_id)
        output_len = tokens.nelement()
        tokens = self.process_stop_id(print_stop_words, finished, tokens.tolist(), stop_word_ids)
        text, all_text = self.tokenids_decode(tokens, self.decoding_states[i], return_incremental)
        text, self.token_buffers[i] = self.process_stop_str(finished,
                                                            return_incremental,
                                                            print_stop_words,
                                                            text, all_text,
                                                            stop_word_str_list,
                                                            self.token_buffers[i])
        return output_len, text

    def process_stop_id(self,
                        print_stop_words: bool,
                        finished: bool,
                        tokens: List[int],
                        stop_word_ids: List[List[int]]) -> List[int]:

        stop_word_id_slices = get_stop_word_slices(stop_word_ids)
        if not print_stop_words:
            if not finished:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_id_slices)
            else:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_ids)
        return tokens

    def process_stop_str(self,
                         finished: bool,
                         return_incremental: bool,
                         print_stop_words: bool,
                         text: str,
                         all_text: str,
                         stop_word_str_list: List[str],
                         token_buffer: str):

        if stop_word_str_list and not finished and match_stop_words(all_text, stop_word_str_list):
            finished = True

        stop_word_str_slices = get_stop_word_slices(stop_word_str_list)

        if not print_stop_words:
            if not return_incremental:
                if not finished:
                    text = truncate_response_with_stop_words(text, stop_word_str_slices)
                else:
                    text = truncate_response_with_stop_words(text, stop_word_str_list)
            else:
                if not finished:
                    text = token_buffer + text
                    trunc_text = truncate_response_with_stop_words(text, stop_word_str_slices)
                    token_buffer = text[len(trunc_text):]
                    text = trunc_text
                else:
                    text = truncate_response_with_stop_words(token_buffer + text, stop_word_str_list)
        return text, token_buffer

    def tokenids_decode(self,
                        tokens: List[int],
                        decoding_state: Optional[DecodingState] = None,
                        return_incremental: bool = False):

        if decoding_state is None:
            all_text = self.tokenizer.decode(tokens)
            return all_text, all_text

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            new_text = IncrementDecodingUtils.detokenize_incrementally(self.tokenizer, tokens, decoding_state)
            decoding_state.all_text += new_text
        else:
            all_text = self.tokenizer.decode(tokens)
            new_text = all_text[len(decoding_state.all_text):]
            decoding_state.all_text = all_text

        return new_text if return_incremental == True else decoding_state.all_text, decoding_state.all_text

