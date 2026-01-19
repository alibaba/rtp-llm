from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rtp_llm.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.utils.word_util import (
    get_stop_word_slices,
    match_stop_words,
    remove_padding_eos_with_numpy,
    truncate_response_with_stop_words,
    truncate_token_with_stop_word_id,
)


class TokenProcessor:
    def __init__(self, tokenizer, special_tokens):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

    def __call__(self, prompt: str) -> Any:
        return self.tokenizer(
            prompt, return_offsets_mapping=True, return_attention_mask=False
        )

    def decode(self, token_id: List[int]) -> str:
        return self.tokenizer.decode(token_id)

    def encode(self, prompt: Union[str, bytes]) -> torch.tensor:
        try:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8", errors="ignore")
            th = self.tokenizer.encode(prompt)
            return th
        except Exception as e:
            raise e


class TokenProcessorPerStream:
    decoding_states: List[DecodingState]
    ouput_tokens_list: List[npt.NDArray[np.int32]]
    token_buffers: List[str]
    has_num_beams: bool

    def __init__(self, has_num_beams: bool, size: int, token_processor: TokenProcessor):
        self.has_num_beams = has_num_beams
        self.tokenizer = token_processor.tokenizer
        self.special_tokens = token_processor.special_tokens
        if not has_num_beams:
            self.decoding_states = [DecodingState() for _ in range(size)]
        else:
            # num_beams不等于1的情况下，不能进行增量decode，因为过去的token id会变化
            self.decoding_states = [None] * size
        self.token_buffers = [""] * size
        self.ouput_tokens_list = [np.empty(0, dtype=np.int32) for _ in range(size)]

    def decode_tokens(
        self,
        i: int,
        tokens: npt.NDArray[np.int32],
        finished: bool,
        print_stop_words: bool,
        stop_word_str_list: List[str],
        stop_word_ids: List[List[int]],
        return_incremental: bool = False,
    ):
        if not self.has_num_beams:
            if len(self.ouput_tokens_list[i]) == 0:
                self.ouput_tokens_list[i] = tokens
            else:
                self.ouput_tokens_list[i] = np.concatenate(
                    (self.ouput_tokens_list[i], tokens), axis=1
                )
            tokens = self.ouput_tokens_list[i]
        tokens = remove_padding_eos_with_numpy(
            tokens, self.special_tokens.eos_token_id
        ).tolist()
        output_len = len(tokens)
        stop_word_id_slices = get_stop_word_slices(stop_word_ids)
        tokens = self.process_stop_id(
            print_stop_words, finished, tokens, stop_word_ids, stop_word_id_slices
        )
        text, all_text = self.tokenids_decode(
            tokens, self.decoding_states[i], return_incremental
        )
        text, self.token_buffers[i] = self.process_stop_str(
            finished,
            return_incremental,
            print_stop_words,
            text,
            all_text,
            stop_word_str_list,
            self.token_buffers[i],
        )
        return output_len, text

    def batch_decode_tokens(
        self,
        batch_token_ids: List[npt.NDArray[np.int32]],
        batch_finished: List[bool],
        print_stop_words: bool,
        stop_word_str_list: List[str],
        stop_word_ids: List[List[int]],
        return_incremental: bool = False,
    ):
        batch_output_lens = []
        final_texts = []
        texts_to_process = []
        all_texts_for_stop_words = []
        tokens_to_decode_batch = []
        stop_word_id_slices = get_stop_word_slices(stop_word_ids)
        for i, tokens in enumerate(batch_token_ids):
            if not self.has_num_beams:
                if len(self.ouput_tokens_list[i]) == 0:
                    self.ouput_tokens_list[i] = tokens
                else:
                    self.ouput_tokens_list[i] = np.concatenate(
                        (self.ouput_tokens_list[i], tokens), axis=1
                    )
                tokens = self.ouput_tokens_list[i]
            tokens = remove_padding_eos_with_numpy(
                tokens, self.special_tokens.eos_token_id
            ).tolist()

            batch_output_lens.append(len(tokens))

            tokens = self.process_stop_id(
                print_stop_words,
                batch_finished[i],
                tokens,
                stop_word_ids,
                stop_word_id_slices,
            )

            tokens_to_decode_batch.append(tokens)

        if self.has_num_beams:
            all_texts_batch = self.tokenizer.batch_decode(tokens_to_decode_batch)
            texts_to_process = all_texts_batch
            all_texts_for_stop_words = all_texts_batch
        else:
            for i, tokens in enumerate(batch_token_ids):
                decoding_state = self.decoding_states[i]
                new_text, all_text = self.tokenids_decode(
                    tokens, decoding_state, return_incremental=True
                )
                new_text = all_text[len(decoding_state.all_text) :]
            text_for_stop_processing = new_text if return_incremental else all_text
            texts_to_process.append(text_for_stop_processing)
            all_texts_for_stop_words.append(all_text)

        for i in range(len(batch_token_ids)):
            final_text, self.token_buffers[i] = self.process_stop_str(
                batch_finished[i],
                return_incremental,
                print_stop_words,
                texts_to_process[i],
                all_texts_for_stop_words[i],
                stop_word_str_list,
                self.token_buffers[i],
            )
            final_texts.append(final_text)

        return batch_output_lens, final_texts

    def process_stop_id(
        self,
        print_stop_words: bool,
        finished: bool,
        tokens: List[int],
        stop_word_ids: List[List[int]],
        stop_word_id_slices=List[Union[str, List[int]]],
    ) -> List[int]:

        if not print_stop_words:
            if not finished:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_id_slices)
            else:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_ids)
        return tokens

    def process_stop_str(
        self,
        finished: bool,
        return_incremental: bool,
        print_stop_words: bool,
        text: str,
        all_text: str,
        stop_word_str_list: List[str],
        token_buffer: str,
    ):
        if return_incremental:
            text = token_buffer + text

        if stop_word_str_list:
            stop_idx, stop_len = match_stop_words(text, stop_word_str_list)
            if stop_idx != -1:
                if not print_stop_words:
                    text = text[:stop_idx]
                else:
                    text = text[: stop_idx + stop_len]
                token_buffer = ""
                finished = True

        if finished:
            return text, token_buffer

        stop_word_str_slices = get_stop_word_slices(stop_word_str_list)

        if return_incremental or not print_stop_words:
            trunc_text = truncate_response_with_stop_words(
                text, stop_word_str_slices, True, True
            )
            if return_incremental:
                token_buffer = text[len(trunc_text) :]
            text = trunc_text
        return text, token_buffer

    def tokenids_decode(
        self,
        tokens: List[int],
        decoding_state: Optional[DecodingState] = None,
        return_incremental: bool = False,
    ):

        if decoding_state is None:
            all_text = self.tokenizer.decode(tokens)
            return all_text, all_text

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            new_text = IncrementDecodingUtils.detokenize_incrementally(
                self.tokenizer, tokens, decoding_state
            )
            decoding_state.all_text += new_text
        else:
            all_text = self.tokenizer.decode(tokens)
            new_text = all_text[len(decoding_state.all_text) :]
            decoding_state.all_text = all_text

        return (
            new_text if return_incremental == True else decoding_state.all_text
        ), decoding_state.all_text
