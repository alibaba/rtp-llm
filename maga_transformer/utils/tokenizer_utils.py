from typing import List, Optional, Tuple, Union

from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

class DecodingState(object):
    last_input_id_index: int
    all_text: str

    prev_tokens: Optional[List[str]]
    prefix_offset: int = 0
    read_offset: int = 0

    def __init__(self):
        self.last_input_id_index = 0
        self.prev_tokens = None
        self.prefix_offset = 0
        self.read_offset = 0
        self.all_text = ""
    
    def update(self,
               last_input_id_index: int, 
               prev_tokens: Optional[List[str]],
               prefix_offset: int = 0, read_offset: int = 0):
        self.last_input_id_index = last_input_id_index
        self.prev_tokens = prev_tokens
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

    def __str__(self):
        return f"{self.__class__.__name__}(" + ", ".join([f"{k}={v!r}" for k, v in self.__dict__.items()]) + ")"

# Referenced from
# https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/tokenizer.py#L68
def _convert_tokens_to_string_with_added_encoders(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    output_tokens: List[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    sub_texts = []
    current_sub_text = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    legacy_added_tokens = set(tokenizer._added_tokens_encoder.keys()) - set(tokenizer.all_special_tokens) | {
    token for token in tokenizer.additional_special_tokens if tokenizer.convert_tokens_to_ids(token) >= tokenizer.vocab_size
    }

    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in legacy_added_tokens:
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    else:
        return "".join(sub_texts)
    
class IncrementDecodingUtils(object):
    @staticmethod
    def detokenize_incrementally(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        all_input_ids: List[int],
        state: DecodingState,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ):
        output_tokens, prefix_offset, read_offset = IncrementDecodingUtils._get_new_tokens(tokenizer, all_input_ids, state, skip_special_tokens)
        prefix_text, new_text = IncrementDecodingUtils._convert_token_to_string(tokenizer, output_tokens, prefix_offset, read_offset, skip_special_tokens, spaces_between_special_tokens)

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            new_text = new_text[len(prefix_text):]
            state.update(len(output_tokens), output_tokens, read_offset, len(output_tokens))
            return new_text
        else:
            state.update(len(output_tokens), output_tokens, prefix_offset, read_offset)
            return ""

    @staticmethod
    def _get_new_tokens(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                        all_input_ids: List[int],
                        state: DecodingState,
                        skip_special_tokens: bool = False) -> Tuple[List[str], int, int]:
        # first in 
        if state.prev_tokens is None:
            new_tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids, skip_special_tokens=skip_special_tokens)
            prefix_offset = 0
            read_offset = 0
            output_tokens = new_tokens
        else:
            new_tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids[state.last_input_id_index: ], skip_special_tokens=skip_special_tokens)
            prefix_offset = state.prefix_offset
            read_offset = state.read_offset
            output_tokens = state.prev_tokens + new_tokens
        return output_tokens, prefix_offset, read_offset
    
    @staticmethod
    def _convert_token_to_string(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        output_tokens: List[str],
        prefix_offset: int, 
        read_offset: int,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> Tuple[str, str]:
        if tokenizer.is_fast or not tokenizer.get_added_vocab():        
            prefix_text = tokenizer.convert_tokens_to_string(
                output_tokens[prefix_offset:read_offset])
            new_text = tokenizer.convert_tokens_to_string(
                output_tokens[prefix_offset:])
        else:
            prefix_text = _convert_tokens_to_string_with_added_encoders(
                tokenizer,
                output_tokens[prefix_offset:read_offset],
                skip_special_tokens=skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
            new_text = _convert_tokens_to_string_with_added_encoders(
                tokenizer,
                output_tokens[prefix_offset:],
                skip_special_tokens=skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
        return prefix_text, new_text
