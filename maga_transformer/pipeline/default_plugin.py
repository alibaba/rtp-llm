from typing import Any, List, Union, Iterator, Tuple, Callable, Optional, Dict
from transformers import PreTrainedTokenizerBase
from maga_transformer.utils.tokenizer_utils import DecodingState, IncrementDecodingUtils
from maga_transformer.config.generate_config import RequestFormat
from maga_transformer.pipeline.chatapi_format import encode_chatapi
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType

class DefaultPlugin(object):
    @staticmethod
    def modify_prompt_func(prompt: str, **kwargs: Any) -> str:
        return prompt

    @staticmethod
    def multimodal_modify_prompt_func(prompt: str, urls: List[str], mm_token: str, **kwargs: Any) -> Tuple[str, List[Any]]:
        return prompt, urls

    @staticmethod
    def modify_response_func(response: str, **kwargs: Any) -> str:
        return response

    @staticmethod
    def stop_generate_func(response: str, **kwargs: Any) -> bool:
        return False

    @staticmethod
    def process_encode_func(prompt: str, generate_config: Dict[str, Any], special_tokens: Any, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> List[int]:
        if len(prompt) == 0:
            raise FtRuntimeException(ExceptionType.EMPTY_PROMPT_ERROR, "prompt should have at least one token!")
        if generate_config['request_format'] == RequestFormat.CHAT_API:
            return encode_chatapi(prompt, special_tokens, tokenizer)
        if type(prompt) is not str:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "expect string prompt, actual: " + str(prompt))
        return tokenizer.encode(prompt)

    @staticmethod
    def tokenids_decode_func(tokens: List[int], tokenizer: PreTrainedTokenizerBase,
                             decoding_state: Optional[DecodingState] = None, return_incremental: bool = False, **kwargs: Any) -> str:
        if decoding_state is None:
            all_text = tokenizer.decode(tokens)
            # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
            while (len(all_text) > 0) and (u'\uFFFD' == all_text[-1]):
                all_text = all_text[:-1]
            return all_text, all_text

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            new_text = IncrementDecodingUtils.detokenize_incrementally(tokenizer, tokens, decoding_state)
            decoding_state.all_text += new_text
        else:
            all_text = tokenizer.decode(tokens)
            new_text = all_text[len(decoding_state.all_text): ]
            decoding_state.all_text = all_text

        return new_text if return_incremental == True else decoding_state.all_text, decoding_state.all_text