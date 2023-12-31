from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
import torch
import logging
from dataclasses import dataclass, field

from transformers import PreTrainedTokenizer

from maga_transformer.models.base_model import BaseTokenizer, GenerateOutput
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, UsageInfo, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason

@dataclass
class ProcessedOutput:
    output_str: str
    output_token_length: int
    finish_reason: Optional[FinisheReason]

@dataclass
class StreamResponseObject:
    choices: List[ChatCompletionResponseStreamChoice] = field(default_factory=list)
    usage: Optional[UsageInfo] = None

@dataclass
class RendererParams:
    max_seq_len: int
    eos_token_id: int
    stop_word_ids_list: List[List[int]]

class CustomChatRenderer():
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, BaseTokenizer],
                 renderer_params: RendererParams,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = renderer_params.max_seq_len
        self.eos_token_id = renderer_params.eos_token_id
        self.stop_word_ids_list = renderer_params.stop_word_ids_list
        self.stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in self.stop_word_ids_list
        ]
        self.extra_stop_word_ids_list: List[List[int]] = []

    def get_extra_stop_word_ids_list(self) -> List[List[int]]:
        return self.extra_stop_word_ids_list

    def render_chat(self, request: ChatCompletionRequest) -> List[int]:
        raise NotImplementedError

    async def render_response_stream(
            self,
            output_generator: AsyncGenerator[GenerateOutput, None],
            request: ChatCompletionRequest,
            input_token_length: int,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        raise NotImplementedError

    def _check_finish_reason(self, token_ids: List[int]) -> Optional[FinisheReason]:
        if len(token_ids) >= self.max_seq_len:
            return FinisheReason.length
        if token_ids[-1] == self.eos_token_id:
            return FinisheReason.stop
        for stop_word_ids in self.stop_word_ids_list:
            if (len(token_ids) >= len(stop_word_ids)) and (token_ids[-len(stop_word_ids):] == stop_word_ids):
                return FinisheReason.stop
        return None

    def _remove_stop_word_ids(self, output_ids: List[int]) -> List[int]:
        for stop_word_ids in self.stop_word_ids_list:
            for i in range(1, len(stop_word_ids)):
                if output_ids[-i:] == stop_word_ids[:i]:
                    output_ids = output_ids[:-i]
                    break
        return output_ids

    def _process_output_ids_tensor(
            self, input_length, output_ids_tensor: torch.Tensor, finished: bool = False
    ) -> ProcessedOutput:
        output_ids_tensor = output_ids_tensor.cpu().reshape([-1])
        # TODO(wangyin): This slicing shouldn't be done here.
        # model should return output length, ids should be sliced with output length.
        output_ids = output_ids_tensor[output_ids_tensor != self.eos_token_id].tolist()
        finish_reason = self._check_finish_reason(output_ids) if finished else None

        output_ids = output_ids[input_length:]
        output_length = len(output_ids)
        output_ids = self._remove_stop_word_ids(output_ids)
        output_str = self.tokenizer.decode(output_ids)
        output_str = output_str.strip(u'\uFFFD')

        for stop_word in self.stop_words_list:
            output_str = output_str.replace(stop_word, "")
        return ProcessedOutput(output_str, output_length, finish_reason)

