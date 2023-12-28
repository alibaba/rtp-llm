from fastapi import Request
import torch
from typing import Union, Optional, List, Dict, Generator, Coroutine, AsyncGenerator, Any, Iterator
import time
import logging
from dataclasses import dataclass

from transformers import PreTrainedTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria

from maga_transformer.models.base_model import BaseModel, BaseTokenizer, GenerateOutput, GenerateResponse
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.openai.api_datatype import ModelCard, ModelList, ChatMessage, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, UsageInfo, FinisheReason, \
    DeltaMessage, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse
from maga_transformer.openai.template_renderer import TemplateRenderer
from maga_transformer.config.generate_config import GenerateConfig, StopWordIdsCriteria

@dataclass
class ProcessedOutput:
    output_str: str
    output_token_length: int
    finish_reason: Optional[FinisheReason]

# TODO(wangyin): ADD TEST !!!
class OpenaiEndopoint():
    def __init__(self, model: Union[AsyncModel, BaseModel]):
        self.model = model
        self.max_seq_len = self.model.config.max_seq_len

        tokenizer = self.model.tokenizer
        if (tokenizer == None):
            raise AttributeError(f"model [{model}] has no tokenizer!")
        self.tokenizer: Union[PreTrainedTokenizer, BaseTokenizer] = tokenizer
        self.template_renderer = TemplateRenderer(self.tokenizer)

        self.eos_token_id = None
        if isinstance(tokenizer, PreTrainedTokenizer):
            self.eos_token_id = tokenizer.eos_token_id
        if (self.eos_token_id == None):
            self.eos_token_id = self.model.config.special_tokens.eos_token_id

        self.stop_word_ids_list = self.model.config.special_tokens.stop_words_list
        self.stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in self.stop_word_ids_list
        ]

    async def list_models(self):
        global model_args
        model_card = ModelCard(id=self.model.__class__.__name__)
        return ModelList(data=[model_card])

    def _extract_generation_config(self, request: ChatCompletionRequest) -> GenerateConfig:
        # TODO(wangyin): implement this
        config = GenerateConfig()
        if request.temperature != None:
            config.temperature = request.temperature
        if request.top_p != None:
            config.top_p = request.top_p
        if request.max_tokens != None:
            config.max_new_tokens = request.max_tokens
        config.criteria_list = [
            StopWordIdsCriteria(self.stop_word_ids_list)
        ]
        return config

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

        for stop_word in self.stop_words_list:
            output_str = output_str.replace(stop_word, "")
        return ProcessedOutput(output_str, output_length, finish_reason)

    async def _complete_non_stream_response(
            self, input_length: int, output_generator: AsyncGenerator[GenerateOutput, None]
    ) -> ChatCompletionResponse:
        model_output: Optional[GenerateOutput] = None
        async for output in output_generator:
            model_output = output

        if model_output == None:
            raise RuntimeError("model had no output!")

        processed_output = self._process_output_ids_tensor(input_length, model_output.output_ids, True)

        message = ChatMessage(
            role=RoleEnum.assistant,
            content=processed_output.output_str
        )
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=processed_output.finish_reason
        )
        usage = UsageInfo(
            prompt_tokens=input_length,
            total_tokens=input_length + processed_output.output_token_length,
            completion_tokens=processed_output.output_token_length
        )
        return ChatCompletionResponse(
            choices=[choice],
            usage=usage,
            model=self.model.__class__.__name__
        )

    async def _complete_stream_response(
            self, input_length: int, output_generator: AsyncGenerator[GenerateOutput, None]
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        # TODO(wangyin): maybe deal with the case of multiple returns.
        # TODO(wangyin): deal with returning function call.
        yield ChatCompletionStreamResponse(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role=RoleEnum.assistant,
                    ),
                )
            ],
        )

        responded_string = ""
        responded_length = 0
        finish_reason = None
        async for output in output_generator:
            processed_output = self._process_output_ids_tensor(input_length, output.output_ids)
            output_string = processed_output.output_str
            output_length = len(processed_output.output_str)
            finish_reason = processed_output.finish_reason
            if output_length > responded_length:
                delta_string = output_string[responded_length:]
                responded_string = output_string
                responded_length = output_length

                yield ChatCompletionStreamResponse(
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                content=delta_string,
                            ),
                        )
                    ],
                )

        if finish_reason == None:
            logging.warn(f"output [{responded_string}] found no stop reason! use stop as default.")
            finish_reason = FinisheReason.stop

        yield ChatCompletionStreamResponse(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        content="",
                    ),
                    finish_reason=finish_reason
                )
            ],
        )

    async def chat_completion(
            self, chat_request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionStreamResponse, None]]:
        input_ids = self.template_renderer.render(chat_request)
        input_length = len(input_ids)

        input_id_tensor = torch.Tensor(input_ids).int().unsqueeze(0)
        input_length_tensor = torch.Tensor([input_length]).int()
        input_images = [[]]
        generate_config = self._extract_generation_config(chat_request)

        output_generator: AsyncGenerator[GenerateOutput, None] = self.model.generate_stream(
            input_id_tensor,
            input_length_tensor,
            input_images,
            generate_config
        )

        if chat_request.stream:
            return self._complete_stream_response(input_length, output_generator)
        else:
            return await self._complete_non_stream_response(input_length, output_generator)

