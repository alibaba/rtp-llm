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
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, UsageInfo, \
    FinisheReason, DeltaMessage, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse
from maga_transformer.openai.renderers.custom_renderer import RendererParams, ProcessedOutput, \
    StreamResponseObject
from maga_transformer.openai.renderers.renderer_factory import ChatRendererFactory
from maga_transformer.config.generate_config import GenerateConfig

class OpenaiEndopoint():
    def __init__(self, model: Union[AsyncModel, BaseModel]):
        self.model = model
        self.max_seq_len = self.model.config.max_seq_len

        tokenizer = self.model.tokenizer
        if (tokenizer == None):
            raise AttributeError(f"model [{model}] has no tokenizer!")
        self.tokenizer: Union[PreTrainedTokenizer, BaseTokenizer] = tokenizer

        self.eos_token_id = None
        if (isinstance(tokenizer, PreTrainedTokenizer)):
            self.eos_token_id = tokenizer.eos_token_id
        if (self.eos_token_id == None):
            self.eos_token_id = self.model.config.special_tokens.eos_token_id

        self.stop_word_ids_list = self.model.config.special_tokens.stop_words_list
        self.stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in self.stop_word_ids_list
        ]

        render_params = RendererParams(
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_word_ids_list,
        )
        self.chat_renderer = ChatRendererFactory.get_renderer(self.tokenizer, render_params)
        extra_stop_word_ids_list = self.chat_renderer.get_extra_stop_word_ids_list()
        self.stop_word_ids_list.extend(extra_stop_word_ids_list)

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
        config.stop_words_list = self.stop_word_ids_list
        return config

    async def _complete_non_stream_response(
            self, choice_generator: AsyncGenerator[StreamResponseObject, None],
    ) -> ChatCompletionResponse:
        all_choices = []
        usage = None

        async for response in choice_generator:
            if len(response.choices) != len(all_choices):
                if (all_choices == []):
                    all_choices = [
                        ChatCompletionResponseChoice(
                            index=0,
                            message=ChatMessage(
                                role=choice.delta.role or RoleEnum.assistant,
                                content=choice.delta.content or "",
                                function_call=choice.delta.function_call or None,
                            ),
                            finish_reason=choice.finish_reason
                        ) for choice in response.choices
                    ]
                else:
                    raise ValueError(f"response.choices has different length! "
                                     f"[{response.choices}] vs [{all_choices}].")
            else:
                for i in range(len(all_choices)):
                    all_choices[i].message.content += (response.choices[i].delta.content or "")
                    all_choices[i].message.role = response.choices[i].delta.role or all_choices[i].message.role
                    all_choices[i].message.function_call = response.choices[i].delta.function_call or all_choices[i].message.function_call
                    all_choices[i].finish_reason = response.choices[i].finish_reason or all_choices[i].finish_reason
            usage = response.usage or usage

        if (usage == None):
            logging.warn(f"No usage returned from stream response. use empty value.")
            usage = UsageInfo(
                prompt_tokens=0,
                total_tokens=0,
                completion_tokens=0
            )

        return ChatCompletionResponse(
            choices=all_choices,
            usage=usage,
            model=self.model.__class__.__name__
        )

    async def _complete_stream_response(
            self, choice_generator: AsyncGenerator[StreamResponseObject, None],
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        async for response in choice_generator:
            yield ChatCompletionStreamResponse(
                choices=response.choices,
                usage=response.usage,
            )

    async def chat_completion(
            self, chat_request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionStreamResponse, None]]:
        input_ids = self.chat_renderer.render_chat(chat_request)
        input_length = len(input_ids)

        input_id_tensor = torch.Tensor(input_ids).int().unsqueeze(0)
        input_length_tensor = torch.Tensor([input_length]).int()
        input_images = [[]]
        generate_config = self._extract_generation_config(chat_request)

        output_generator: AsyncGenerator[GenerateOutput, None] = self.model.generate_stream(
            input_id_tensor,
            self.tokenizer,
            input_length_tensor,
            input_images,
            generate_config
        )

        choice_generator = self.chat_renderer.render_response_stream(
            output_generator,
            chat_request,
            input_length
        )

        if chat_request.stream:
            return self._complete_stream_response(choice_generator) # type: ignore
        else:
            return await self._complete_non_stream_response(choice_generator) # type: ignore

