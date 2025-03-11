from fastapi import Request
import torch
from typing import Union, Optional, List, Dict, Generator, Coroutine, AsyncGenerator, Any, Iterator
import os
import json
import logging
import json
from functools import partial

from transformers import PreTrainedTokenizerBase
from maga_transformer.utils.util import str_to_bool
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from transformers import PreTrainedTokenizerBase
from maga_transformer.openai.api_datatype import ModelCard, ModelList, ChatMessage, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, UsageInfo, \
    ChatCompletionStreamResponse, \
    DebugInfo
from maga_transformer.openai.renderers.custom_renderer import RendererParams, \
    StreamResponseObject, RenderedInputs, CustomChatRenderer
from maga_transformer.openai.renderer_factory import ChatRendererFactory
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.mm_process_engine import MMProcessEngine
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.backend_rpc_server_visitor import BackendRPCServerVisitor

class OpenaiEndopoint():
    def __init__(self, model_config: GptInitModelParameters,
                 tokenizer: PreTrainedTokenizerBase,
                 backend_rpc_server_visitor: BackendRPCServerVisitor):
        self.model_config = model_config
        self.max_seq_len = self.model_config.max_seq_len

        if (tokenizer == None):
            raise AttributeError(f"tokenizer is none!")
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.backend_rpc_server_visitor = backend_rpc_server_visitor

        self.eos_token_id = None
        if (isinstance(tokenizer, PreTrainedTokenizerBase)):
            self.eos_token_id = tokenizer.eos_token_id
        if (self.eos_token_id == None):
            self.eos_token_id = self.model_config.special_tokens.eos_token_id

        self.stop_words_id_list = self.model_config.special_tokens.stop_words_id_list

        render_params = RendererParams(
            model_type=os.environ["MODEL_TYPE"],
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
            template_type=self.model_config.template_type,
            ckpt_path=self.model_config.ckpt_path
        )

        self.chat_renderer: CustomChatRenderer = ChatRendererFactory.get_renderer(self.tokenizer, render_params)
        logging.info(f"Finally openai endpoint uses renderer: {self.chat_renderer} ")
        self.template_renderer: CustomChatRenderer = self.chat_renderer \
            if isinstance(self.chat_renderer,BasicRenderer) \
            else BasicRenderer(self.tokenizer, render_params)
        logging.info(f"chat_renderer [{self.chat_renderer}] is created.")
        extra_stop_word_ids_list = self.chat_renderer.get_all_extra_stop_word_ids_list()
        self.stop_words_id_list.extend(extra_stop_word_ids_list)
        self.stop_words_str_list = []
        for stop_word_ids in self.stop_words_id_list:
            word = self.tokenizer.decode(stop_word_ids)
            if len(word):
                self.stop_words_str_list.append(word)

        env_stop_words_str = os.environ.get('STOP_WORDS_STR', None)
        env_stop_words_id = os.environ.get('STOP_WORDS_LIST', None)
        env_stop_words_str_list = json.loads(env_stop_words_str) if env_stop_words_str else []
        env_stop_words_id_list = json.loads(env_stop_words_id) if env_stop_words_id else []
        env_force_stop = os.environ.get('FORCE_STOP_WORDS', None)
        if env_force_stop and str_to_bool(env_force_stop):
            self.stop_words_str_list = env_stop_words_str_list
            self.stop_words_id_list = env_stop_words_id_list
        else:
            self.stop_words_str_list = self.stop_words_str_list + env_stop_words_str_list
            self.stop_words_id_list = self.stop_words_id_list + env_stop_words_id_list
        
        logging.info(f"use stop_words_str_list [{self.stop_words_str_list}], " \
                    f"stop_words_id_list [{self.stop_words_id_list}]")

    async def list_models(self):
        global model_args
        model_card = ModelCard(id=self.model_config.model_name)
        return ModelList(data=[model_card])

    def _extract_generation_config(self, request: ChatCompletionRequest) -> GenerateConfig:
        # TODO(wangyin): implement this
        config = request.extra_configs or GenerateConfig()
        if request.stream != None:
            config.is_streaming = request.stream
        if request.temperature != None:
            config.temperature = request.temperature
        if request.top_p != None:
            config.top_p = request.top_p
        if request.max_tokens != None:
            config.max_new_tokens = request.max_tokens
        if request.n != None:
            config.num_return_sequences = request.n
        request_stop_words_list = request.stop if request.stop != None else []
        if isinstance(request_stop_words_list, str):
            request_stop_words_list = [request_stop_words_list]
        config.stop_words_str = self.stop_words_str_list + request_stop_words_list
        config.stop_words_list = self.stop_words_id_list + self.chat_renderer.tokenize_words(request_stop_words_list)
        if request.chat_id != None:
            config.chat_id = request.chat_id
        if request.seed != None:
            config.random_seed = request.seed
        if request.logprobs != None:
            config.return_all_probs = request.logprobs
        if request.logprobs or request.functions:
            config.is_streaming = True
        config.add_special_tokens(self.model_config.special_tokens)
        config.convert_select_tokens(self.model_config.vocab_size, self.tokenizer)
        if request.extend_fields and "max_thinking_tokens" in request.extend_fields.keys() \
            and isinstance(request.extend_fields["max_thinking_tokens"], int):
            config.max_thinking_tokens = request.extend_fields["max_thinking_tokens"]
        config.add_thinking_params(self.tokenizer)
        return config

    async def _collect_complete_response(
            self,
            choice_generator: Optional[AsyncGenerator[StreamResponseObject, None]],
            debug_info: Optional[DebugInfo]) -> ChatCompletionResponse:
        all_choices = []
        usage = None
        aux_info = None
        async for response in choice_generator:
            if len(response.choices) != len(all_choices):
                if (all_choices == []):
                    all_choices = [
                        ChatCompletionResponseChoice(
                            index=i,
                            message=ChatMessage(
                                role=choice.delta.role or RoleEnum.assistant,
                                content=choice.delta.content or None,
                                function_call=choice.delta.function_call or None,
                                tool_calls=choice.delta.tool_calls or None,
                            ),
                            finish_reason=choice.finish_reason,
                            logprobs=choice.logprobs,
                        ) for i, choice in enumerate(response.choices)
                    ]
                else:
                    raise ValueError(f"response.choices has different length! "
                                     f"[{response.choices}] vs [{all_choices}].")
            else:
                for i in range(len(all_choices)):
                    if all_choices[i].message.content == None:
                        all_choices[i].message.content = (response.choices[i].delta.content or None)
                    else:
                        all_choices[i].message.content += (response.choices[i].delta.content or "")
                    if all_choices[i].message.reasoning_content == None:
                        all_choices[i].message.reasoning_content = (response.choices[i].delta.reasoning_content or None)
                    else:
                        all_choices[i].message.reasoning_content += (response.choices[i].delta.reasoning_content or "")
                    all_choices[i].message.role = response.choices[i].delta.role or all_choices[i].message.role
                    all_choices[i].message.function_call = response.choices[i].delta.function_call or all_choices[i].message.function_call
                    all_choices[i].message.tool_calls = (
                        response.choices[i].delta.tool_calls
                        or all_choices[i].message.tool_calls
                    )
                    all_choices[i].finish_reason = response.choices[i].finish_reason or all_choices[i].finish_reason
                    if all_choices[i].logprobs != None:
                        if response.choices[i].logprobs != None:
                            all_choices[i].logprobs.content += response.choices[i].logprobs.content
                    else:
                        all_choices[i].logprobs = response.choices[i].logprobs
            usage = response.usage or usage
            aux_info = response.aux_info or aux_info

        if (usage == None):
            logging.warning(f"No usage returned from stream response. use empty value.")
            usage = UsageInfo(
                prompt_tokens=0,
                total_tokens=0,
                completion_tokens=0
            )
        return ChatCompletionResponse(
            choices=all_choices,
            usage=usage,
            aux_info=aux_info,
            model=self.model_config.model_name,
            debug_info=debug_info,
        )

    def _complete_stream_response(
            self, choice_generator: AsyncGenerator[StreamResponseObject, None],
            debug_info: Optional[DebugInfo]
    ) -> CompleteResponseAsyncGenerator:
        async def response_generator():
            debug_info_responded = False
            async for response in choice_generator:
                yield ChatCompletionStreamResponse(
                    choices=response.choices,
                    usage=response.usage,
                    aux_info=response.aux_info,
                    debug_info=debug_info if not debug_info_responded else None
                )
                debug_info_responded = True

        complete_response_collect_func = partial(self._collect_complete_response, debug_info=debug_info)
        return CompleteResponseAsyncGenerator(response_generator(), complete_response_collect_func)

    def _get_debug_info(self, renderer: CustomChatRenderer,
                        renderered_input: RenderedInputs, gen_config: GenerateConfig) -> DebugInfo:
        if renderered_input.rendered_prompt != "":
            prompt = renderered_input.rendered_prompt
        else:
            prompt = self.tokenizer.decode(renderered_input.input_ids)
        return DebugInfo(
            input_prompt=prompt,
            input_ids=renderered_input.input_ids,
            input_urls=[mm_input.url for mm_input in renderered_input.multimodal_inputs],
            tokenizer_info=str(self.tokenizer),
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
            stop_words_list=self.stop_words_str_list,
            renderer_info=renderer.get_renderer_info(),
            generate_config=gen_config
        )

    def render_chat(self, chat_request: ChatCompletionRequest):
        renderer = self.template_renderer if chat_request.user_template else self.chat_renderer
        prepopulate_str = ""
        if len(chat_request.messages) > 0 and chat_request.messages[-1].partial:
            prepopulate_str = str(chat_request.messages[-1].content)
            chat_request.messages.pop()
        rendered_input = renderer.render_chat(chat_request)
        if prepopulate_str != "":
            rendered_input.rendered_prompt += prepopulate_str
            rendered_input.input_ids += self.tokenizer.encode(prepopulate_str)
        return rendered_input

    def chat_completion(
            self, request_id: int, chat_request: ChatCompletionRequest, raw_request: Request
    ) -> CompleteResponseAsyncGenerator:
        renderer = self.template_renderer if chat_request.user_template else self.chat_renderer
        rendered_input = self.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)

        mm_inputs = []
        if self.model_config.is_multimodal:
            mm_inputs = rendered_input.multimodal_inputs
        else:
            mm_inputs = []

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(generate_config.sp_advice_prompt)

        debug_info = self._get_debug_info(renderer, rendered_input, generate_config) \
            if chat_request.debug_info else None
        choice_generator = renderer.generate_choice(
            request_id,
            rendered_input.input_ids,
            mm_inputs,
            generate_config,
            self.backend_rpc_server_visitor,
            chat_request
        )

        return self._complete_stream_response(choice_generator, debug_info)

    def chat_render(self, chat_request: ChatCompletionRequest) -> DebugInfo:
        renderer = self.template_renderer if chat_request.user_template else self.chat_renderer
        rendered_input = renderer.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)
        debug_info = self._get_debug_info(renderer, rendered_input, generate_config)
        return debug_info
