import itertools
import json
import logging
from functools import partial
from typing import AsyncGenerator, List, Optional

from fastapi import Request

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DebugInfo,
    FunctionCall,
    ModelCard,
    ModelList,
    RoleEnum,
    ToolCall,
    UsageInfo,
)
from rtp_llm.openai.renderer_factory import ChatRendererFactory
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
    StreamResponseObject,
)
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


class OpenaiEndpoint(object):
    def __init__(
        self,
        model_config: GptInitModelParameters,
        tokenizer: BaseTokenizer,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
    ):
        self._init_basic_attributes(model_config, tokenizer, backend_rpc_server_visitor)
        self._init_renderers()
        self._init_stop_words()

    def _init_basic_attributes(
        self,
        model_config: GptInitModelParameters,
        tokenizer: BaseTokenizer,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
    ) -> None:
        self.model_config = model_config
        self.max_seq_len = self.model_config.max_seq_len

        if tokenizer is None:
            raise AttributeError(f"tokenizer is none!")
        self.tokenizer: BaseTokenizer = tokenizer
        self.backend_rpc_server_visitor = backend_rpc_server_visitor

        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.model_config.special_tokens.eos_token_id

        self.stop_words_id_list = self.model_config.special_tokens.stop_words_id_list

    def _init_renderers(self) -> None:
        render_params = RendererParams(
            model_type=self.model_config.py_env_configs.model_config.model_type,
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
            template_type=self.model_config.template_type,
            ckpt_path=self.model_config.ckpt_path,
        )

        self.chat_renderer: CustomChatRenderer = ChatRendererFactory.get_renderer(
            self.tokenizer, render_params
        )
        logging.info(f"Finally openai endpoint uses renderer: {self.chat_renderer}")

        self.template_renderer: CustomChatRenderer = (
            self.chat_renderer
            if isinstance(self.chat_renderer, BasicRenderer)
            else BasicRenderer(self.tokenizer, render_params)
        )
        logging.info(f"chat_renderer [{self.chat_renderer}] is created.")

        extra_stop_word_ids_list = self.chat_renderer.get_all_extra_stop_word_ids_list()
        self.stop_words_id_list.extend(extra_stop_word_ids_list)

    def _init_stop_words(self) -> None:
        self.stop_words_str_list = self.model_config.special_tokens.stop_words_str_list

        # Merge environment stop words
        env_config = self.model_config.py_env_configs.generate_env_config
        env_stop_words_str_list = (
            json.loads(env_config.stop_words_str) if env_config.stop_words_str else []
        )
        env_stop_words_id_list = (
            json.loads(env_config.stop_words_list) if env_config.stop_words_list else []
        )

        if env_config.force_stop_words:
            self.stop_words_str_list = env_stop_words_str_list
            self.stop_words_id_list = env_stop_words_id_list
        else:
            self.stop_words_str_list += env_stop_words_str_list
            self.stop_words_id_list += env_stop_words_id_list

        # Sync between stop word IDs and strings
        stop_words_str_list_from_id = []
        for stop_word_ids in self.stop_words_id_list:
            word = self.tokenizer.decode(stop_word_ids)
            if len(word):
                stop_words_str_list_from_id.append(word)

        stop_words_id_list_from_str = []
        for stop_word_str in self.stop_words_str_list:
            ids = self.tokenizer.encode(stop_word_str)
            if len(ids):
                stop_words_id_list_from_str.append(ids)

        self.stop_words_str_list += stop_words_str_list_from_id
        self.stop_words_id_list += stop_words_id_list_from_str

        # Dedup and log
        self.stop_words_str_list = list(set(self.stop_words_str_list))
        self.stop_words_id_list = self._dedup_stop_words_list(self.stop_words_id_list)

        logging.info(
            f"use stop_words_str_list [{self.stop_words_str_list}], "
            f"stop_words_id_list [{self.stop_words_id_list}]"
        )

    async def list_models(self):
        global model_args
        model_card = ModelCard(id=self.model_config.model_name)
        return ModelList(data=[model_card])

    def _dedup_stop_words_list(
        self, stop_words_list: List[List[int]]
    ) -> List[List[int]]:
        return [i for i, _ in itertools.groupby(sorted(stop_words_list))]

    def _extract_generation_config(
        self, request: ChatCompletionRequest
    ) -> GenerateConfig:
        config = request.extra_configs or GenerateConfig()

        self._apply_basic_config(request, config)
        self._apply_stop_words_config(request, config)
        self._apply_advanced_config(request, config)

        config.convert_select_tokens(len(self.tokenizer), self.tokenizer)
        config.add_thinking_params(self.tokenizer)
        return config

    def _apply_basic_config(
        self, request: ChatCompletionRequest, config: GenerateConfig
    ) -> None:
        if request.trace_id is not None:
            config.trace_id = request.trace_id
        if request.stream:
            config.is_streaming = True
        if request.temperature is not None:
            config.temperature = request.temperature
        if request.top_p is not None:
            config.top_p = request.top_p
        if request.max_tokens is not None:
            config.max_new_tokens = request.max_tokens
        if request.n is not None:
            config.num_return_sequences = request.n

    def _apply_stop_words_config(
        self, request: ChatCompletionRequest, config: GenerateConfig
    ) -> None:
        request_stop_words_list = request.stop if request.stop is not None else []
        if isinstance(request_stop_words_list, str):
            request_stop_words_list = [request_stop_words_list]

        config.stop_words_str = list(
            set(
                self.stop_words_str_list
                + request_stop_words_list
                + config.stop_words_str
            )
        )
        config.stop_words_list = self._dedup_stop_words_list(
            self.stop_words_id_list
            + self.chat_renderer.tokenize_words(config.stop_words_str)
            + config.stop_words_list
        )

    def _apply_advanced_config(
        self, request: ChatCompletionRequest, config: GenerateConfig
    ) -> None:
        if request.chat_id is not None:
            config.chat_id = request.chat_id
        if request.seed is not None:
            config.random_seed = request.seed
        if request.logprobs is not None:
            config.return_all_probs = request.logprobs
        if request.logprobs or request.functions:
            config.is_streaming = True

        if (
            request.extra_configs
            and request.extra_configs.max_thinking_tokens is not None
            and isinstance(request.extra_configs.max_thinking_tokens, int)
        ):
            config.max_thinking_tokens = request.extra_configs.max_thinking_tokens

    def _merge_tool_calls(
        self,
        existing_tool_calls: Optional[List[ToolCall]],
        delta_tool_calls: Optional[List[ToolCall]],
    ) -> Optional[List[ToolCall]]:
        """
        合并增量的 tool_calls 到现有的 tool_calls 中
        Args:
            existing_tool_calls: 现有的 tool_calls 列表
            delta_tool_calls: 增量的 tool_calls 列表
        Returns:
            合并后的 tool_calls 列表
        """
        if delta_tool_calls is None:
            return existing_tool_calls
        if existing_tool_calls is None:
            existing_tool_calls = []

        for delta_tool_call in delta_tool_calls:
            # Find existing tool_call with same index
            existing_tool_call = None
            if delta_tool_call.index is not None:
                for tool_call in existing_tool_calls:
                    if tool_call.index == delta_tool_call.index:
                        existing_tool_call = tool_call
                        break

            if existing_tool_call is None:
                # Add new tool_call
                self._add_new_tool_call(existing_tool_calls, delta_tool_call)
            else:
                # Update existing tool_call
                if delta_tool_call.id:
                    existing_tool_call.id = delta_tool_call.id
                if delta_tool_call.type:
                    existing_tool_call.type = delta_tool_call.type
                if delta_tool_call.function:
                    self._update_function_call(
                        existing_tool_call, delta_tool_call.function
                    )

        return existing_tool_calls

    def _add_new_tool_call(
        self, tool_calls: List[ToolCall], delta_tool_call: ToolCall
    ) -> None:
        new_tool_call = ToolCall(
            index=delta_tool_call.index,
            id=delta_tool_call.id,
            type=delta_tool_call.type,
            function=FunctionCall(
                name=(
                    delta_tool_call.function.name if delta_tool_call.function else None
                ),
                arguments=(
                    delta_tool_call.function.arguments
                    if delta_tool_call.function
                    else None
                ),
            ),
        )
        tool_calls.append(new_tool_call)

    def _update_function_call(
        self, tool_call: ToolCall, delta_function: FunctionCall
    ) -> None:
        if tool_call.function is None:
            tool_call.function = FunctionCall(
                name=delta_function.name,
                arguments=delta_function.arguments,
            )
        else:
            if delta_function.name:
                tool_call.function.name = delta_function.name
            if delta_function.arguments:
                if tool_call.function.arguments is None:
                    tool_call.function.arguments = delta_function.arguments
                else:
                    tool_call.function.arguments += delta_function.arguments

    def _append_string_field(
        self, existing_value: Optional[str], delta_value: Optional[str]
    ) -> Optional[str]:
        """拼接字符串字段，处理 None 的情况"""
        if existing_value is None:
            return delta_value or None
        return existing_value + (delta_value or "")

    def _update_choice_with_delta(
        self, choice: ChatCompletionResponseChoice, delta_choice
    ) -> None:
        """使用增量数据更新 choice"""
        # 更新字符串字段
        choice.message.content = self._append_string_field(
            choice.message.content, delta_choice.delta.content
        )
        choice.message.reasoning_content = self._append_string_field(
            choice.message.reasoning_content, delta_choice.delta.reasoning_content
        )

        # 更新其他字段（取最新值）
        choice.message.role = delta_choice.delta.role or choice.message.role
        choice.message.function_call = (
            delta_choice.delta.function_call or choice.message.function_call
        )

        # 合并 tool_calls
        choice.message.tool_calls = self._merge_tool_calls(
            choice.message.tool_calls, delta_choice.delta.tool_calls
        )

        # 更新 finish_reason
        choice.finish_reason = delta_choice.finish_reason or choice.finish_reason

        # 处理 logprobs
        if choice.logprobs is not None and delta_choice.logprobs is not None:
            choice.logprobs.content += delta_choice.logprobs.content
        elif delta_choice.logprobs is not None:
            choice.logprobs = delta_choice.logprobs

    def _initialize_choices(
        self, response_choices: List
    ) -> List[ChatCompletionResponseChoice]:
        """从响应的第一批数据初始化 choices"""
        return [
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
            )
            for i, choice in enumerate(response_choices)
        ]

    async def _collect_complete_response(
        self,
        choice_generator: Optional[AsyncGenerator[StreamResponseObject, None]],
        debug_info: Optional[DebugInfo],
    ) -> ChatCompletionResponse:
        all_choices = []
        usage = None
        aux_info = None
        extra_outputs = None

        async for response in choice_generator:
            # 初始化 choices（仅第一次）
            if not all_choices:
                all_choices = self._initialize_choices(response.choices)
            elif len(response.choices) != len(all_choices):
                raise ValueError(
                    f"response.choices has different length! "
                    f"[{response.choices}] vs [{all_choices}]."
                )
            else:
                # 更新每个 choice
                for i, delta_choice in enumerate(response.choices):
                    self._update_choice_with_delta(all_choices[i], delta_choice)

            # 更新全局字段
            usage = response.usage or usage
            aux_info = response.aux_info or aux_info
            extra_outputs = response.extra_outputs or extra_outputs

        # 确保 usage 不为 None
        if usage is None:
            logging.warning("No usage returned from stream response. use empty value.")
            usage = UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0)

        return ChatCompletionResponse(
            choices=all_choices,
            usage=usage,
            aux_info=aux_info,
            model=self.model_config.model_name,
            debug_info=debug_info,
            extra_outputs=extra_outputs,
        )

    def _complete_stream_response(
        self,
        choice_generator: AsyncGenerator[StreamResponseObject, None],
        debug_info: Optional[DebugInfo],
    ) -> CompleteResponseAsyncGenerator:
        async def response_generator():
            debug_info_responded = False
            async for response in choice_generator:
                yield ChatCompletionStreamResponse(
                    choices=response.choices,
                    usage=response.usage,
                    aux_info=response.aux_info,
                    debug_info=debug_info if not debug_info_responded else None,
                    extra_outputs=response.extra_outputs,
                )
                debug_info_responded = True

        complete_response_collect_func = partial(
            self._collect_complete_response, debug_info=debug_info
        )
        return CompleteResponseAsyncGenerator(
            response_generator(), complete_response_collect_func
        )

    def _get_debug_info(
        self,
        renderer: CustomChatRenderer,
        renderered_input: RenderedInputs,
        gen_config: GenerateConfig,
    ) -> DebugInfo:
        if renderered_input.rendered_prompt != "":
            prompt = renderered_input.rendered_prompt
        else:
            prompt = self.tokenizer.decode(renderered_input.input_ids)
        return DebugInfo(
            input_prompt=prompt,
            input_ids=renderered_input.input_ids,
            input_urls=[
                mm_input.url for mm_input in renderered_input.multimodal_inputs
            ],
            tokenizer_info=str(self.tokenizer),
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
            stop_words_list=self.stop_words_str_list,
            renderer_info=renderer.get_renderer_info(),
            generate_config=gen_config,
        )

    def render_chat(self, chat_request: ChatCompletionRequest):
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
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
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = self.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)

        mm_inputs = rendered_input.multimodal_inputs

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(
                generate_config.sp_advice_prompt
            )

        debug_info = (
            self._get_debug_info(renderer, rendered_input, generate_config)
            if chat_request.debug_info
            else None
        )

        choice_generator = renderer.generate_choice(
            request_id,
            rendered_input.input_ids,
            mm_inputs,
            generate_config,
            self.backend_rpc_server_visitor,
            chat_request,
        )

        return self._complete_stream_response(choice_generator, debug_info)

    def chat_render(self, chat_request: ChatCompletionRequest) -> DebugInfo:
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = renderer.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)
        debug_info = self._get_debug_info(renderer, rendered_input, generate_config)
        return debug_info
