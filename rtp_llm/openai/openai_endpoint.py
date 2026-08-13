import copy
import itertools
import json
import logging
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Request

from rtp_llm.config.generate_config import GenerateConfig, ReturnAllProbsMode
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.frontend.recommendation_parser import parse_and_fill_banned_combo
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMetadata,
    ChatCompletionStreamResponse,
    ChatMessage,
    DebugInfo,
    FunctionCall,
    ModelCard,
    ModelList,
    RoleEnum,
    ToolCall,
    UsageInfo,
    create_chat_completion_response_metadata,
)
from rtp_llm.openai.renderer_factory import ChatRendererFactory
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
    StreamResponseObject,
)
from rtp_llm.ops import SpecialTokens
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.complete_response_async_generator import (
    CloseDependencyRegistry,
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.base_model_datatypes import (
    RequestDeadlineAnchor,
    initialize_request_deadlines,
)


class OpenaiEndpoint(object):
    def __init__(
        self,
        model_config: ModelConfig,
        misc_config: PyMiscellaneousConfig,
        vit_config: VitConfig,
        tokenizer: BaseTokenizer,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
    ):
        # Get values from model_config
        self.generate_env_config = model_config.generate_env_config
        self.max_seq_len = model_config.max_seq_len
        self.model_name = model_config.model_name
        self.special_tokens = model_config.special_tokens
        template_type = model_config.template_type
        ckpt_path = model_config.ckpt_path
        render_config = model_config.render_config

        if tokenizer == None:
            raise AttributeError(f"tokenizer is none!")
        self.tokenizer: BaseTokenizer = tokenizer
        self.backend_rpc_server_visitor = backend_rpc_server_visitor

        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id == None:
            self.eos_token_id = self.special_tokens.eos_token_id

        self.stop_words_id_list = self.special_tokens.stop_words_id_list

        render_params = RendererParams(
            model_type=model_config.model_type,
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
            template_type=template_type,
            ckpt_path=ckpt_path,
        )

        self.chat_renderer: CustomChatRenderer = ChatRendererFactory.get_renderer(
            self.tokenizer,
            render_params,
            self.generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        logging.info(f"Finally openai endpoint uses renderer: {self.chat_renderer} ")
        self.template_renderer: CustomChatRenderer = (
            self.chat_renderer
            if isinstance(self.chat_renderer, BasicRenderer)
            else BasicRenderer(
                self.tokenizer,
                render_params,
                self.generate_env_config,
                render_config,
                ckpt_path,
                misc_config,
                vit_config,
            )
        )
        logging.info(f"chat_renderer [{self.chat_renderer}] is created.")
        extra_stop_word_ids_list = self.chat_renderer.get_all_extra_stop_word_ids_list()
        self.stop_words_id_list.extend(extra_stop_word_ids_list)
        self.stop_words_str_list = self.special_tokens.stop_words_str_list

        env_stop_words_str = self.generate_env_config.stop_words_str
        env_stop_words_id = self.generate_env_config.stop_words_list
        env_stop_words_str_list = (
            json.loads(env_stop_words_str) if env_stop_words_str else []
        )
        env_stop_words_id_list = (
            json.loads(env_stop_words_id) if env_stop_words_id else []
        )
        env_force_stop = self.generate_env_config.force_stop_words
        if env_force_stop:
            self.stop_words_str_list = env_stop_words_str_list
            self.stop_words_id_list = env_stop_words_id_list
        else:
            self.stop_words_str_list = (
                self.stop_words_str_list + env_stop_words_str_list
            )
            self.stop_words_id_list = self.stop_words_id_list + env_stop_words_id_list

        # sync between stop word id str and stop words id list
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

        # dedup stop words
        self.stop_words_str_list = list(set(self.stop_words_str_list))
        self.stop_words_id_list = self._dedup_stop_words_list(self.stop_words_id_list)

        logging.info(
            f"use stop_words_str_list [{self.stop_words_str_list}], "
            f"stop_words_id_list [{self.stop_words_id_list}]"
        )

    async def list_models(self):
        model_card = ModelCard(id=self.model_name)
        return ModelList(data=[model_card])

    def _dedup_stop_words_list(
        self, stop_words_list: List[List[int]]
    ) -> List[List[int]]:
        return [i for i, _ in itertools.groupby(sorted(stop_words_list))]

    def _extract_generation_config(
        self, request: ChatCompletionRequest
    ) -> GenerateConfig:
        # TODO(wangyin): implement this
        config = request.extra_configs or GenerateConfig()
        if request.trace_id != None:
            config.trace_id = request.trace_id
        if request.stream == True:
            config.is_streaming = True
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
        if request.chat_id != None:
            config.chat_id = request.chat_id
        if request.seed != None:
            config.random_seed = request.seed
        if request.logprobs != None:
            if not request.logprobs:
                config.return_all_probs = ReturnAllProbsMode.NONE
            # Priority: if extra_configs.return_all_probs is already set to
            # something non-NONE (typically ORIGINAL), honor that — caller has
            # explicitly opted into a specific mode. Only fall through to
            # logprobs_mode when no extra_configs override is present.
            elif config.return_all_probs == ReturnAllProbsMode.NONE:
                if request.logprobs_mode == "original":
                    config.return_all_probs = ReturnAllProbsMode.ORIGINAL
                else:
                    config.return_all_probs = ReturnAllProbsMode.DEFAULT
        if request.logprobs or request.functions:
            config.is_streaming = True
        config.convert_select_tokens(len(self.tokenizer), self.tokenizer)

        if (
            request.extra_configs
            and request.extra_configs.max_thinking_tokens is not None
            and isinstance(request.extra_configs.max_thinking_tokens, int)
        ):
            config.max_thinking_tokens = request.extra_configs.max_thinking_tokens
        # add_thinking_params now accepts generate_env_config parameter
        config.add_thinking_params(self.tokenizer, self.generate_env_config)
        if request.debug_info:
            config.return_output_ids = True
        return config

    @staticmethod
    def _merge_function_call(
        existing_function_call: Optional[FunctionCall],
        delta_function_call: Optional[FunctionCall],
    ) -> Optional[FunctionCall]:
        if delta_function_call is None:
            return existing_function_call
        if existing_function_call is None:
            return FunctionCall(
                name=delta_function_call.name,
                arguments=delta_function_call.arguments,
            )

        if delta_function_call.name:
            if (
                existing_function_call.name
                and existing_function_call.name != delta_function_call.name
            ):
                raise ValueError("conflicting function call name in response stream")
            if not existing_function_call.name:
                existing_function_call.name = delta_function_call.name
        if delta_function_call.arguments:
            existing_function_call.arguments = (
                existing_function_call.arguments or ""
            ) + delta_function_call.arguments
        return existing_function_call

    @staticmethod
    def _merge_tool_calls(
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
        for tool_call in delta_tool_calls:
            if tool_call.index is not None and (
                type(tool_call.index) is not int or tool_call.index < 0
            ):
                raise ValueError(
                    f"tool call index must be a non-negative integer: {tool_call.index!r}"
                )
            if tool_call.id is not None and not isinstance(tool_call.id, str):
                raise ValueError(
                    f"tool call id must be a string or null: {tool_call.id!r}"
                )
        indexes = [
            tool_call.index
            for tool_call in delta_tool_calls
            if tool_call.index is not None
        ]
        ids = [tool_call.id for tool_call in delta_tool_calls if tool_call.id]
        if len(indexes) != len(set(indexes)) or len(ids) != len(set(ids)):
            raise ValueError("response chunk contains duplicate tool call identity")
        identityless_delta_count = sum(
            tool_call.index is None and not tool_call.id
            for tool_call in delta_tool_calls
        )
        if identityless_delta_count and len(delta_tool_calls) != 1:
            raise ValueError(
                "tool call delta is missing index and id in a multi-call response"
            )

        for delta_tool_call in delta_tool_calls:
            existing_by_index = None
            if delta_tool_call.index is not None:
                existing_by_index = next(
                    (
                        existing
                        for existing in existing_tool_calls
                        if existing.index == delta_tool_call.index
                    ),
                    None,
                )
            existing_by_id = None
            if delta_tool_call.id:
                existing_by_id = next(
                    (
                        existing
                        for existing in existing_tool_calls
                        if existing.id and existing.id == delta_tool_call.id
                    ),
                    None,
                )
            if (
                existing_by_index is not None
                and existing_by_id is not None
                and existing_by_index is not existing_by_id
            ):
                raise ValueError("tool call delta has conflicting index and id")

            existing_tool_call = existing_by_index or existing_by_id
            if (
                existing_tool_call is None
                and delta_tool_call.index is None
                and not delta_tool_call.id
            ):
                if len(existing_tool_calls) == 1:
                    existing_tool_call = existing_tool_calls[0]
                else:
                    raise ValueError(
                        "tool call delta is missing index and id without a unique call"
                    )
            if existing_tool_call is None and (
                delta_tool_call.index is not None or bool(delta_tool_call.id)
            ):
                partially_matching_existing = [
                    existing
                    for existing in existing_tool_calls
                    if not (
                        existing.index is not None
                        and delta_tool_call.index is not None
                        and existing.index != delta_tool_call.index
                    )
                    and not (
                        bool(existing.id)
                        and bool(delta_tool_call.id)
                        and existing.id != delta_tool_call.id
                    )
                ]
                if partially_matching_existing:
                    raise ValueError(
                        "tool call identity cannot be resolved from partial identity"
                    )
            if existing_tool_call is None:
                # 创建新的 tool_call
                new_tool_call = ToolCall(
                    index=delta_tool_call.index,
                    id=delta_tool_call.id,
                    type=delta_tool_call.type,
                    function=FunctionCall(
                        name=(
                            delta_tool_call.function.name
                            if delta_tool_call.function
                            else None
                        ),
                        arguments=(
                            delta_tool_call.function.arguments
                            if delta_tool_call.function
                            else None
                        ),
                    ),
                )
                existing_tool_calls.append(new_tool_call)
            else:
                # 增量更新现有的 tool_call
                if (
                    delta_tool_call.index is not None
                    and existing_tool_call.index is not None
                    and delta_tool_call.index != existing_tool_call.index
                ):
                    raise ValueError("tool call delta has conflicting index and id")
                if (
                    bool(delta_tool_call.id)
                    and bool(existing_tool_call.id)
                    and delta_tool_call.id != existing_tool_call.id
                ):
                    raise ValueError("tool call delta has conflicting index and id")
                if existing_tool_call.index is None:
                    existing_tool_call.index = delta_tool_call.index
                if not existing_tool_call.id:
                    existing_tool_call.id = delta_tool_call.id
                if (
                    delta_tool_call.type
                    and existing_tool_call.type
                    and delta_tool_call.type != existing_tool_call.type
                ):
                    raise ValueError("conflicting tool call type in response stream")
                if delta_tool_call.type and not existing_tool_call.type:
                    existing_tool_call.type = delta_tool_call.type
                if delta_tool_call.function:
                    if existing_tool_call.function is None:
                        existing_tool_call.function = FunctionCall(
                            name=delta_tool_call.function.name,
                            arguments=delta_tool_call.function.arguments,
                        )
                    else:
                        if delta_tool_call.function.name:
                            if (
                                existing_tool_call.function.name
                                and existing_tool_call.function.name
                                != delta_tool_call.function.name
                            ):
                                raise ValueError(
                                    "conflicting tool call function name in response stream"
                                )
                            if not existing_tool_call.function.name:
                                existing_tool_call.function.name = (
                                    delta_tool_call.function.name
                                )
                        if delta_tool_call.function.arguments:
                            if existing_tool_call.function.arguments is None:
                                existing_tool_call.function.arguments = (
                                    delta_tool_call.function.arguments
                                )
                            else:
                                existing_tool_call.function.arguments += (
                                    delta_tool_call.function.arguments
                                )
        return existing_tool_calls

    @staticmethod
    async def _collect_complete_response(
        choice_generator: Optional[AsyncGenerator[StreamResponseObject, None]],
        debug_info: Optional[DebugInfo],
        tokenizer: Optional[Any] = None,
        model_name: str = "",
        response_metadata: Optional[ChatCompletionResponseMetadata] = None,
    ) -> ChatCompletionResponse:
        response_metadata = (
            response_metadata or create_chat_completion_response_metadata()
        )
        all_choices_by_index: Dict[int, ChatCompletionResponseChoice] = {}
        usage = None
        aux_info = None
        extra_outputs = None
        async for response in choice_generator:
            choice_indexes = [choice.index for choice in response.choices]
            if any(
                type(index) is not int or index < 0 for index in choice_indexes
            ):
                raise ValueError(
                    "choice index must be a non-negative integer: "
                    f"{choice_indexes}"
                )
            if len(choice_indexes) != len(set(choice_indexes)):
                raise ValueError(
                    "response chunk contains duplicate choice indexes: "
                    f"{choice_indexes}"
                )

            for choice in response.choices:
                accumulated_choice = all_choices_by_index.get(choice.index)
                if accumulated_choice is None:
                    accumulated_choice = ChatCompletionResponseChoice(
                        index=choice.index,
                        message=ChatMessage(
                            role=RoleEnum.assistant,
                            content=None,
                        ),
                    )
                    all_choices_by_index[choice.index] = accumulated_choice

                delta = choice.delta
                if delta.content is not None:
                    if accumulated_choice.message.content is None:
                        accumulated_choice.message.content = delta.content or None
                    else:
                        accumulated_choice.message.content += delta.content
                if delta.reasoning_content is not None:
                    if accumulated_choice.message.reasoning_content is None:
                        accumulated_choice.message.reasoning_content = (
                            delta.reasoning_content or None
                        )
                    else:
                        accumulated_choice.message.reasoning_content += (
                            delta.reasoning_content
                        )
                accumulated_choice.message.role = (
                    delta.role or accumulated_choice.message.role
                )
                accumulated_choice.message.function_call = (
                    OpenaiEndpoint._merge_function_call(
                        accumulated_choice.message.function_call,
                        delta.function_call,
                    )
                )
                accumulated_choice.message.tool_calls = (
                    OpenaiEndpoint._merge_tool_calls(
                        accumulated_choice.message.tool_calls,
                        delta.tool_calls,
                    )
                )
                accumulated_choice.finish_reason = (
                    choice.finish_reason or accumulated_choice.finish_reason
                )
                if choice.logprobs is not None:
                    if accumulated_choice.logprobs is None:
                        accumulated_choice.logprobs = choice.logprobs.model_copy(
                            deep=True
                        )
                    else:
                        for field_name in ("content", "refusal"):
                            delta_logprobs = getattr(choice.logprobs, field_name)
                            if delta_logprobs:
                                accumulated_logprobs = getattr(
                                    accumulated_choice.logprobs, field_name
                                )
                                if accumulated_logprobs is None:
                                    accumulated_logprobs = []
                                    setattr(
                                        accumulated_choice.logprobs,
                                        field_name,
                                        accumulated_logprobs,
                                    )
                                accumulated_logprobs.extend(
                                    item.model_copy(deep=True)
                                    for item in delta_logprobs
                                )
            if response.usage is not None:
                usage = response.usage.model_copy(deep=True)
            if response.aux_info is not None:
                aux_info = copy.deepcopy(response.aux_info)
            if response.extra_outputs is not None:
                extra_outputs = response.extra_outputs.model_copy(deep=True)

        for accumulated_choice in all_choices_by_index.values():
            tool_calls = accumulated_choice.message.tool_calls
            if not tool_calls:
                continue
            if len(tool_calls) > 1 and any(
                tool_call.index is None for tool_call in tool_calls
            ):
                raise ValueError(
                    "multiple tool calls cannot be ordered with a missing index"
                )
            if all(tool_call.index is not None for tool_call in tool_calls):
                accumulated_choice.message.tool_calls = sorted(
                    tool_calls, key=lambda tool_call: tool_call.index
                )

        all_choices = [
            all_choices_by_index[index] for index in sorted(all_choices_by_index)
        ]

        if usage == None:
            logging.warning(f"No usage returned from stream response. use empty value.")
            usage = UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0)

        if (
            debug_info is not None
            and extra_outputs is not None
            and extra_outputs.output_ids is not None
        ):
            debug_info.output_ids = extra_outputs.output_ids
            if tokenizer:
                debug_info.raw_output = [
                    tokenizer.decode(output_ids)
                    for output_ids in extra_outputs.output_ids
                ]

        return ChatCompletionResponse(
            id=response_metadata.id,
            created=response_metadata.created,
            choices=all_choices,
            usage=usage,
            aux_info=aux_info,
            model=model_name,
            debug_info=debug_info,
            extra_outputs=extra_outputs,
        )

    @staticmethod
    def _complete_stream_response(
        choice_generator: AsyncGenerator[StreamResponseObject, None],
        debug_info: Optional[DebugInfo],
        tokenizer: Optional[Any] = None,
        model_name: str = "",
        close_dependencies: Optional[CloseDependencyRegistry] = None,
    ) -> CompleteResponseAsyncGenerator:
        response_metadata = create_chat_completion_response_metadata()

        async def response_generator():
            debug_info_responded = False
            try:
                async for response in choice_generator:
                    output = None
                    if (
                        debug_info is not None
                        and response.extra_outputs is not None
                        and response.extra_outputs.output_ids is not None
                    ):
                        output = DebugInfo()
                        output.output_ids = response.extra_outputs.output_ids
                        output.raw_output = [
                            tokenizer.decode(output_ids)
                            for output_ids in response.extra_outputs.output_ids
                        ]

                    yield ChatCompletionStreamResponse(
                        id=response_metadata.id,
                        created=response_metadata.created,
                        model=model_name,
                        choices=response.choices,
                        usage=response.usage,
                        aux_info=response.aux_info,
                        debug_info=debug_info if not debug_info_responded else output,
                        extra_outputs=response.extra_outputs,
                    )
                    debug_info_responded = True
            finally:
                await choice_generator.aclose()

        complete_response_collect_func = partial(
            OpenaiEndpoint._collect_complete_response,
            debug_info=debug_info,
            tokenizer=tokenizer,
            model_name=model_name,
            response_metadata=response_metadata,
        )
        return CompleteResponseAsyncGenerator(
            response_generator(),
            complete_response_collect_func,
            close_dependencies=close_dependencies or (),
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
        renderer._update_request_from_rendered_prompt(
            chat_request, rendered_input.rendered_prompt
        )
        return rendered_input

    def chat_completion(
        self,
        request_id: int,
        chat_request: ChatCompletionRequest,
        raw_request: Request,
        request_deadline_anchor: Optional[RequestDeadlineAnchor] = None,
    ) -> CompleteResponseAsyncGenerator:
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = self.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)

        # 生成式推荐：chat 链路同样需要从 rendered_prompt 解析已曝光商品并填充
        # banned_combo_token_ids。函数内部做了开关与空值短路，对非推荐场景零侵入。
        parse_and_fill_banned_combo(
            rendered_input.rendered_prompt, generate_config, self.tokenizer
        )

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

        close_dependencies = CloseDependencyRegistry()
        choice_generator = renderer.generate_choice(
            request_id,
            rendered_input.input_ids,
            mm_inputs,
            generate_config,
            self.backend_rpc_server_visitor,
            chat_request,
            request_deadline_anchor,
            close_dependencies=close_dependencies,
        )

        return self._complete_stream_response(
            choice_generator,
            debug_info,
            self.tokenizer,
            chat_request.model or self.model_name,
            close_dependencies,
        )

    def _prepare_chat_input(
        self,
        request_id: int,
        chat_request,
        request_deadline_anchor: Optional[RequestDeadlineAnchor] = None,
    ):
        import torch

        from rtp_llm.utils.base_model_datatypes import GenerateInput

        rendered_input = self.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(
                generate_config.sp_advice_prompt
            )

        input_id_tensor = torch.Tensor(rendered_input.input_ids).int().unsqueeze(0)
        gen_input = GenerateInput(
            request_id=request_id,
            token_ids=input_id_tensor,
            mm_inputs=rendered_input.multimodal_inputs,
            generate_config=generate_config,
            tokenizer=self.tokenizer,
        )
        if request_deadline_anchor is not None:
            initialize_request_deadlines(
                gen_input,
                request_deadline_anchor.monotonic_s,
                request_deadline_anchor.unix_ms,
            )
        return gen_input, generate_config

    async def _render_single_output(self, outputs, chat_request, generate_config):
        """Render a single GenerateOutputs into a ChatCompletionResponse.
        Only non-streaming mode is supported for batch inference."""
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )

        async def _single_output_gen(out):
            yield out

        merged_gen = await renderer._merge_non_streaming_outputs(
            _single_output_gen(outputs)
        )
        choice_generator = renderer.render_response_stream(
            merged_gen, chat_request, generate_config
        )
        return await self._collect_complete_response(
            choice_generator,
            None,
            self.tokenizer,
            chat_request.model or self.model_name,
        )

    async def batch_chat_completion(
        self,
        base_request_id: int,
        batch_request,
        request_deadline_anchor: Optional[RequestDeadlineAnchor] = None,
    ) -> list:
        inputs = []
        all_configs = []
        for i, chat_request in enumerate(batch_request.requests):
            if chat_request.stream:
                raise ValueError(
                    f"batch chat completion does not support streaming (request index {i})"
                )
            chat_request.stream = False
            gen_input, generate_config = self._prepare_chat_input(
                base_request_id + i, chat_request, request_deadline_anchor
            )
            generate_config.is_streaming = False
            inputs.append(gen_input)
            all_configs.append(generate_config)

        batch_outputs = await self.backend_rpc_server_visitor.batch_enqueue(inputs)

        responses = []
        for i, outputs in enumerate(batch_outputs):
            complete_response = await self._render_single_output(
                outputs, batch_request.requests[i], all_configs[i]
            )
            responses.append(complete_response)

        return responses

    def chat_render(self, chat_request: ChatCompletionRequest) -> DebugInfo:
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = renderer.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)
        debug_info = self._get_debug_info(renderer, rendered_input, generate_config)
        return debug_info
