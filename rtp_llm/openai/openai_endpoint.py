import itertools
import json
import logging
from functools import partial
from typing import Any, AsyncGenerator, List, Optional

from fastapi import Request

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
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
from rtp_llm.ops import SpecialTokens
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
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
            config.return_all_probs = request.logprobs
        if request.logprobs or request.functions:
            config.is_streaming = True
        config.convert_select_tokens(len(self.tokenizer), self.tokenizer)

        # 处理 logit_bias 参数
        if request.logit_bias is not None:
            # OpenAI API 中 logit_bias 的 key 是字符串类型的 token ID
            # 需要转换为整数类型
            config.logit_bias = {int(k): v for k, v in request.logit_bias.items()}

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
        for delta_tool_call in delta_tool_calls:
            # 查找是否已存在相同 index 的 tool_call
            existing_tool_call = None
            if delta_tool_call.index is not None:
                for existing in existing_tool_calls:
                    if existing.index == delta_tool_call.index:
                        existing_tool_call = existing
                        break
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
                if delta_tool_call.id:
                    existing_tool_call.id = delta_tool_call.id
                if delta_tool_call.type:
                    existing_tool_call.type = delta_tool_call.type
                if delta_tool_call.function:
                    if existing_tool_call.function is None:
                        existing_tool_call.function = FunctionCall(
                            name=delta_tool_call.function.name,
                            arguments=delta_tool_call.function.arguments,
                        )
                    else:
                        if delta_tool_call.function.name:
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
    ) -> ChatCompletionResponse:
        all_choices = []
        usage = None
        aux_info = None
        extra_outputs = None
        async for response in choice_generator:
            if len(response.choices) != len(all_choices):
                if all_choices == []:
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
                        )
                        for i, choice in enumerate(response.choices)
                    ]
                else:
                    raise ValueError(
                        f"response.choices has different length! "
                        f"[{response.choices}] vs [{all_choices}]."
                    )
            else:
                for i in range(len(all_choices)):
                    if all_choices[i].message.content == None:
                        all_choices[i].message.content = (
                            response.choices[i].delta.content or None
                        )
                    else:
                        all_choices[i].message.content += (
                            response.choices[i].delta.content or ""
                        )
                    if all_choices[i].message.reasoning_content == None:
                        all_choices[i].message.reasoning_content = (
                            response.choices[i].delta.reasoning_content or None
                        )
                    else:
                        all_choices[i].message.reasoning_content += (
                            response.choices[i].delta.reasoning_content or ""
                        )
                    all_choices[i].message.role = (
                        response.choices[i].delta.role or all_choices[i].message.role
                    )
                    all_choices[i].message.function_call = (
                        response.choices[i].delta.function_call
                        or all_choices[i].message.function_call
                    )
                    all_choices[i].message.tool_calls = (
                        OpenaiEndpoint._merge_tool_calls(
                            all_choices[i].message.tool_calls,
                            response.choices[i].delta.tool_calls,
                        )
                    )
                    all_choices[i].finish_reason = (
                        response.choices[i].finish_reason
                        or all_choices[i].finish_reason
                    )
                    if all_choices[i].logprobs != None:
                        if response.choices[i].logprobs != None:
                            all_choices[i].logprobs.content += response.choices[
                                i
                            ].logprobs.content
                    else:
                        all_choices[i].logprobs = response.choices[i].logprobs
            usage = response.usage or usage
            aux_info = response.aux_info or aux_info
            extra_outputs = response.extra_outputs or extra_outputs

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
            choices=all_choices,
            usage=usage,
            aux_info=aux_info,
            model="",
            debug_info=debug_info,
            extra_outputs=extra_outputs,
        )

    @staticmethod
    def _complete_stream_response(
        choice_generator: AsyncGenerator[StreamResponseObject, None],
        debug_info: Optional[DebugInfo],
        tokenizer: Optional[Any] = None,
    ) -> CompleteResponseAsyncGenerator:
        async def response_generator():
            debug_info_responded = False

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
                    choices=response.choices,
                    usage=response.usage,
                    aux_info=response.aux_info,
                    debug_info=debug_info if not debug_info_responded else output,
                    extra_outputs=response.extra_outputs,
                )
                debug_info_responded = True

        complete_response_collect_func = partial(
            OpenaiEndpoint._collect_complete_response,
            debug_info=debug_info,
            tokenizer=tokenizer,
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

        return self._complete_stream_response(
            choice_generator, debug_info, self.tokenizer
        )

    def chat_render(self, chat_request: ChatCompletionRequest) -> DebugInfo:
        renderer = (
            self.template_renderer if chat_request.user_template else self.chat_renderer
        )
        rendered_input = renderer.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)
        debug_info = self._get_debug_info(renderer, rendered_input, generate_config)
        return debug_info
