import copy
import functools
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from typing_extensions import override

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    GPTFunctionDefinition,
    RendererInfo,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    RendererParams,
    StreamStatus,
    StreamStatusSync,
    ThinkStatus,
)
from rtp_llm.openai.renderers.qwen_reasoning_tool_renderer import (
    QwenReasoningToolRenderer,
)
from rtp_llm.utils.base_model_datatypes import GenerateOutput
from rtp_llm.utils.word_util import truncate_response_with_stop_words

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

DUMMY_THOUGHT = {
    "en": "\nThought: I now know the final answer.\nFinal answer: ",
    "zh": "\nThought: 我会作答了。\nFinal answer: ",
}

_TEXT_COMPLETION_CMD = object()


class QwenStreamStatus(StreamStatus):
    generating_function_call: bool = False
    total_output_string: str = ""

    def __init__(self, request: ChatCompletionRequest):
        super().__init__(request)

    def update_result(self):
        self.last_token_length = len(self.output_ids) - len(self.last_output_ids)
        self.last_output_ids = self.output_ids
        self.responded_string = self.total_output_string[: -len("\nAction:")]

    @property
    def responded_length(self):
        return len(self.responded_string)

    @property
    def output_length(self):
        return len(self.total_output_string)

    def check_stop_reason(self):
        if self.finish_reason == None:
            logging.debug(
                f"output [{self.responded_string}] found no stop reason! use stop as default."
            )
            self.finish_reason = FinisheReason.stop


class QwenStreamStatusSync(StreamStatusSync):
    generating_function_call: bool = False
    total_output_string: str = ""

    def __init__(self, request: ChatCompletionRequest):
        super().__init__(request)

    def update_result(self):
        self.responded_string = self.total_output_string[: -len("\nAction:")]

    @property
    def responded_length(self):
        return len(self.responded_string)

    @property
    def output_length(self):
        return len(self.total_output_string)

    def check_stop_reason(self):
        if self.finish_reason == None:
            logging.debug(
                f"output [{self.responded_string}] found no stop reason! use stop as default."
            )
            self.finish_reason = FinisheReason.stop


@dataclass
class ProcessedOutput:
    output_str: str
    output_token_length: int
    finish_reason: Optional[FinisheReason]


# TODO(wangyin): pass `max_window_size` to here.
def make_context(
    tokenizer: BaseTokenizer,
    query: str,
    history: List[Tuple[str, str]] = [],
    system: str = "",
    max_window_size: int = 6144,
):
    history = copy.deepcopy(history)
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return f"{role}\n{content}", tokenizer.encode(
            role, allowed_special=set()
        ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_text, system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

    raw_text = ""
    context_tokens = []

    for turn_query, turn_response in reversed(history):
        query_text, query_tokens_part = _tokenize_str("user", turn_query)
        query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
        response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
        response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

        next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
        prev_chat = (
            f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
        )

        current_context_size = (
            len(system_tokens) + len(next_context_tokens) + len(context_tokens)
        )
        if current_context_size < max_window_size:
            context_tokens = next_context_tokens + context_tokens
            raw_text = prev_chat + raw_text
        else:
            break

    context_tokens = system_tokens + context_tokens
    raw_text = f"{im_start}{system_text}{im_end}" + raw_text
    context_tokens += (
        nl_tokens
        + im_start_tokens
        + _tokenize_str("user", query)[1]
        + im_end_tokens
        + nl_tokens
        + im_start_tokens
        + tokenizer.encode("assistant")
        + nl_tokens
    )
    raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
    return raw_text, context_tokens


class QwenRenderer(CustomChatRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        self.add_extra_stop_word_ids([[37763, 367, 25], [151643]])  # Observation:

        self.qwen_reasoning_tool_renderer = QwenReasoningToolRenderer(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )

        self.template_chat_renderer: Optional[BasicRenderer] = None
        try:
            if tokenizer.chat_template != None:
                logging.info(
                    f"qwen model has chat_template [{tokenizer.chat_template}], "
                    "which will be used for non-function call dialogue."
                )
                self.template_chat_renderer = BasicRenderer(
                    tokenizer,
                    renderer_params,
                    generate_env_config,
                    render_config,
                    ckpt_path,
                    misc_config,
                    vit_config,
                )
        except AttributeError:
            pass

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        if request.tools or self.in_think_mode(request):
            return self.qwen_reasoning_tool_renderer.render_chat(request)

        if (self.template_chat_renderer != None) and (
            (request.functions == None) or (len(request.functions) == 0)
        ):
            return self.template_chat_renderer.render_chat(request)

        query, history, system = self.parse_messages(
            request.messages, request.functions
        )
        logging.debug(
            "parsed query: %s, history: %s, system: %s", query, history, system
        )
        prompt = ""
        input_ids = []
        if query == _TEXT_COMPLETION_CMD:
            prompt = self.text_complete_last_message(history)
            input_ids = self.tokenizer.encode(prompt)
        else:
            assert isinstance(query, str)
            prompt, input_ids = make_context(self.tokenizer, query, history, system)
        return RenderedInputs(input_ids=input_ids, rendered_prompt=prompt)

    def text_complete_last_message(self, history):
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
        for i, (query, response) in enumerate(history):
            query = query.lstrip("\n").rstrip()
            response = response.lstrip("\n").rstrip()
            prompt += f"\n{im_start}user\n{query}{im_end}"
            prompt += f"\n{im_start}assistant\n{response}{im_end}"
        prompt = prompt[: -len(im_end)]

        return prompt

    def parse_messages(
        self,
        messages: List[ChatMessage],
        functions: Optional[List[GPTFunctionDefinition]] = None,
    ):
        if all(m.role != "user" for m in messages):
            raise ValueError("At least one message must be from user.")

        messages = copy.deepcopy(messages)
        if messages[0].role == "system":
            system = messages.pop(0).content.lstrip("\n").rstrip()
        else:
            system = "You are a helpful assistant."

        if functions:
            tools_text = []
            tools_name_text = []
            for func_info in functions:
                name = func_info.name
                name_m = func_info.name_for_model or name
                name_h = func_info.name_for_human or name
                desc = func_info.description
                desc_m = func_info.description_for_model or desc
                tool = TOOL_DESC.format(
                    name_for_model=name_m,
                    name_for_human=name_h,
                    # Hint: You can add the following format requirements in description:
                    #   "Format the arguments as a JSON object."
                    #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                    description_for_model=desc_m,
                    parameters=json.dumps(func_info.parameters, ensure_ascii=False),
                )
                tools_text.append(tool)
                tools_name_text.append(name_m)
            tools_text = "\n\n".join(tools_text)
            tools_name_text = ", ".join(tools_name_text)
            instruction = (
                REACT_INSTRUCTION.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                )
                .lstrip("\n")
                .rstrip()
            )
        else:
            instruction = ""

        messages_with_fncall = messages
        messages = []
        for m_idx, m in enumerate(messages_with_fncall):
            role, content, func_call = m.role, m.content, m.function_call
            content = content or ""
            content = content.lstrip("\n").rstrip()
            if role == "function":
                if (len(messages) == 0) or (messages[-1].role != "assistant"):
                    raise ValueError(
                        f"Invalid request: Expecting role assistant before role function."
                    )
                messages[-1].content += f"\nObservation: {content}"
                if m_idx == len(messages_with_fncall) - 1:
                    # add a prefix for text completion
                    messages[-1].content += "\nThought:"
            elif role == "assistant":
                if len(messages) == 0:
                    raise ValueError(
                        f"Invalid request: Expecting role user before role assistant."
                    )
                if func_call is None:
                    if functions:
                        content = f"Thought: I now know the final answer.\nFinal Answer: {content}"
                else:
                    f_name, f_args = func_call.name, func_call.arguments
                    if not content.startswith("Thought:"):
                        content = f"Thought: {content}"
                    content = f"{content}\nAction: {f_name}\nAction Input: {f_args}"
                if messages[-1].role == "user":
                    messages.append(
                        ChatMessage(
                            role=RoleEnum.assistant,
                            content=content.lstrip("\n").rstrip(),
                        )
                    )
                else:
                    messages[-1].content += "\n" + content
            elif role == "user":
                messages.append(
                    ChatMessage(role="user", content=content.lstrip("\n").rstrip())
                )
            else:
                raise ValueError(f"Invalid request: Incorrect role {role}.")

        query = _TEXT_COMPLETION_CMD
        if messages[-1].role == "user":
            query = messages[-1].content
            messages = messages[:-1]

        history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
        for i in range(0, len(messages), 2):
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                usr_msg = messages[i].content.lstrip("\n").rstrip()
                bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
                if instruction and (i == len(messages) - 2):
                    usr_msg = f"{instruction}\n\nQuestion: {usr_msg}"
                    instruction = ""
                history.append([usr_msg, bot_msg])
            else:
                raise ValueError(
                    "Invalid request: Expecting exactly one user (or function) role before every assistant role."
                )
        if instruction:
            assert query is not _TEXT_COMPLETION_CMD
            query = f"{instruction}\n\nQuestion: {query}"
        return query, history, system

    def in_think_mode(self, request: ChatCompletionRequest):
        if request.disable_thinking():
            return False
        return super().in_think_mode(request)

    @override
    def should_process_think(self, request: ChatCompletionRequest):
        # 留出方法给子类重写, 避免重复的think处理
        return False

    def _parse_function_response(self, response: str) -> Optional[DeltaMessage]:
        func_name, func_args = "", ""
        i = response.rfind("\nAction:")
        j = response.rfind("\nAction Input:")
        k = response.rfind("\nObservation:")
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is omitted by the LLM,
                # because the output text may have discarded the stop word.
                response = response.rstrip() + "\nObservation:"  # Add it back.
            k = response.rfind("\nObservation:")
            func_name = response[i + len("\nAction:") : j].strip()
            func_args = response[j + len("\nAction Input:") : k].strip()
        logging.info(
            f"parsed function from response: [{response}]: {func_name}, {func_args}"
        )
        if func_name:
            return DeltaMessage(
                content=response[:i],
                function_call=FunctionCall(name=func_name, arguments=func_args),
            )
        return None
        # z = response.rfind("\nFinal Answer: ")
        # if z >= 0:
        #     response = response[z + len("\nFinal Answer: ") :]

    async def _update_single_status(
        self,
        status: StreamStatus,
        output: GenerateOutput,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        if status.request.tools or self.in_think_mode(status.request):
            return await self.qwen_reasoning_tool_renderer._update_single_status(
                status,
                output,
                max_new_tokens,
                stop_words_str,
                stop_word_slice_list,
                is_streaming,
            )

        if not isinstance(status, QwenStreamStatus):
            return await super()._update_single_status(
                status,
                output,
                max_new_tokens,
                stop_words_str,
                stop_word_slice_list,
                is_streaming,
            )
        if status.finish_reason != None:
            return await self._create_empty_delta(status.output.aux_info)
        status.update_output(
            output,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        status.total_output_string = self.tokenizer.decode(status.output_ids).strip()
        if (len(status.total_output_string)) and (
            "\uFFFD" == status.total_output_string[-1]
        ):
            return await self._create_empty_delta(output.aux_info)
            # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if status.total_output_string.endswith("\nAction:"):
            status.generating_function_call = True
            return await self._create_empty_delta(output.aux_info)
        if status.generating_function_call:
            return await self._create_empty_delta(output.aux_info)

        # Process stop words on total_output_string
        status.total_output_string, _ = self._process_stop_words(
            status.total_output_string,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
            status,
        )

        if len(status.total_output_string) > status.responded_length + len("\nAction:"):
            status.delta_output_string = status.total_output_string[
                status.responded_length : status.output_length - len("\nAction:")
            ]

            # Check delta for partial stop word buffering
            _, should_buffer = self._process_stop_words(
                status.delta_output_string,
                stop_words_str,
                stop_word_slice_list,
                is_streaming,
                status,
            )

            if should_buffer:
                return await self._create_empty_delta(output.aux_info)

            # Build delta output
            if len(status.delta_output_string) > 0:
                status.update_result()
                return OutputDelta(
                    status.delta_output_string,
                    await self._generate_log_probs(status, output),
                    status.input_token_length,
                    status.output_token_length,
                    status.reuse_length,
                )
        return await self._create_empty_delta(output.aux_info)

    # override
    async def _create_status_list(
        self, n: int, request: ChatCompletionRequest
    ) -> List[StreamStatus]:
        if request.tools or self.in_think_mode(request):
            return await self.qwen_reasoning_tool_renderer._create_status_list(
                n, request
            )
        if request.functions and (len(request.functions) > 0):
            return [QwenStreamStatus(request) for _ in range(n)]
        else:
            return [StreamStatus(request) for _ in range(n)]

    # override
    async def _flush_buffer(
        self,
        buffer_list: List[StreamStatus],
        stop_words_str: List[str],
        is_streaming: bool,
        think_status: ThinkStatus,
    ):
        if buffer_list[0].request.tools or self.in_think_mode(buffer_list[0].request):
            return await self.qwen_reasoning_tool_renderer._flush_buffer(
                buffer_list, stop_words_str, is_streaming, think_status
            )

        if not isinstance(buffer_list[0], QwenStreamStatus):
            return await super()._flush_buffer(
                buffer_list, stop_words_str, is_streaming, think_status
            )
        output_items: List[OutputDelta] = []
        for status in buffer_list:
            if status.generating_function_call:
                function_message = self._parse_function_response(
                    status.total_output_string[status.responded_length :]
                )
                if function_message == None:
                    logging.warning(
                        f"output [{status.total_output_string}] failed to parse function from [{status.responded_length}]. "
                        "regarded as normal output."
                    )
                    function_message = ""
                else:
                    status.finish_reason = FinisheReason.function_call
                output_items.append(
                    OutputDelta(
                        function_message,
                        await self._generate_log_probs(status, status.output),
                        status.input_token_length,
                        status.output_token_length,
                        status.reuse_length,
                    )
                )
            else:
                trunc_string = truncate_response_with_stop_words(
                    status.total_output_string[status.responded_length :],
                    stop_words_str,
                    is_streaming,
                )
                output_items.append(
                    OutputDelta(
                        trunc_string,
                        await self._generate_log_probs(status, status.output),
                        status.input_token_length,
                        status.output_token_length,
                        status.reuse_length,
                    )
                )
        return await self._generate_stream_response(output_items, think_status)

    # override
    def _update_single_status_sync(
        self,
        status: StreamStatusSync,
        input_len,  # output.aux_info
        output_len,  # output.aux_info
        reuse_len,  # output.aux_info
        all_probs: torch.Tensor,
        output_ids: torch.Tensor,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        # function call is disabled when logprobs is required.
        if not isinstance(status, QwenStreamStatusSync):
            return super()._update_single_status_sync(
                status,
                input_len,
                output_len,
                reuse_len,
                all_probs,
                output_ids,
                max_new_tokens,
                stop_words_str,
                stop_word_slice_list,
                is_streaming,
            )
        if status.finish_reason != None:
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)
        status.update_output_sync(
            output_ids,
            input_len,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        status.total_output_string = self.tokenizer.decode(status.output_ids).strip()
        if (len(status.total_output_string)) and (
            "\uFFFD" == status.total_output_string[-1]
        ):
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)
            # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if status.total_output_string.endswith("\nAction:"):
            status.generating_function_call = True
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)
        if status.generating_function_call:
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)

        # Process stop words on total_output_string
        status.total_output_string, _ = self._process_stop_words(
            status.total_output_string,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
            status,
        )

        if len(status.total_output_string) > status.responded_length + len("\nAction:"):
            status.delta_output_string = status.total_output_string[
                status.responded_length : status.output_length - len("\nAction:")
            ]

            # Check delta for partial stop word buffering
            _, should_buffer = self._process_stop_words(
                status.delta_output_string,
                stop_words_str,
                stop_word_slice_list,
                is_streaming,
                status,
            )

            if should_buffer:
                return self._create_empty_delta_sync(input_len, output_len, reuse_len)

            # Build delta output
            if len(status.delta_output_string) > 0:
                status.update_result()
                return OutputDelta(
                    output_str=status.delta_output_string,
                    logprobs=self._generate_log_probs_sync(
                        status, all_probs, output_ids
                    ),
                    input_length=input_len,
                    output_length=output_len,
                    reuse_length=reuse_len,
                )
        return self._create_empty_delta_sync(input_len, output_len, reuse_len)

    # override
    def _create_status_list_sync(self, n: int, body: str) -> List[StreamStatusSync]:
        request = self.getRequest(body)
        if request.logprobs:
            return [StreamStatusSync(request) for _ in range(n)]
        else:
            return [QwenStreamStatusSync(request) for _ in range(n)]

    # override
    def _flush_buffer_sync(
        self,
        buffer_list: List[StreamStatusSync],
        input_len_list,
        output_len_list,
        reuse_len_list,
        all_probs_list,
        output_ids_list,
        stop_words_str: List[str],
        is_streaming: bool,
    ):
        if not isinstance(buffer_list[0], QwenStreamStatusSync):
            return super()._flush_buffer_sync(
                buffer_list,
                input_len_list,
                output_len_list,
                reuse_len_list,
                all_probs_list,
                output_ids_list,
                stop_words_str,
                is_streaming,
            )
        output_items: List[OutputDelta] = []
        for status, input_len, output_len, reuse_len, all_probs, output_ids in zip(
            buffer_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            all_probs_list,
            output_ids_list,
        ):
            if status.generating_function_call:
                function_message = self._parse_function_response(
                    status.total_output_string[status.responded_length :]
                )
                if function_message == None:
                    logging.warning(
                        f"output [{status.total_output_string}] failed to parse function from [{status.responded_length}]. "
                        "regarded as normal output."
                    )
                    function_message = ""
                else:
                    status.finish_reason = FinisheReason.function_call
                output_items.append(
                    OutputDelta(
                        function_message,
                        self._generate_log_probs_sync(status, all_probs, output_ids),
                        input_len,
                        output_len,
                        reuse_len,
                    )
                )
            else:
                trunc_string = truncate_response_with_stop_words(
                    status.total_output_string[status.responded_length :],
                    stop_words_str,
                    is_streaming,
                )
                output_items.append(
                    OutputDelta(
                        trunc_string,
                        self._generate_log_probs_sync(status, all_probs, output_ids),
                        input_len,
                        output_len,
                        reuse_len,
                    )
                )
        return self._generate_stream_response_sync(output_items)

    def get_renderer_info(self) -> RendererInfo:
        renderer_info = super().get_renderer_info()
        if self.template_chat_renderer:
            renderer_info.template = self.template_chat_renderer.chat_template
        return renderer_info


register_renderer("qwen", QwenRenderer)
register_renderer("qwen_7b", QwenRenderer)
register_renderer("qwen_13b", QwenRenderer)
register_renderer("qwen_1b8", QwenRenderer)
register_renderer("qwen_2", QwenRenderer)
register_renderer("qwen_2_moe", QwenRenderer)
