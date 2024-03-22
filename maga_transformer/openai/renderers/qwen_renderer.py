import copy
import json
import re
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator

from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from maga_transformer.tokenizer.tokenization_qwen2 import Qwen2Tokenizer
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, \
    ChatCompletionRequest, RoleEnum, FunctionCall, ChatCompletionResponseStreamChoice, \
    DeltaMessage, FinisheReason, UsageInfo, RendererInfo
from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams, \
    StreamResponseObject, RenderedInputs
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.openai.renderer_factory_register import register_renderer
from maga_transformer.utils.word_util import get_stop_word_slice_list, truncate_response_with_stop_words

QwenTokenizerTypes = Union[QWenTokenizer, Qwen2Tokenizer]

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

@dataclass
class ProcessedOutput:
    output_str: str
    output_token_length: int
    finish_reason: Optional[FinisheReason]

# TODO(wangyin): pass `max_window_size` to here.
def make_context(
    tokenizer: QwenTokenizerTypes,
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
        response_text, response_tokens_part = _tokenize_str(
            "assistant", turn_response
        )
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
    def __init__(self, tokenizer: QwenTokenizerTypes, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        self.add_extra_stop_word_ids([[37763, 367, 25]]) # Observation:

        self.template_chat_renderer: Optional[BasicRenderer] = None
        try:
            if tokenizer.chat_template != None:
                logging.info(f"qwen model has chat_template [{tokenizer.chat_template}], "
                             "which will be used for non-function call dialogue.")
                self.template_chat_renderer = BasicRenderer(tokenizer, renderer_params)
        except AttributeError:
            pass

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        assert (isinstance(self.tokenizer, QwenTokenizerTypes))

        if (self.template_chat_renderer != None) and \
            ((request.functions == None) or (len(request.functions) == 0)):
            return self.template_chat_renderer.render_chat(request)

        query, history, system = self.parse_messages(request.messages, request.functions)
        print(f"parsed query: {query}, history: {history}, system: {system}")
        input_ids = []
        if (query == _TEXT_COMPLETION_CMD):
            input_ids = self.text_complete_last_message(history)
        else:
            assert (isinstance(query, str))
            input_ids = make_context(self.tokenizer, query, history, system)[1]
        return RenderedInputs(input_ids=input_ids)

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

        return self.tokenizer.encode(prompt)

    def parse_messages(
            self,
            messages: List[ChatMessage],
            functions: Optional[List[GPTFunctionDefinition]] = None
    ):
        if all(m.role != "user" for m in messages):
            raise ValueError("At least one message must be from user.")

        messages = copy.deepcopy(messages)
        if messages[0].role == 'system':
            system = messages.pop(0).content.lstrip('\n').rstrip()
        else:
            system = 'You are a helpful assistant.'

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
            instruction = (REACT_INSTRUCTION.format(
                tools_text=tools_text,
                tools_name_text=tools_name_text,
            ).lstrip('\n').rstrip())
        else:
            instruction = ''

        messages_with_fncall = messages
        messages = []
        for m_idx, m in enumerate(messages_with_fncall):
            role, content, func_call = m.role, m.content, m.function_call
            content = content or ""
            content = content.lstrip("\n").rstrip()
            if role == "function":
                if (len(messages) == 0) or (messages[-1].role != "assistant"):
                    raise ValueError(f"Invalid request: Expecting role assistant before role function.")
                messages[-1].content += f'\nObservation: {content}'
                if m_idx == len(messages_with_fncall) - 1:
                    # add a prefix for text completion
                    messages[-1].content += '\nThought:'
            elif role == 'assistant':
                if len(messages) == 0:
                    raise ValueError(f"Invalid request: Expecting role user before role assistant.")
                if func_call is None:
                    if functions:
                        content = f'Thought: I now know the final answer.\nFinal Answer: {content}'
                else:
                    f_name, f_args = func_call.name, func_call.arguments
                    if not content.startswith('Thought:'):
                        content = f'Thought: {content}'
                    content = f'{content}\nAction: {f_name}\nAction Input: {f_args}'
                if messages[-1].role == 'user':
                    messages.append(
                        ChatMessage(role=RoleEnum.assistant, content=content.lstrip('\n').rstrip())
                    )
                else:
                    messages[-1].content += '\n' + content
            elif role == 'user':
                messages.append(
                    ChatMessage(role='user',content=content.lstrip('\n').rstrip()))
            else:
                raise ValueError(f"Invalid request: Incorrect role {role}.")

        query = _TEXT_COMPLETION_CMD
        if messages[-1].role == 'user':
            query = messages[-1].content
            messages = messages[:-1]

        history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
        for i in range(0, len(messages), 2):
            if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
                usr_msg = messages[i].content.lstrip('\n').rstrip()
                bot_msg = messages[i + 1].content.lstrip('\n').rstrip()
                if instruction and (i == len(messages) - 2):
                    usr_msg = f'{instruction}\n\nQuestion: {usr_msg}'
                    instruction = ''
                history.append([usr_msg, bot_msg])
            else:
                raise ValueError(
                    "Invalid request: Expecting exactly one user (or function) role before every assistant role."
                )
        if instruction:
            assert query is not _TEXT_COMPLETION_CMD
            query = f'{instruction}\n\nQuestion: {query}'
        return query, history, system

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
        logging.info(f"parsed function from response: [{response}]: {func_name}, {func_args}")
        if func_name:
            return DeltaMessage(
                content=response[:i],
                function_call=FunctionCall(name=func_name, arguments=func_args),
            )
        return None
        # z = response.rfind("\nFinal Answer: ")
        # if z >= 0:
        #     response = response[z + len("\nFinal Answer: ") :]

    def _process_output_ids_tensor(
            self, input_length, output_ids_tensor: torch.Tensor, finished: bool = False
    ) -> ProcessedOutput:
        output_ids_tensor = output_ids_tensor.cpu().reshape([-1])
        # TODO(wangyin): This slicing shouldn't be done here.
        # model should return output length, ids should be sliced with output length.
        output_ids = output_ids_tensor[output_ids_tensor != self.eos_token_id].tolist()
        finish_reason = self._check_finish_reason(output_ids, input_length) if finished else None

        output_length = len(output_ids)
        output_ids = self._remove_stop_word_ids(output_ids)
        output_str = self.tokenizer.decode(output_ids)
        output_str = output_str.strip(u'\uFFFD')

        for stop_word in self.stop_words_list:
            output_str = output_str.replace(stop_word, "")
        return ProcessedOutput(output_str, output_length, finish_reason)

    async def render_response_stream(
            self,
            output_generator: AsyncGenerator[GenerateOutput, None],
            request: ChatCompletionRequest,
            generate_config: GenerateConfig,
            input_token_length: int,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        index = 0
        output_string = ""
        output_length = 0
        responded_string = ""
        responded_length = 0
        output_token_length = 0
        finish_reason: Optional[FinisheReason] = None
        generating_function_call = False
        stop_word_slice_list = get_stop_word_slice_list(generate_config.stop_words_str)

        async for output in output_generator:
            if output_token_length == 0:
                yield StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            role=RoleEnum.assistant,
                        ),
                    )]
                )

            processed_output = self._process_output_ids_tensor(
                input_token_length, output.output_ids, output.finished)
            output_string = processed_output.output_str.strip()
            output_length = len(processed_output.output_str)
            finish_reason = processed_output.finish_reason
            output_token_length = processed_output.output_token_length

            if (output_string.endswith("\nAction:")):
                generating_function_call = True
                continue

            if (generating_function_call):
                continue

            if (output_length > responded_length + len('\nAction:')):
                delta_string = output_string[responded_length : output_length - len('\nAction:')]
                trunc_string = truncate_response_with_stop_words(delta_string, stop_word_slice_list)
                if trunc_string != delta_string:
                    continue
                responded_string = output_string[: output_length - len('\nAction:')]
                responded_length = len(responded_string)

                yield StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            content=delta_string,
                        ),
                    )],
                    usage=UsageInfo(
                        prompt_tokens=input_token_length,
                        total_tokens=input_token_length + output_token_length,
                        completion_tokens=output_token_length
                    )
                )

        if (generating_function_call):
            function_message = self._parse_function_response(output_string[responded_length:])
            if (function_message == None):
                logging.warn(f"output [{output_string}] failed to parse function from [{responded_length}]. "
                                "regarded as normal output.")
            else:
                finish_reason = FinisheReason.function_call
                responded_string = output_string
                responded_length = output_length
                yield StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=function_message,
                        finish_reason=finish_reason,
                    )],
                    usage=UsageInfo(
                        prompt_tokens=input_token_length,
                        total_tokens=input_token_length + output_token_length,
                        completion_tokens=output_token_length
                    )
                )

        if finish_reason == None:
            logging.debug(f"output [{responded_string}] found no stop reason! use stop as default.")
            finish_reason = FinisheReason.stop

        if responded_length < output_length:
            index += 1
            yield StreamResponseObject(
                choices=[ChatCompletionResponseStreamChoice(
                    index=index,
                    delta=DeltaMessage(
                        content=truncate_response_with_stop_words(output_string[responded_length:], generate_config.stop_words_str),
                    ),
                )],
                usage=UsageInfo(
                    prompt_tokens=input_token_length,
                    total_tokens=input_token_length + output_token_length,
                    completion_tokens=output_token_length
                )
            )

        yield StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(
                index=index + 1,
                delta=DeltaMessage(
                    content="",
                ),
                finish_reason=finish_reason
            )],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length
            ),
            aux_info=output.aux_info if request.aux_info else None
        )

    def get_renderer_info(self) -> RendererInfo:
        renderer_info = super().get_renderer_info()
        if self.template_chat_renderer:
            renderer_info.template = self.template_chat_renderer.chat_template
        return renderer_info

register_renderer('qwen', QwenRenderer)
register_renderer('qwen_7b', QwenRenderer)
register_renderer('qwen_13b', QwenRenderer)
register_renderer('qwen_1b8', QwenRenderer)
register_renderer('qwen_2', QwenRenderer)
