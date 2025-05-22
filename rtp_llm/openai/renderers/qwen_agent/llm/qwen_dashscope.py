import os
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional, Union, Literal

import dashscope

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, Message
from qwen_agent.llm.text_base import BaseTextChatModel
from qwen_agent.log import logger
# region youmi
import copy
from qwen_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.utils.utils import has_chinese_messages, merge_generate_cfgs
from qwen_agent.llm.base import _truncate_input_messages_roughly
# end region


@register_llm('qwen_dashscope')
class QwenChatAtDS(BaseTextChatModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'qwen-max'
        initialize_dashscope(cfg)

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = [msg.model_dump() for msg in messages]
        logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        response = dashscope.Generation.call(
            self.model,
            messages=messages,  # noqa
            result_format='message',
            stream=True,
            **generate_cfg)
        if delta_stream:
            return self._delta_stream_output(response)
        else:
            return self._full_stream_output(response)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = [msg.model_dump() for msg in messages]
        logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        response = dashscope.Generation.call(
            self.model,
            messages=messages,  # noqa
            result_format='message',
            stream=False,
            **generate_cfg)
        if response.status_code == HTTPStatus.OK:
            return [Message(ASSISTANT, response.output.choices[0].message.content)]
        else:
            raise ModelServiceError(code=response.code, message=response.message)

    def _continue_assistant_response(
        self,
        messages: List[Message],
        generate_cfg: dict,
        stream: bool,
    ) -> Iterator[List[Message]]:
        prompt = self._build_text_completion_prompt(messages)
        logger.debug(f'LLM Input:\n{pformat(prompt, indent=2)}')
        response = dashscope.Generation.call(
            self.model,
            prompt=prompt,  # noqa
            result_format='message',
            stream=True,
            use_raw_prompt=True,
            **generate_cfg)
        it = self._full_stream_output(response)
        if stream:
            return it  # streaming the response
        else:
            *_, final_response = it  # return the final response without streaming
            return final_response

    # region youmi generate completion prompt
    def generate_completion_prompt(
        self,
        messages: List[Union[Message, Dict]],
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[Dict] = None
    ) -> str:
        """ copy from LLM chat interface.

        Args:
            messages: Inputted messages.
            functions: Inputted functions for function calling. OpenAI format supported.
        Returns:
            the generated prompt
        """
        generate_cfg = merge_generate_cfgs(base_generate_cfg=self.generate_cfg, new_generate_cfg=extra_generate_cfg)
        if 'lang' in generate_cfg:
            lang: Literal['en', 'zh'] = generate_cfg.pop('lang')
        else:
            lang: Literal['en', 'zh'] = 'zh' if has_chinese_messages(messages) else 'en'

        messages = copy.deepcopy(messages)

        _return_message_type = 'dict'
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'
        messages = new_messages

        if messages[0].role != SYSTEM:
            messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages

        # Not precise. It's hard to estimate tokens related with function calling and multimodal items.
        messages = _truncate_input_messages_roughly(
            messages=messages,
            max_tokens=generate_cfg.pop('max_input_tokens', DEFAULT_MAX_INPUT_TOKENS),
        )

        messages = self._preprocess_messages(messages, lang=lang)

        if 'function_choice' in generate_cfg:
            fn_choice = generate_cfg['function_choice']
            valid_fn_choices = [f.get('name', f.get('name_for_model', None)) for f in (functions or [])]
            valid_fn_choices = ['auto', 'none'] + [f for f in valid_fn_choices if f]
            if fn_choice not in valid_fn_choices:
                raise ValueError(f'The value of function_choice must be one of the following: {valid_fn_choices}. '
                                 f'But function_choice="{fn_choice}" is received.')
            if fn_choice == 'auto':
                del generate_cfg['function_choice']
            if fn_choice == 'none':
                raise NotImplementedError('Not implemented function_choice="none" yet.')  # TODO:

        if functions:
            fncall_mode = True
        else:
            fncall_mode = False
            for k in ['parallel_function_calls', 'function_choice']:
                if k in generate_cfg:
                    del generate_cfg[k]

        if fncall_mode:
            messages = self._prepend_fncall_system(messages, functions, lang=lang)
            prompt = self._build_text_completion_prompt(messages)
            return prompt
        else:
            prompt = self._build_text_completion_prompt(messages)
            return prompt
    # end region
    

    @staticmethod
    def _build_text_completion_prompt(messages: List[Message]) -> str:
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        if messages[0].role == SYSTEM:
            sys = messages[0].content
            assert isinstance(sys, str)
            prompt = f'{im_start}{SYSTEM}\n{sys}{im_end}'
        else:
            prompt = f'{im_start}{SYSTEM}\n{DEFAULT_SYSTEM_MESSAGE}{im_end}'
        if messages[-1].role != ASSISTANT:
            messages.append(Message(ASSISTANT, ''))
        for msg in messages:
            assert isinstance(msg.content, str)
            if msg.role == USER:
                query = msg.content.lstrip('\n').rstrip()
                prompt += f'\n{im_start}{USER}\n{query}{im_end}'
            elif msg.role == ASSISTANT:
                response = msg.content.lstrip('\n').rstrip()
                prompt += f'\n{im_start}{ASSISTANT}\n{response}{im_end}'
        assert prompt.endswith(im_end)
        prompt = prompt[:-len(im_end)]
        return prompt

    @staticmethod
    def _delta_stream_output(response) -> Iterator[List[Message]]:
        last_len = 0
        delay_len = 5
        in_delay = False
        text = ''
        for chunk in response:
            if chunk.status_code == HTTPStatus.OK:
                text = chunk.output.choices[0].message.content
                if (len(text) - last_len) <= delay_len:
                    in_delay = True
                    continue
                else:
                    in_delay = False
                    real_text = text[:-delay_len]
                    now_rsp = real_text[last_len:]
                    yield [Message(ASSISTANT, now_rsp)]
                    last_len = len(real_text)
            else:
                raise ModelServiceError(code=chunk.code, message=chunk.message)
        if text and (in_delay or (last_len != len(text))):
            yield [Message(ASSISTANT, text[last_len:])]

    @staticmethod
    def _full_stream_output(response) -> Iterator[List[Message]]:
        for chunk in response:
            if chunk.status_code == HTTPStatus.OK:
                yield [Message(ASSISTANT, chunk.output.choices[0].message.content)]
            else:
                raise ModelServiceError(code=chunk.code, message=chunk.message)


def initialize_dashscope(cfg: Optional[Dict] = None) -> None:
    cfg = cfg or {}

    api_key = cfg.get('api_key', '')
    base_http_api_url = cfg.get('base_http_api_url', None)
    base_websocket_api_url = cfg.get('base_websocket_api_url', None)

    if not api_key:
        api_key = os.getenv('DASHSCOPE_API_KEY', 'EMPTY')
    if not base_http_api_url:
        base_http_api_url = os.getenv('DASHSCOPE_HTTP_URL', None)
    if not base_websocket_api_url:
        base_websocket_api_url = os.getenv('DASHSCOPE_WEBSOCKET_URL', None)

    api_key = api_key.strip()
    dashscope.api_key = api_key
    if base_http_api_url is not None:
        dashscope.base_http_api_url = base_http_api_url.strip()
    if base_websocket_api_url is not None:
        dashscope.base_websocket_api_url = base_websocket_api_url.strip()
