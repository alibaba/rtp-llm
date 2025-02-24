import os
import copy
import sys
import json
import logging
import pathlib
import torch
from typing import Any, Dict, List, Tuple, Optional, Union, NamedTuple

from maga_transformer.config.generate_config import RequestFormat, GenerateConfig
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException

request_id_field_name = "__request_id__"

class Request(NamedTuple):
    request_id: int
    batch_infer: bool
    input_texts: Any
    input_urls: Any
    generate_configs: List[GenerateConfig]
    is_streaming: bool

    @property
    def num_return_sequences(self) -> int:
        return self.generate_configs[0].num_return_sequences

    @property
    def incremental(self) -> bool:
        return self.generate_configs[0].return_incremental

class RequestExtractor:
    def __init__(self, default_generate_config: GenerateConfig):
        self.default_generate_config = default_generate_config

    def extract_request(self, kwargs: Dict[str, Any]) -> Tuple[Request, Dict[str, Any]]:
        generate_config, remain_args = self._format_request(kwargs)
        request = self._get_request(generate_config, remain_args)
        return request, remain_args

    @staticmethod
    def is_streaming(req: Dict[str, Any]):
        return req.get(
            'yield_generator',
            req.get('generation_config',
                    req.get('generate_config', {})
                    ).get('yield_generator', False))

    def _format_generate_config(self, kwargs: Dict[str, Any]) -> Tuple[GenerateConfig, Dict[str, Any]]:
        config_json = kwargs.pop('generate_config', kwargs.pop('generation_config', {}))
        generate_config = copy.deepcopy(self.default_generate_config)
        remain_config_json = generate_config.update_and_pop(config_json)
        remain_kwargs = generate_config.update_and_pop(kwargs)

        def update_optional(key: str, params: List[str]) -> None:
            for source in [remain_config_json, remain_kwargs]:
                for param in params:
                    if param in source:
                        setattr(generate_config, key, source[param])
                        return

        update_optional('return_hidden_states', ['return_hidden_states', 'output_hidden_states'])
        update_optional('return_logits', ['return_logits', 'output_logits'])
        update_optional('return_input_ids', ['return_input_ids', 'output_input_ids'])
        update_optional('max_new_tokens', ['gen_length', 'max_new_tokens'])
        if self.is_streaming(kwargs) or (kwargs.get('stream', False) == True):
            generate_config.is_streaming = True
        return generate_config, remain_kwargs

    def _format_request(self, kwargs: Dict[str, Any]) -> Tuple[GenerateConfig, Dict[str, Any]]:
        generate_config, remain_kwargs = self._format_generate_config(kwargs)
        return self._format_chat_api_messages(generate_config, remain_kwargs)

    def _get_text(self, kwargs: Dict[str,Any]):
        input_texts: Optional[Union[List[str], List[List[Dict[str, str]]]]] = None
        if "prompt_batch" in kwargs:
            input_texts = kwargs.pop('prompt_batch')
            if not isinstance(input_texts, list):
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch input should be list")
        else:
            prompt: Union[str, List[str], List[Dict[str, str]]]
            if 'prompt' in kwargs:
                prompt = kwargs.pop('prompt')
            elif 'text' in kwargs:
                prompt = kwargs.pop('text')
            else:
                raise Exception('can not find prompt or text in request')
            if isinstance(prompt, str):
                input_texts = [prompt]
            # for AutoML format
            elif isinstance(prompt, list) and isinstance(prompt[0], dict):
                input_texts = [prompt]
            else:
                input_texts = prompt
        if input_texts is None:
            raise FtRuntimeException(ExceptionType.NO_PROMPT_ERROR, "not input prompt")
        return input_texts

    def _get_urls(self, input_len: int, kwargs: Dict[str,Any]):
        mm_urls: Optional[Union[List[str], List[List[str]]]] = None
        urls = kwargs.pop('images', kwargs.pop('urls', None))
        if urls is not None and not isinstance(urls, list):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "input urls should be list")
        if "prompt_batch" in kwargs:
            if urls is not None:
                if not isinstance(urls[0], list):
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch urls should be list[list]")
                if len(urls) != input_len:
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch urls and input should have same length")
                mm_urls = urls
            else:
                mm_urls = [[]] * input_len
        else:
            if urls == None or len(urls) == 0:
                mm_urls = [[]] * input_len
            elif len(urls) > 0 and isinstance(urls[0], str):
                mm_urls = [urls]
            else:
                mm_urls = urls
        return mm_urls

    def _get_request_id(self, kwargs: Dict[str,Any]) -> int:
        return kwargs[request_id_field_name]

    def _is_batch(self, kwargs: Dict[str, Any]) -> bool:
         return "prompt_batch" in kwargs

    def _get_adapter(self, generate_config: GenerateConfig, input_len: int) -> List[GenerateConfig]:
        generate_configs: List[GenerateConfig] = [generate_config] * input_len
        adapter_name = generate_config.adapter_name
        if adapter_name != None:
            if (isinstance(adapter_name, str) and input_len != 1) or \
               (isinstance(adapter_name, list) and  input_len != len(adapter_name)):
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "adapter_name is not alignment")
            for i in range(input_len):
                generate_configs[i] = copy.copy(generate_configs[i])
                generate_configs[i].adapter_name = adapter_name[i] if isinstance(adapter_name, list) else adapter_name
        return generate_configs

    def extend_sequences(self, input_texts: Any, input_urls: Any, generate_configs: List[GenerateConfig]):
        return input_texts, input_urls, generate_configs

    def _get_request(self, generate_config: GenerateConfig, kwargs: Dict[str,Any]) -> Request:
        request_id = self._get_request_id(kwargs)
        batch_infer = self._is_batch(kwargs)
        input_texts = self._get_text(kwargs)
        input_urls = self._get_urls(len(input_texts), kwargs)
        generate_configs = self._get_adapter(generate_config, len(input_texts))
        input_texts, input_urls, generate_configs = self.extend_sequences(input_texts, input_urls, generate_configs)
        is_streaming = RequestExtractor.is_streaming(kwargs)
        if generate_config.is_streaming:
            is_streaming = True
        return Request(request_id, batch_infer, input_texts, input_urls, generate_configs, is_streaming)

    def _format_chat_api_messages(self, generate_config: GenerateConfig, kwargs: Dict[str, Any]) -> Tuple[GenerateConfig, Dict[str, Any]]:
        if 'messages' in kwargs:
            assert 'prompt' not in kwargs
            messages = kwargs.pop('messages')
            assert isinstance(messages, list)
            kwargs['prompt'] = messages
            generate_config.request_format = RequestFormat.CHAT_API

        prompt = kwargs.get('prompt', None)
        functions = kwargs.get('functions', None)
        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            generate_config.request_format = RequestFormat.CHAT_API

        if generate_config.request_format == RequestFormat.CHAT_API:
            if isinstance(prompt, str):
                prompt = json.loads(prompt, strict=False)
            if prompt == None:
                prompt_batch = kwargs.pop('prompt_batch', None)
                if not isinstance(prompt_batch, list):
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt_batch should be list")
                if len(prompt_batch) > 1:
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt_batch does not support batch size > 1 now.")
                prompt = prompt_batch[0]
            if prompt == None:
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "No prompt!")
            assert isinstance(prompt, list)
            assert isinstance(prompt[0], dict)

            # if functions passed, temporarily add them to messages to ease passing to next stage
            if functions:
                function_message = {
                    "role": "tools",
                    "functions": functions
                }
                prompt = [function_message] + prompt
            kwargs['prompt'] = prompt
        else:
            if functions:
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                                         "functions only supported in openai api format")
        return generate_config, kwargs
