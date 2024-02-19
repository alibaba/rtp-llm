import os
import copy
import sys
import json
import logging
import pathlib
import torch
from typing import Any, Dict, List, Tuple, Optional, Union, NamedTuple

from maga_transformer.config.generate_config import RequestFormat

class Request(NamedTuple):
    batch_infer: bool
    input_texts: Any
    input_images: Any
    generate_configs: Any
    num_return_sequences: int

class RequestExtractor:
    def __init__(self, default_generate_config):
        self.default_generate_config = default_generate_config
        
    def extract_request(self, kwargs: Any):
        num_return_sequences = self._format_request(kwargs)
        request = self._get_request(kwargs, num_return_sequences)
        return request
    
    def _format_request(self, kwargs: Any):
        generate_config = copy.deepcopy(self.default_generate_config)
        generate_config.update(kwargs.get('generate_config', kwargs.get('generation_config', {})))
    
        generate_config["return_hidden_states"] = generate_config.get("return_hidden_states", False) or generate_config.get("output_hidden_states", False)
        generate_config["return_logits"] = generate_config.get("return_logits", False) or generate_config.get("output_logits", False)
        generate_config["return_input_ids"] = generate_config.get("return_input_ids", False) or generate_config.get("output_input_ids", False)
        kwargs['generate_config'] = generate_config        
        
        if 'num_return_sequences' in generate_config:
            kwargs['num_return_sequences'] = generate_config.pop("num_return_sequences")
        num_return_sequences = int(kwargs.pop('num_return_sequences', 1)) 

        if 'text' in kwargs:
            kwargs['prompt'] = kwargs.pop('text')
        if 'gen_length' in kwargs:
            kwargs['max_new_tokens'] = kwargs.pop('gen_length')
                
        self._format_chat_api_messages(kwargs)
        
        return num_return_sequences

    def _get_text(self, kwargs: Dict[str,Any]):
        input_texts: Optional[Union[List[str], List[List[Dict[str, str]]]]] = None
        if "prompt_batch" in kwargs:
            batch_infer = True
            input_texts = kwargs.pop('prompt_batch')
            if not isinstance(input_texts, list):
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch input should be list")
        else:
            prompt: Union[str, List[str], List[Dict[str, str]]] = kwargs.pop('prompt')
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

    def _get_images(self, input_len, kwargs: Dict[str,Any]):
        input_images: Optional[Union[List[str], List[List[str]]]] = None
        images = kwargs.pop('images', None)
        if images is not None and not isinstance(images, list):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "input images should be list")
        if "prompt_batch" in kwargs:
            if images is not None:
                if not isinstance(images[0], list):
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch images should be list[list]")
                if len(images) != input_len:
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch images and input should have same length")
                input_images = images
            else:
                input_images = [[]] * input_len
        else:
            if images == None or len(images) == 0:
                input_images = [[]] * input_len
            elif len(images) > 0 and isinstance(images[0], str):
                input_images = [images]
            else:
                input_images = images
        return input_images

    def _is_batch(self, kwargs: Dict[str,Any]):
         return "prompt_batch" in kwargs

    def _get_adapter(self, input_len, kwargs: Dict[str,Any]):
        generate_config = kwargs.pop('generate_config')
        generate_configs = [generate_config] * input_len
        adapter_name = generate_config.get("adapter_name", None)
        if adapter_name != None:
            if (isinstance(adapter_name, str) and input_len != 1) or \
               (isinstance(adapter_name, list) and  input_len != len(adapter_name)):
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "adapter_name is not alignment")
            for i in range(input_len):
                generate_configs[i] = copy.copy(generate_configs[i])
                generate_configs[i]['adapter_name'] = adapter_name[i] if isinstance(adapter_name, list) else adapter_name

        return generate_configs

    def extend_sequences(self, input_texts, input_images, generate_configs, num_return_sequences):
        # check adapter_name size is same with prompt
        def repeat_elements(lst, n):
            return [e for e in lst for _ in range(n)]
        if num_return_sequences:
            input_texts = repeat_elements(input_texts, num_return_sequences)
            input_images = repeat_elements(input_images, num_return_sequences)
            generate_configs = repeat_elements(generate_configs, num_return_sequences)
        return input_texts, input_images, generate_configs

    def _get_request(self, kwargs: Dict[str,Any], num_return_sequences) -> Tuple[List[Any], List[Any], bool]:
        batch_infer = self._is_batch(kwargs)
        input_texts = self._get_text(kwargs)
        input_images = self._get_images(len(input_texts), kwargs)
        generate_configs = self._get_adapter(len(input_texts), kwargs)
        input_texts, input_images, generate_configs = self.extend_sequences(input_texts, input_images, generate_configs, num_return_sequences)
        
        return Request(batch_infer, input_texts, input_images, generate_configs, num_return_sequences)

    def _format_chat_api_messages(self, kwargs: Any) -> None:
        if 'messages' in kwargs:
            assert 'prompt' not in kwargs
            messages = kwargs.pop('messages')
            assert isinstance(messages, list)
            kwargs['prompt'] = messages
            kwargs['generate_config']['request_format'] = RequestFormat.CHAT_API

        prompt = kwargs.get('prompt', None)
        functions = kwargs.get('functions', None)
        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            kwargs['generate_config']['request_format'] = RequestFormat.CHAT_API

        if kwargs['generate_config'].get('request_format', None) == RequestFormat.CHAT_API:
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