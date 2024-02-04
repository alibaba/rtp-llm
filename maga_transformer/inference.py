import os
import copy
import sys
import json
import logging
import pathlib
import torch
from typing import Any, Dict, List, Tuple, Optional, Union

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.absolute()))

from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.utils.util import copy_gemm_config
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.models.base_model import GenerateResponse
from maga_transformer.config.generate_config import RequestFormat
from maga_transformer.model_factory import ModelFactory, AsyncModel

class InferenceWorker():
    def __init__(self) -> None:
        copy_gemm_config()
        logging.info("starting InferenceWorker")
        if not torch.cuda.is_available():
            raise Exception("GPU not found")

        self.model = ModelFactory.create_from_env()
        self.pipeline = Pipeline(self.model, self.model.tokenizer)
        logging.info("Load model done.")

    def inference(self, **kwargs: Any):
        num_return_sequences = self._format_request(kwargs)
        input_texts, input_images, generate_configs, batch_infer = self._get_input(kwargs, num_return_sequences)
        kwargs.pop('generate_config')
        if len(input_texts) > 1 or batch_infer:
            generators = [self._yield_generate(text, images, generate_config=generate_config, **kwargs)
                          for text, images, generate_config in zip(input_texts, input_images, generate_configs)]
            incremental = generate_configs[0].get('return_incremental', False)
            return self._batch_async_generators(incremental, num_return_sequences, generators, batch_infer)
        else:
            return self._yield_generate(input_texts[0], input_images[0],
                                        generate_config=generate_configs[0], **kwargs)

    def stop(self) -> None:
        if isinstance(self.model, AsyncModel):
            logging.info("stoping InferenceWorker")
            self.model.stop()

    def _format_response(self, gen_responses: GenerateResponse,
                         return_hidden_states: bool = False,
                         calculate_loss: int = 0,
                         return_logits: bool = False) -> Dict[str, Any]:
        generate_texts = gen_responses.generate_texts
        finished = gen_responses.generate_output.finished
        beam_width = gen_responses.generate_output.output_ids.shape[0]
        aux_info = gen_responses.generate_output.aux_info
        hidden_states = gen_responses.generate_output.hidden_states
        loss = gen_responses.generate_output.loss
        logits = gen_responses.generate_output.logits

        if beam_width > 1:
            aux_info.beam_responses = generate_texts

        response = {
            "response": generate_texts[0],
            "finished": finished,
            "aux_info": aux_info.model_dump(mode='json'),
        }
        if return_hidden_states:
            response["hidden_states"] = hidden_states.tolist()
        if calculate_loss:
            response['loss'] = lost.tolist()
        if return_logits:
            response['logits'] = logits.tolist()

        return response

    async def _yield_generate(self, text: str, images: List[str], **kwargs: Any):
        calculate_loss = 0
        return_hidden_states = False
        return_logits = False
        generate_config = kwargs.get("generate_config", None)
        if generate_config:
            return_hidden_states = generate_config.get("return_hidden_states", False) or generate_config.get("output_hidden_states", False)
            calculate_loss = generate_config.get("calculate_loss", 0)
            return_logits = generate_config.get("return_logits", False) or generate_config.get("output_logits", False)
            output_input_ids = generate_config.get("output_input_ids", False) or generate_config.get("return_input_ids", False)

            generate_config["return_hidden_states"] = return_hidden_states
            generate_config["return_logits"] = return_logits
            generate_config["return_input_ids"] = output_input_ids

        stream = self.pipeline.pipeline_async(prompt=text, images=images, **kwargs)
        async for generate_response in stream:
            yield self._format_response(generate_response, return_hidden_states, calculate_loss, return_logits)

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

    def _format_request(self, kwargs: Any):
        generate_config = copy.deepcopy(self.model.default_generate_config)
        generate_config.update(kwargs.get('generate_config', kwargs.get('generation_config', {})))
        kwargs['generate_config'] = generate_config
        if 'num_return_sequences' in generate_config:
            kwargs['num_return_sequences'] = generate_config.pop("num_return_sequences")
        if 'text' in kwargs:
            kwargs['prompt'] = kwargs.pop('text')
        if 'gen_length' in kwargs:
            kwargs['max_new_tokens'] = kwargs.pop('gen_length')
        self._format_chat_api_messages(kwargs)
        num_return_sequences = int(kwargs.pop('num_return_sequences', 1))
        return num_return_sequences

    def _get_input(self, kwargs: Dict[str,Any], num_return_sequences) -> Tuple[List[Any], List[Any], bool]:
        input_texts: Optional[Union[List[str], List[List[Dict[str, str]]]]] = None
        input_images: Optional[Union[List[str], List[List[str]]]] = None
        images = kwargs.pop('images', None)
        generate_config = kwargs['generate_config']
        adapter_name = generate_config.get("adapter_name", None)
        if images is not None and not isinstance(images, list):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "input images should be list")
        batch_infer = False
        if "prompt_batch" in kwargs:
            batch_infer = True
            input_texts = kwargs.pop('prompt_batch')
            if not isinstance(input_texts, list):
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch input should be list")
            if images is not None:
                if not isinstance(images[0], list):
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch images should be list[list]")
                if len(images) != len(input_texts):
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "prompt batch images and input should have same length")
                input_images = images
            else:
                input_images = [[]] * len(input_texts)
        else:
            prompt: Union[str, List[str], List[Dict[str, str]]] = kwargs.pop('prompt')
            if isinstance(prompt, str):
                input_texts = [prompt]
            # for AutoML format
            elif isinstance(prompt, list) and isinstance(prompt[0], dict):
                input_texts = [prompt]
            else:
                input_texts = prompt
            if images == None or len(images) == 0:
                input_images = [[]] * len(input_texts)
            elif len(images) > 0 and isinstance(images[0], str):
                input_images = [images]
            else:
                input_images = images
        if input_texts is None:
            raise FtRuntimeException(ExceptionType.NO_PROMPT_ERROR, "not input prompt")

            # check adapter_name size is same with prompt
        generate_configs = [generate_config] * len(input_texts)
        if adapter_name != None:
            if (isinstance(adapter_name, str) and len(input_texts) != 1) or \
               (isinstance(adapter_name, list) and  len(input_texts) != len(adapter_name)):
                raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "adapter_name is not alignment")
            for i in range(len(input_texts)):
                generate_configs[i] = copy.copy(generate_configs[i])
                generate_configs[i]['adapter_name'] = adapter_name[i] if isinstance(adapter_name, list) else adapter_name

        def repeat_elements(lst, n):
            return [e for e in lst for _ in range(n)]
        if num_return_sequences:
            input_texts = repeat_elements(input_texts, num_return_sequences)
            input_images = repeat_elements(input_images, num_return_sequences)
            generate_configs = repeat_elements(generate_configs, num_return_sequences)
        return input_texts, input_images, generate_configs, batch_infer

    def is_streaming(self, req: Dict[str, Any]):
        return req.get(
            'yield_generator',
            req.get('generation_config',
                    req.get('generate_config', {})
                    ).get('yield_generator', False))

    def update(self, version_info: VersionInfo):
        lora_infos = dict()
        if version_info.peft_info != None:
            lora_infos = version_info.peft_info.get("lora_info", {})
        return self.model.update(lora_infos)


    @staticmethod
    async def _batch_async_generators(incremental, num_return_sequences, generators, batch_infer):
        iterators = [gen.__aiter__() for gen in generators]
        done_idxs = set()
        batch_state = [None] * len(iterators)
        while True:
            for idx, itr in enumerate(iterators):
                try:
                    batch_state[idx] = await itr.__anext__()
                except StopAsyncIteration:
                    done_idxs.add(idx)
                if idx in done_idxs:
                    if batch_state[idx] is None:
                        batch_state[idx] = {"response": '', 'finished':True, 'aux_info':{}}
                    if incremental:
                        batch_state[idx]['response'] = ''
            if len(done_idxs) == len(iterators):
                break
            if num_return_sequences is not None:
                batch_size = int(len(batch_state) / num_return_sequences)
            batch = batch_state
            if num_return_sequences > 1:
                new_batch = []
                for batch_idx in range(batch_size):
                    seqs = batch_state[batch_idx * num_return_sequences:(batch_idx + 1) * num_return_sequences]
                    new_batch.append({"response": [seq['response'] for seq in seqs],
                                      'finished': [seq['finished'] for seq in seqs],
                                      'aux_info':[seq['aux_info'] for seq in seqs]})
                batch = new_batch
            if batch_infer:
                yield {'response_batch':batch}
            else:
                yield batch[0]
