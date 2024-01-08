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
        num_return_sequences  = self._format_request(kwargs)
        input_texts, input_images, batch_inference = self._get_input(kwargs)
        return self._yield_generate(input_texts, input_images, num_return_sequences, batch_inference, **kwargs)

    def stop(self) -> None:
        if isinstance(self.model, AsyncModel):
            logging.info("stoping InferenceWorker")
            self.model.stop()

    def _reshape_response(self, responses: List[str], finished: List[bool], num_return_sequences: int) -> Tuple[List[List[str]], List[bool]]:
        new_responses: List[List[str]] = []
        new_finished: List[bool] = []
        batch_size = len(responses) // num_return_sequences
        for i in range(0, batch_size):
            single_res: List[str] = []
            single_finished: List[bool] = []
            for j in range(0, num_return_sequences):
                single_res.append(responses[i * num_return_sequences + j])
                single_finished.append(finished[i * num_return_sequences + j])
            new_responses.append(single_res)
            new_finished.append(all(single_finished))
        return new_responses, new_finished

    def _format_response(self, gen_responses: GenerateResponse,
                        batch_response: bool, num_return_sequences: Optional[int],
                        last_response: bool = False, return_hidden_states: bool = False,
                        calculate_loss: int = 0, return_logits: bool = False) -> Dict[str, Any]:
        responses = gen_responses.batch_response
        finished = gen_responses.generate_output.finished.tolist()
        batch_size = gen_responses.generate_output.output_ids.shape[0]
        beam_width = gen_responses.generate_output.output_ids.shape[1]
        aux_info = gen_responses.generate_output.aux_info
        hidden_states = gen_responses.generate_output.hidden_states
        loss = gen_responses.generate_output.loss
        logits = gen_responses.generate_output.logits

        if last_response:
            finished = [True] * len(finished)
        if num_return_sequences:
            responses, finished = self._reshape_response(responses, finished, num_return_sequences)
            if aux_info:
                aux_info = [aux_info[i: i + num_return_sequences] for i in range(0, len(aux_info), num_return_sequences)]
        if beam_width > 1:
            assert(len(responses) == batch_size * beam_width)
            assert(aux_info != None)
            assert(len(aux_info) == batch_size)
            initial_responses = []
            initial_finished = []
            for i in range(0, batch_size):
                current_beams = responses[i * beam_width: (i + 1) * beam_width]
                initial_responses.append(current_beams[0])
                initial_finished.append(finished[i * beam_width])
                aux_info[i]["beam_responses"] = current_beams
            responses = initial_responses
            finished = initial_finished

        if aux_info is not None:
            assert(len(aux_info) == len(responses))
        else:
            aux_info = [{} for _ in range(len(responses))]

        if return_hidden_states:
            assert(len(hidden_states) == len(responses))
            if isinstance(hidden_states, list):
                hidden_states = [_.tolist() if isinstance(_, torch.Tensor) else [] for _ in hidden_states]
            else:
                hidden_states = hidden_states.tolist()

        if calculate_loss:
            assert(len(loss) == len(responses))
            if isinstance(loss, list):
                loss = [_.tolist() if isinstance(_, torch.Tensor) else [] for _ in loss]
            else:
                loss = loss.tolist()

        if return_logits:
            assert(len(logits) == len(responses))
            if isinstance(logits, list):
                logits = [_.tolist() if isinstance(_, torch.Tensor) else [] for _ in logits]
            else:
                logits = logits.tolist()

        if batch_response:
            response = {
                "response_batch": [
                    {
                        "response": response,
                        "finished": finish,
                        "aux_info": aux
                    }
                    for response, finish, aux in zip(responses, finished, aux_info)
                ]
            }
            if return_hidden_states:
                decimals = aux_info[0].get("decimals", None)
                if decimals:
                    for index in range(len(response["response_batch"])):
                        response["response_batch"][index]["hidden_states"] = ','.join('{:.{}f}'.format(e, decimals) for e in hidden_states[index])
                else:
                    for index in range(len(response["response_batch"])):
                        response["response_batch"][index]["hidden_states"] = hidden_states[index]
            if calculate_loss:
                for index in range(len(response["response_batch"])):
                    response["response_batch"][index]["loss"] = loss[index]
            if return_logits:
                for index in range(len(response["response_batch"])):
                    response["response_batch"][index]["logits"] = logits[index]

        else:
            response = {
                "response": responses[0],
                "finished": all(finished),
                "aux_info": aux_info[0]
            }
            if return_hidden_states:
                response["hidden_states"] = hidden_states[0]

            if calculate_loss:
                response["loss"] = loss[0]

            if return_logits:
                response["logits"] = logits[0]

        return response

    async def _yield_generate(self, texts: List[str], images: List[List[str]], num_return_sequences: Optional[int], batch_response: bool, **kwargs: Any):
        if num_return_sequences:
            new_texts: List[str] = []
            for text in texts:
                new_texts.extend([text] * num_return_sequences)
            new_images: List[str] = []
            for image in images:
                new_images.extend([image] * num_return_sequences)
        else:
            new_texts = texts
            new_images = images

        calculate_loss = 0
        return_hidden_states = False
        return_logits = False
        generate_config = kwargs.get("generate_config", None)
        if generate_config:
            return_hidden_states = generate_config.get("return_hidden_states", False) or generate_config.get("output_hidden_states", False)
            calculate_loss = generate_config.get("calculate_loss", 0)
            return_logits = generate_config.get("return_logits", False) or generate_config.get("output_logits", False)
            output_input_ids = generate_config.get("output_input_ids", False)

            generate_config["return_hidden_states"] = return_hidden_states
            generate_config["return_logits"] = return_logits
            generate_config["return_input_ids"] = output_input_ids

        stream = self.pipeline.pipeline_async(prompts=new_texts, images=new_images, **kwargs)
        generate_response = None
        async for generate_response in stream:
            yield self._format_response(generate_response, batch_response, num_return_sequences, False, return_hidden_states, calculate_loss, return_logits)
        assert generate_response is not None
        yield self._format_response(generate_response, batch_response, num_return_sequences, True, return_hidden_states, calculate_loss, return_logits)

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
        num_return_sequences = kwargs.pop('num_return_sequences', None)
        return num_return_sequences

    def _get_input(self, kwargs: Dict[str,Any]) -> Tuple[List[Any], List[Any], bool]:
        input_texts: Optional[Union[List[str], List[List[Dict[str, str]]]]] = None
        input_images: Optional[Union[List[str], List[List[str]]]] = None
        images = kwargs.pop('images', None)
        adapter_name = kwargs['generate_config'].get("adapter_name", None)
        if images is not None and not isinstance(images, list):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "input images should be list")
        batch_inference = False
        if "prompt_batch" in kwargs:
            input_texts = kwargs.pop('prompt_batch')
            batch_inference = True
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
            # check adapter_name size is same with prompt
            if adapter_name != None:
                if (isinstance(adapter_name, str) and len(input_texts) != 1) or \
                    (isinstance(adapter_name, list) and  len(input_texts) != len(adapter_name)):
                    raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "adapter_name is not alignment")
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
            # check adapter_name size is same with prompt
            if adapter_name != None :
                if (isinstance(adapter_name, str) and len(input_texts) != 1) or \
                    (isinstance(adapter_name, list) and  len(adapter_name) != 1):
                    FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "adapter_name is not alignment")
        if input_texts is None:
            raise FtRuntimeException(ExceptionType.NO_PROMPT_ERROR, "not input prompt")
        return input_texts, input_images, batch_inference

    def is_streaming(self, req: Dict[str, Any]):
        return req.get(
            'yield_generator',
            req.get('generation_config',
                    req.get('generate_config', {})
                    ).get('yield_generator', False))
