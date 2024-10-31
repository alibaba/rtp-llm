from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
import functools
import os
import torch
import asyncio
import logging
from dataclasses import dataclass, field
from PIL import Image
from concurrent.futures import Future

from transformers import PreTrainedTokenizerBase

from maga_transformer.models.base_model import GenerateOutput, BaseModel, GenerateInput, GenerateOutputs, AuxInfo
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.gpt_init_model_parameters import TemplateType
from maga_transformer.utils.mm_process_engine import MMProcessEngine
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, UsageInfo, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, \
    RoleEnum, RendererInfo, PromptTokensDetails
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.utils.word_util import get_stop_word_slices, truncate_response_with_stop_words, is_truncated
from maga_transformer.utils.multimodal_util import MMUrlType, MultimodalInput, MMPreprocessConfig

class StreamStatus:
    index: int = 0
    output: GenerateOutput
    origin_output_ids: torch.Tensor = torch.empty(0, dtype=torch.int32)
    output_ids: torch.Tensor = torch.empty(0, dtype=torch.int32)
    last_output_ids: List[int] = []
    last_token_length: int = 0
    finish_reason = None
    tokenizer = None
    responded_string = ""
    delta_output_string = ""
    
    def update_output(self, output: GenerateOutput, clean_output_func, check_finish_func, remove_stop_word_ids_func):
        self.index += 1
        self.output = output
        self.origin_output_ids = torch.cat((self.origin_output_ids, output.output_ids), dim=1)
        self.output_ids = clean_output_func(self.origin_output_ids)
        self.finish_reason = check_finish_func(self.output_ids, self.input_token_length)
        self.output_ids = remove_stop_word_ids_func(self.output_ids)

    def update_result(self):
        self.last_token_length = len(self.output_ids) - len(self.last_output_ids)
        self.last_output_ids = self.output_ids
        self.responded_string += self.delta_output_string

    @property
    def output_token_length(self):
        return len(self.output_ids)
    
    @property
    def input_token_length(self):
        return self.output.aux_info.input_len

    @property
    def reuse_length(self):
        return self.output.aux_info.reuse_len
    
    @property
    def prev_token_id(self):
        return self.last_output_ids[-self.last_token_length:]
    
    @property
    def tokens_to_decode(self):
        return self.prev_token_id + self.output_ids[len(self.last_output_ids):]
    
def generate_stream_response(status: StreamStatus, request = None, is_first: bool = False, is_last: bool = False):
        choices = None
        usage = None
        aux_info = None
        if is_first:
            choices=[ChatCompletionResponseStreamChoice(
                    index=status.index,
                    delta=DeltaMessage(
                        role=RoleEnum.assistant,
                        content="",
                    )
            )]
        else:
            usage=UsageInfo(
                    prompt_tokens=status.input_token_length,
                    total_tokens=status.input_token_length + status.output_token_length,
                    completion_tokens=status.output_token_length,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=status.reuse_length) if status.reuse_length > 0 else None
                )
            if is_last:
                choices=[ChatCompletionResponseStreamChoice(
                    index=status.index + 1,
                    delta=DeltaMessage(
                        content="",
                    ),
                    finish_reason=status.finish_reason
                )]
                aux_info=output.aux_info if request.aux_info else None
            else:
                choices=[ChatCompletionResponseStreamChoice(
                        index=status.index,
                        delta=DeltaMessage(
                            content=status.delta_output_string,
                        ),
                    )]
        return StreamResponseObject(choices=choices, usage=usage, aux_info=aux_info)

@dataclass
class StreamResponseObject:
    choices: List[ChatCompletionResponseStreamChoice] = field(default_factory=list)
    usage: Optional[UsageInfo] = None
    aux_info: Optional[AuxInfo] = None

@dataclass
class RendererParams:
    model_type: str
    max_seq_len: int
    eos_token_id: int
    stop_word_ids_list: List[List[int]]
    template_type: TemplateType = TemplateType.chat
    ckpt_path: str = ""

class RenderedInputs:
    input_ids: List[int] = []
    multimodal_inputs: List[MultimodalInput] = []
    rendered_prompt: str = ""

    def __init__(self, input_ids: List[int], rendered_prompt: str = "", input_urls: List[str] = [], input_urls_type: List[MMUrlType] = [], preprocess_configs: List[MMPreprocessConfig] = []):
        self.input_ids = input_ids
        self.rendered_prompt = rendered_prompt
        self.multimodal_inputs = []
        if len(input_urls_type) == 0:
            input_urls_type = [MMUrlType.DEFAULT] * len(input_urls)
        elif len(input_urls_type) != len(input_urls):
            raise Exception(f"the number of multimodal input types must match url, now types {len(input_urls_type)} urls {len(input_urls)}")
        
        if len(preprocess_configs) == 0:
            preprocess_configs = [MMPreprocessConfig()] * len(input_urls)
        elif len(preprocess_configs) != len(preprocess_configs):
            raise Exception(f"the number of multimodal preprocess config must match url, now types {len(preprocess_configs)} urls {len(input_urls)}")
        
        for url, type, config in zip(input_urls, input_urls_type, preprocess_configs):
            self.multimodal_inputs.append(MultimodalInput(url, type, config))

class CustomChatRenderer():
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 renderer_params: RendererParams,
    ):
        self.tokenizer = tokenizer
        self.model_type = renderer_params.model_type
        self.max_seq_len = renderer_params.max_seq_len
        self.eos_token_id = renderer_params.eos_token_id
        self.stop_word_ids_list = renderer_params.stop_word_ids_list
        self.stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in self.stop_word_ids_list
        ]
        self.ckpt_path = renderer_params.ckpt_path
        # NOTE: stop words or their ids only need to be added to one of these two lists.
        self.extra_stop_words: List[str] = []
        self.extra_stop_word_ids_list: List[List[int]] = []

    def __str__(self) -> str:
        return str(self.get_renderer_info())

    def __repr__(self) -> str:
        return self.__str__()

    def get_renderer_info(self) -> RendererInfo:
        extra_stop_word_ids_list = self.get_all_extra_stop_word_ids_list()
        extra_stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in extra_stop_word_ids_list
        ]
        if len(extra_stop_words_list) and isinstance(extra_stop_words_list[0], list):
            extra_stop_words_list = [l[0] for l in extra_stop_words_list]
        return RendererInfo(
            class_name=self.__class__.__name__,
            renderer_model_type=self.model_type,
            extra_stop_word_ids_list=extra_stop_word_ids_list,
            extra_stop_words_list=extra_stop_words_list,
        )

    def add_extra_stop_words(self, extra_stop_words: List[str]):
        self.extra_stop_words.extend(extra_stop_words)

    def add_extra_stop_word_ids(self, extra_stop_word_ids: List[List[int]]):
        self.extra_stop_word_ids_list.extend(extra_stop_word_ids)

    def tokenize_words(self, words: List[str]) -> List[List[int]]:
        ids_list = []
        for word in words:
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                token_id = self.tokenizer.convert_tokens_to_ids(word)
                if isinstance(token_id, int):
                    ids_list.append([token_id])
                elif isinstance(token_id, list):
                    ids_list.append(token_id)
                else:
                    ids_list.append(self.tokenizer.encode(word, add_special_tokens=True))
            else:
                ids_list.append(self.tokenizer.encode(word))
        return ids_list

    def get_all_extra_stop_word_ids_list(self) -> List[List[int]]:
        ids_list_from_words = self.tokenize_words(self.extra_stop_words)
        return self.extra_stop_word_ids_list + ids_list_from_words

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        raise NotImplementedError

    async def generate_choice(
            self,
            request_id: int,
            input_ids: List[int],
            mm_inputs: List[MultimodalInput],
            generate_config: GenerateConfig,
            model: Union[AsyncModel, BaseModel],
            request: ChatCompletionRequest
    ) -> AsyncGenerator[StreamResponseObject, None]:

        token_type_ids = []
        input_id_tensor = torch.Tensor(input_ids).int().unsqueeze(0)

        generate_config.is_streaming = True
        output_generator: AsyncGenerator[GenerateOutput, None] = model.enqueue(
            GenerateInput(
                request_id=request_id,
                token_ids=input_id_tensor,
                mm_inputs=mm_inputs,
                generate_config=generate_config,
                tokenizer=self.tokenizer,
                token_type_ids=token_type_ids
            )
        )

        async for response in self.render_response_stream(output_generator,
                                                          request,
                                                          generate_config):
            yield response

    async def render_response_stream(
            self,
            output_generator: AsyncGenerator[GenerateOutputs, None],
            request: ChatCompletionRequest,
            generate_config: GenerateConfig
    ) -> AsyncGenerator[StreamResponseObject, None]:
        stop_word_slice_list = get_stop_word_slices(generate_config.stop_words_str)
        status = StreamStatus()

        async for outputs in output_generator:
            if status.index == 0:
                yield generate_stream_response(status, is_first=True)

            output = outputs.generate_outputs[0]
            status.update_output(output, self._clean_output_ids, functools.partial(self._check_finish_reason, max_new_tokens=generate_config.max_new_tokens), self._remove_stop_word_ids)
            
            decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
            decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        
            # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
            if (len(decoded_string)) and (u'\uFFFD' == decoded_string[-1]):
                continue
            status.delta_output_string = decoded_string[len(decoded_prev_token):]
            if is_truncated(status.delta_output_string, generate_config.stop_words_str):
                status.finish_reason = FinisheReason.stop
                break

            if not is_truncated(status.delta_output_string, stop_word_slice_list):
                status.update_result()
                stream_response = generate_stream_response(status)
                status.delta_output_string = ""
                yield stream_response

        if not is_truncated(status.delta_output_string, generate_config.stop_words_str):
            status.responded_string += status.delta_output_string
            yield generate_stream_response(status)

        if status.finish_reason == None:
            logging.debug(f"output [{status.responded_string}] found no stop reason! use stop as default.")
            status.finish_reason = FinisheReason.stop

        yield generate_stream_response(status, is_last=True, request=request)

    def _check_finish_reason(self, token_ids: List[int], input_token_length: int, max_new_tokens: int = -1) -> Optional[FinisheReason]:
        stop_word_ids_list_all = self.get_all_extra_stop_word_ids_list() + self.stop_word_ids_list
        if max_new_tokens > 0 and len(token_ids) >= max_new_tokens:
            return FinisheReason.length
        if len(token_ids) + input_token_length >= self.max_seq_len:
            return FinisheReason.length
        if token_ids and token_ids[-1] == self.eos_token_id:
            return FinisheReason.stop
        for stop_word_ids in stop_word_ids_list_all:
            if (len(token_ids) >= len(stop_word_ids)) and (token_ids[-len(stop_word_ids):] == stop_word_ids):
                return FinisheReason.stop
        return None

    def _remove_stop_word_ids(self, output_ids: List[int]) -> List[int]:
        stop_word_ids_list_all = self.get_all_extra_stop_word_ids_list() + self.stop_word_ids_list
        for stop_word_ids in stop_word_ids_list_all:
            #  此处应该从最大的范围开始判断
            # 有可能会有stopword_ids 重复的情况，比如[144575, 14098, 144575]
            # 若从1开始判断会导致 去除了最后一个 144575 就退出了
            for i in range(len(stop_word_ids) + 1, 1, -1):
                if output_ids[-i:] == stop_word_ids[:i]:
                    output_ids = output_ids[:-i]
                    break
        return output_ids

    def _clean_output_ids(self, output_ids_tensor: torch.Tensor) -> list[int]:
        output_ids_tensor = output_ids_tensor.cpu().reshape([-1])
        # TODO(wangyin): This slicing shouldn't be done here.
        # model should return output length, ids should be sliced with output length.
        output_ids = output_ids_tensor[output_ids_tensor != self.eos_token_id].tolist()
        return output_ids
