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
from maga_transformer.openai.renderers.qwen_tool_renderer import QwenToolStreamStatus
from maga_transformer.utils.mm_process_engine import MMProcessEngine
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, UsageInfo, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, \
    RoleEnum, RendererInfo, PromptTokensDetails, \
    ChatCompletionTokenLogprob, TopLogprob, ChoiceLogprobs
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.utils.word_util import get_stop_word_slices, truncate_response_with_stop_words, is_truncated
from maga_transformer.utils.multimodal_util import MMUrlType, MultimodalInput, MMPreprocessConfig

class StreamStatus:
    index: int = 0
    request: ChatCompletionRequest
    output: Optional[GenerateOutput] = None
    origin_output_ids: torch.Tensor = torch.empty(0, dtype=torch.int32)
    output_ids: torch.Tensor = torch.empty(0, dtype=torch.int32)
    last_output_ids: List[int] = []
    last_token_length: int = 0
    finish_reason = None
    tokenizer = None
    responded_string = ""
    delta_output_string = ""

    def __init__(self, request: ChatCompletionRequest):
        self.request = request

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

@dataclass
class OutputDelta():
    output_str: Union[str, DeltaMessage]
    logprobs: Optional[ChatCompletionTokenLogprob]
    input_length: int
    output_length: int
    reuse_length: int


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
        self.stop_words_id_list = renderer_params.stop_word_ids_list
        self.stop_words_str_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in self.stop_words_id_list
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

    def _check_all_finished(self, status_list) -> bool:
        for s in status_list:
            if s.finish_reason == None:
                return False
        return True

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

    async def _create_empty_delta(self, aux_info: AuxInfo):
        return OutputDelta(
            output_str="",
            logprobs=None,
            input_length=aux_info.input_len,
            output_length=aux_info.output_len,
            reuse_length=aux_info.reuse_len
        )

    async def _generate_log_probs(self, status: StreamStatus, output: Optional[GenerateOutput]) -> Optional[ChatCompletionTokenLogprob]:
        assert output is not None
        if not status.request.logprobs:
            return None
        prob_return_num = status.request.top_logprobs or 1
        all_probs = output.all_probs
        output_id = output.output_ids
        if output_id == None:
            return None
        selected_id = output_id[-1].item()
        if (all_probs == None):
            raise Exception("all_probs is None when logprobs is true. There should be a internal bug.")
        all_probs = all_probs.squeeze()
        probs, tokens = all_probs.sort(descending=True)
        non_zero_size = probs.nonzero().shape[0]
        log_values = probs.log()
        prob_return_num = min(prob_return_num, non_zero_size)

        selected_token = self.tokenizer.decode([selected_id])
        chat_logprob = ChatCompletionTokenLogprob(
            token=selected_token,
            bytes=list(selected_token.encode("utf-8", errors="replace")),
            logprob=all_probs[output_id].log().item(),
            top_logprobs=[]
        )
        for i in range(prob_return_num):
            token = self.tokenizer.decode(tokens[i].item())
            chat_logprob.top_logprobs.append(TopLogprob(
                token=token,
                logprob=log_values[i].item(),
                bytes=list(token.encode("utf-8", errors="replace")),
            ))

        logging.debug(f"chat_logprob: {chat_logprob.model_dump_json(indent=4)}")

        return chat_logprob

    async def _update_single_status(self, status: StreamStatus, output: GenerateOutput, max_new_tokens: int, stop_words_str: List[str], stop_word_slice_list: List[str], is_streaming: bool) -> OutputDelta:
        if status.finish_reason != None:
            return await self._create_empty_delta(status.output.aux_info)
        status.update_output(output, self._clean_output_ids, functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens), self._remove_stop_word_ids)
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and u'\uFFFD' == decoded_string[-1]:
                return await self._create_empty_delta(output.aux_info)
        else:
            while (len(decoded_string) > 0) and (u'\uFFFD' == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        status.delta_output_string = decoded_string[len(decoded_prev_token):]
        if is_truncated(status.delta_output_string, stop_words_str, is_streaming):
            status.finish_reason = FinisheReason.stop
            return await self._create_empty_delta(output.aux_info)
        if not is_truncated(status.delta_output_string, stop_word_slice_list, is_streaming):
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=await self._generate_log_probs(status, output),
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
                reuse_length=output.aux_info.reuse_len)
            status.delta_output_string = ""
            return delta
        else:
            return await self._create_empty_delta(output.aux_info)

    async def _generate_first(self, n: int):
        return StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(
                            role=RoleEnum.assistant,
                            content="",
                        ),
                    ) for i in range(n)]
        )

    async def _generate_stream_response(self, items: List[OutputDelta]) -> StreamResponseObject:
        if len(items) == 0:
            raise Exception("output items length should not be 0")
        input_lengths = items[0].input_length
        output_lengths = sum([x.output_length for x in items])
        reuse_lengths = items[0].reuse_length
        return StreamResponseObject(
                choices=[ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(
                        content=item.output_str,
                    ) if isinstance(item.output_str, str) else item.output_str,
                    logprobs=ChoiceLogprobs(
                        content=[item.logprobs] if item.logprobs != None else None,
                        refusal=None
                    ) if item.logprobs != None else None
                ) for i, item in enumerate(items)],
                usage=UsageInfo(
                    prompt_tokens=input_lengths,
                    total_tokens=input_lengths + output_lengths,
                    completion_tokens=output_lengths,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=reuse_lengths) if reuse_lengths > 0 else None
                )
        )

    async def _flush_buffer(self, buffer_list: List[StreamStatus], stop_words_str: List[str], is_streaming: bool):
        output_items: List[OutputDelta] = []
        for buffer in buffer_list:
            if buffer.output is None:
                raise Exception("last output should not be None")
            aux_info = buffer.output.aux_info
            trunc_string = truncate_response_with_stop_words(buffer.delta_output_string, stop_words_str, is_streaming)
            output_items.append(OutputDelta(
                trunc_string,
                await self._generate_log_probs(buffer, buffer.output),
                aux_info.input_len,
                aux_info.output_len,
                aux_info.reuse_len))
        return await self._generate_stream_response(output_items)

    async def _generate_final(self, buffer_list: List[StreamStatus]):
        input_token_length = 0
        output_token_length = 0
        reuse_length = 0
        aux_info = None
        for i, buffer in enumerate(buffer_list):
            if buffer.output is None:
                raise Exception("buffer last output should not be None")
            # 判断buffer有无generating_tool_call这个属性
            if isinstance(buffer, QwenToolStreamStatus) and buffer.generating_tool_call:
                buffer.finish_reason = FinisheReason.tool_call

            if buffer.finish_reason == None:
                logging.debug(f"output {i} found no stop reason! use stop as default.")
                buffer.finish_reason = FinisheReason.stop
            if i == 0:
                input_token_length = buffer.output.aux_info.input_len
                reuse_length = buffer.output.aux_info.reuse_len
            output_token_length += buffer.output.aux_info.output_len
        return StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(
                    content="",
                ),
                finish_reason=buffer.finish_reason
            ) for i, buffer in enumerate(buffer_list)],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=reuse_length) if reuse_length > 0 else None
            ),
            aux_info=aux_info
        )

    async def _create_status_list(self, n: int, request: ChatCompletionRequest) -> List[StreamStatus]:
        return [StreamStatus(request) for _ in range(n)]

    async def render_response_stream(
            self,
            output_generator: AsyncGenerator[GenerateOutputs, None],
            request: ChatCompletionRequest,
            generate_config: GenerateConfig
    ) -> AsyncGenerator[StreamResponseObject, None]:
        stop_word_slice_list = get_stop_word_slices(generate_config.stop_words_str)
        num_return_sequences = request.n if request.n is not None else 1
        status_list = await self._create_status_list(num_return_sequences, request)
        index = 0
        async for outputs in output_generator:
            if index == 0:
                yield await self._generate_first(num_return_sequences)
            index += 1
            if len(outputs.generate_outputs) != num_return_sequences:
                raise Exception("output num != num_return_sequences")
            delta_list: List[OutputDelta] = []
            for status, output in zip(status_list, outputs.generate_outputs):
                delta_list.append(await self._update_single_status(status, output, generate_config.max_new_tokens, generate_config.stop_words_str, stop_word_slice_list, generate_config.is_streaming))
            yield await self._generate_stream_response(delta_list)
            if self._check_all_finished(status_list):
                break
        if index != 0:
            yield await self._flush_buffer(status_list, generate_config.stop_words_str, generate_config.is_streaming)
            yield await self._generate_final(status_list)

    def _check_finish_reason(self, token_ids: List[int], input_token_length: int, max_new_tokens: int = -1) -> Optional[FinisheReason]:
        stop_word_ids_list_all = self.get_all_extra_stop_word_ids_list() + self.stop_words_id_list
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
        stop_word_ids_list_all = self.get_all_extra_stop_word_ids_list() + self.stop_words_id_list
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
