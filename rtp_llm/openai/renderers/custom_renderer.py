import copy
import functools
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import TemplateType
from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionExtraOutputs,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    CompletionTokensDetails,
    DeltaMessage,
    FinisheReason,
    PromptTokensDetails,
    RendererInfo,
    RoleEnum,
    TopLogprob,
    UsageInfo,
)
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)
from rtp_llm.utils.multimodal_util import MMPreprocessConfig, MMUrlType, MultimodalInput
from rtp_llm.utils.util import has_overlap_kmp
from rtp_llm.utils.word_util import (
    get_stop_word_slices,
    is_truncated,
    truncate_response_with_stop_words,
)

THINK_MODE = StaticConfig.generate_env_config.think_mode

THINK_START_TAG = StaticConfig.generate_env_config.think_start_tag.encode(
    "utf-8"
).decode("unicode_escape")
THINK_END_TAG = StaticConfig.generate_env_config.think_end_tag.encode("utf-8").decode(
    "unicode_escape"
)


class StreamStatus:
    index: int = 0
    request: ChatCompletionRequest
    output: Optional[GenerateOutput] = None
    output_ids: torch.Tensor = torch.empty(0, dtype=torch.int32)
    output_ids_list: List[int] = []
    last_output_ids: List[int] = []
    last_token_length: int = 0
    finish_reason = None
    tokenizer = None
    responded_string = ""
    delta_output_string = ""

    def __init__(self, request: ChatCompletionRequest):
        self.request = request

    def update_output(
        self,
        output: GenerateOutput,
        check_finish_func,
        remove_stop_word_ids_func,
    ):
        self.index += 1
        self.output = output
        delta_output_ids = output.output_ids.cpu().flatten().tolist()
        self.output_ids_list = copy.deepcopy(self.output_ids_list + delta_output_ids)
        self.finish_reason = check_finish_func(
            self.output_ids_list, self.input_token_length
        )
        self.output_ids = remove_stop_word_ids_func(
            self.output_ids_list, delta_output_ids
        )

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
        return self.last_output_ids[-self.last_token_length :]

    @property
    def tokens_to_decode(self):
        return self.prev_token_id + self.output_ids[len(self.last_output_ids) :]

    def __str__(self):
        return (
            f"StreamStatus("
            f"index={self.index}, "
            f"request={self.request}, "
            f"output={self.output}, "
            f"output_ids={self.output_ids}, "
            f"last_output_ids={self.last_output_ids}, "
            f"last_token_length={self.last_token_length}, "
            f"finish_reason={self.finish_reason}, "
            f"tokenizer={self.tokenizer}, "
            f"responded_string={self.responded_string!r}, "
            f"delta_output_string={self.delta_output_string!r})"
        )


class StreamStatusSync:
    index: int = 0
    request: ChatCompletionRequest
    output_ids: torch.Tensor = torch.empty(0, dtype=torch.int32)
    last_output_ids: List[int] = []
    output_ids_list: List[int] = []
    last_token_length: int = 0
    finish_reason = None
    tokenizer = None
    responded_string = ""
    delta_output_string = ""

    def __init__(self, request: ChatCompletionRequest):
        self.request = request

    def update_output_sync(
        self,
        output_ids,
        input_len,
        check_finish_func,
        remove_stop_word_ids_func,
    ):
        self.index += 1
        delta_output_ids = output.output_ids.cpu().flatten().tolist()
        self.output_ids_list = copy.deepcopy(self.output_ids_list + delta_output_ids)
        self.finish_reason = check_finish_func(self.output_ids_list, input_len)
        self.output_ids = remove_stop_word_ids_func(
            self.output_ids_list, delta_output_ids
        )

    def update_result(self):
        self.last_token_length = len(self.output_ids) - len(self.last_output_ids)
        self.last_output_ids = self.output_ids
        self.responded_string += self.delta_output_string

    @property
    def prev_token_id(self):
        return self.last_output_ids[-self.last_token_length :]

    @property
    def tokens_to_decode(self):
        return self.prev_token_id + self.output_ids[len(self.last_output_ids) :]


@dataclass
class StreamResponseObject:
    choices: List[ChatCompletionResponseStreamChoice] = field(default_factory=list)
    usage: Optional[UsageInfo] = None
    aux_info: Optional[AuxInfo] = None
    extra_outputs: Optional[ChatCompletionExtraOutputs] = None


@dataclass
class ResponseObject:
    choices: List[ChatCompletionResponseChoice] = field(default_factory=list)
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
    custom_modal_config: Optional[Dict[str, Any]] = None


@dataclass
class OutputDelta:
    output_str: Union[str, DeltaMessage]
    logprobs: Optional[ChatCompletionTokenLogprob]
    input_length: int
    output_length: int
    reuse_length: int
    extra_outputs: Optional[ChatCompletionExtraOutputs] = None


@dataclass
class ThinkStatus:
    enable_think_mode: bool = False
    in_think_mode: bool = False
    think_buffer: str = ""
    think_tokens: int = 0
    is_streaming: bool = False


class RenderedInputs:
    input_ids: List[int] = []
    multimodal_inputs: List[MultimodalInput] = []
    rendered_prompt: str = ""

    def __init__(
        self,
        input_ids: List[int],
        rendered_prompt: str = "",
        input_urls: List[str] = [],
        input_urls_type: List[MMUrlType] = [],
        preprocess_configs: List[MMPreprocessConfig] = [],
        input_datas: List[bytes] = [],
    ):
        self.input_ids = input_ids
        self.rendered_prompt = rendered_prompt
        self.multimodal_inputs = []
        if len(input_urls_type) == 0:
            input_urls_type = [MMUrlType.DEFAULT] * len(input_urls)
        elif len(input_urls_type) != len(input_urls):
            raise Exception(
                f"the number of multimodal input types must match url, now types {len(input_urls_type)} urls {len(input_urls)}"
            )

        if len(preprocess_configs) == 0:
            preprocess_configs = [MMPreprocessConfig()] * len(input_urls)
        elif len(preprocess_configs) != len(preprocess_configs):
            raise Exception(
                f"the number of multimodal preprocess config must match url, now types {len(preprocess_configs)} urls {len(input_urls)}"
            )
        
        if len(input_datas) == 0:
            input_datas = [b""] * len(input_urls)
        elif len(input_datas) != len(input_urls):
            raise Exception(
                f"the number of multimodal input data must match url, now datas {len(input_datas)} urls {len(input_urls)}"
            )

        for url, type, config, data in zip(input_urls, input_urls_type, preprocess_configs, input_datas):
            self.multimodal_inputs.append(MultimodalInput(url=url, mm_type=type, config=config, data=data))


class CustomChatRenderer:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
    ):
        self.py_env_configs = PyEnvConfigs()
        self.py_env_configs.update_from_env()
        self.tokenizer = tokenizer
        self.model_type = renderer_params.model_type
        self.max_seq_len = renderer_params.max_seq_len
        self.eos_token_id = renderer_params.eos_token_id
        self.stop_words_id_list = renderer_params.stop_word_ids_list
        self.stop_words_str_list = [
            self.tokenizer.decode(stop_word_ids)
            for stop_word_ids in self.stop_words_id_list
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
            self.tokenizer.decode(stop_word_ids)
            for stop_word_ids in extra_stop_word_ids_list
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
            token_id = self.tokenizer.convert_tokens_to_ids(word)
            if isinstance(token_id, int):
                ids_list.append([token_id])
            elif isinstance(token_id, list):
                ids_list.append(token_id)
            else:
                ids_list.append(self.tokenizer.encode(word, add_special_tokens=True))
        return ids_list

    def get_all_extra_stop_word_ids_list(self) -> List[List[int]]:
        ids_list_from_words = self.tokenize_words(self.extra_stop_words)
        return self.extra_stop_word_ids_list + ids_list_from_words

    def _check_all_finished(self, status_list) -> bool:
        for s in status_list:
            if s.finish_reason == None:
                return False
        return True

    def getRequest(self, request: str) -> ChatCompletionRequest:
        return ChatCompletionRequest(**(json.loads(request)))

    def _setup_chat_template(self, template_file_name: str = "chat_template.jinja"):
        """设置聊天模板, 兼容从文件读取chat_template的情况"""
        self.chat_template = self.tokenizer.chat_template
        if not self.chat_template:
            logging.warning(
                f"Tokenizer try load chat_template from {template_file_name}."
            )
            tokenizer_path = self.tokenizer.path
            if tokenizer_path and os.path.exists(tokenizer_path):
                default_template_path = os.path.join(tokenizer_path, template_file_name)
                if os.path.exists(default_template_path):
                    with open(default_template_path, "r") as f:
                        # load all content
                        self.chat_template = f.read()
                    logging.info(
                        f"loaded default chat template from {default_template_path}"
                    )
                else:
                    logging.warning(
                        f"Default chat template not found at {default_template_path}, using empty template."
                    )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        raise NotImplementedError

    async def generate_choice(
        self,
        request_id: int,
        input_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[StreamResponseObject, None]:

        token_type_ids = []
        input_id_tensor = torch.Tensor(input_ids).int().unsqueeze(0)
        output_generator: AsyncGenerator[GenerateOutputs, None] = (
            await backend_rpc_server_visitor.enqueue(
                GenerateInput(
                    request_id=request_id,
                    token_ids=input_id_tensor,
                    mm_inputs=mm_inputs,
                    generate_config=generate_config,
                    tokenizer=self.tokenizer,
                    token_type_ids=token_type_ids,
                )
            )
        )

        # 处理非流式请求的合并逻辑
        if not generate_config.is_streaming:
            output_generator = await self._merge_non_streaming_outputs(output_generator)

        async for response in self.render_response_stream(
            output_generator, request, generate_config
        ):
            yield response

    async def _merge_non_streaming_outputs(
        self, output_generator: AsyncGenerator[GenerateOutputs, None]
    ) -> AsyncGenerator[GenerateOutputs, None]:
        """
        合并非流式请求的多个输出为单个输出

        Args:
            output_generator: 原始的输出生成器

        Returns:
            包含单个合并输出的新生成器
        """
        # 收集所有输出
        collected_outputs = []
        async for output in output_generator:
            collected_outputs.append(output)

        # 合并输出
        merged_output = self._merge_generate_outputs(collected_outputs)

        # 创建新的单次输出generator
        async def single_output_generator():
            yield merged_output

        return single_output_generator()

    def _merge_generate_outputs(
        self, collected_outputs_list: List[GenerateOutputs]
    ) -> GenerateOutputs:
        """
        合并多个GenerateOutputs为单个GenerateOutputs

        Args:
            collected_outputs_list: 要合并的GenerateOutputs列表

        Returns:
            合并后的GenerateOutputs
        """
        if len(collected_outputs_list) <= 1:
            return (
                collected_outputs_list[0]
                if collected_outputs_list
                else GenerateOutputs(generate_outputs=[])
            )

        # 获取最后一个作为基础
        final_outputs = collected_outputs_list[-1]

        # 对每个choice/beam分别处理
        merged_generate_outputs = []
        for i, final_output in enumerate(final_outputs.generate_outputs):
            # 收集这个choice在所有GenerateOutputs中的output_ids
            all_output_ids = []
            for outputs in collected_outputs_list:
                if i < len(outputs.generate_outputs):
                    output_ids = outputs.generate_outputs[i].output_ids
                    all_output_ids.append(output_ids)

            # 合并output_ids
            merged_output_ids = (
                torch.cat(all_output_ids, dim=1)
                if all_output_ids
                else final_output.output_ids
            )

            # 深拷贝final_output并只替换output_ids
            merged_output = copy.deepcopy(final_output)
            merged_output.output_ids = merged_output_ids

            merged_generate_outputs.append(merged_output)

        return GenerateOutputs(generate_outputs=merged_generate_outputs)

    async def _create_empty_delta(self, aux_info: AuxInfo):
        return OutputDelta(
            output_str="",
            logprobs=None,
            input_length=aux_info.input_len,
            output_length=aux_info.output_len,
            reuse_length=aux_info.reuse_len,
        )

    async def _generate_log_probs(
        self, status: StreamStatus, output: Optional[GenerateOutput]
    ) -> Optional[ChatCompletionTokenLogprob]:
        assert output is not None
        if not status.request.logprobs:
            return None
        prob_return_num = status.request.top_logprobs or 1
        all_probs = output.all_probs
        output_id = output.output_ids
        if output_id == None:
            return None
        selected_id = output_id[-1].item()
        if all_probs == None:
            raise Exception(
                "all_probs is None when logprobs is true. There should be a internal bug."
            )
        all_probs = all_probs.squeeze()
        non_zero_size = all_probs.nonzero().shape[0]
        prob_return_num = min(prob_return_num, non_zero_size)
        # 使用 topk 提高计算效率，只计算需要的前 k 个值
        probs, tokens = all_probs.topk(
            prob_return_num, dim=-1, largest=True, sorted=True
        )
        log_values = probs.log()

        selected_token = self.tokenizer.decode([selected_id])
        chat_logprob = ChatCompletionTokenLogprob(
            token=selected_token,
            bytes=list(selected_token.encode("utf-8", errors="replace")),
            logprob=all_probs[output_id].log().item(),
            top_logprobs=[],
        )
        for i in range(prob_return_num):
            token = self.tokenizer.decode(tokens[i].item())
            chat_logprob.top_logprobs.append(
                TopLogprob(
                    token=token,
                    logprob=log_values[i].item(),
                    bytes=list(token.encode("utf-8", errors="replace")),
                )
            )

        logging.debug("chat_logprob: %s", chat_logprob.model_dump_json(indent=4))

        return chat_logprob

    async def _generate_extra_outputs(
        self, output: GenerateOutput, generate_config: GenerateConfig
    ) -> Optional[ChatCompletionExtraOutputs]:
        final_result = None

        def result():
            nonlocal final_result
            if final_result is None:
                final_result = ChatCompletionExtraOutputs()
            return final_result

        if generate_config.return_hidden_states and output.hidden_states is not None:
            result().hidden_states = output.hidden_states.tolist()
        if (
            generate_config.return_all_hidden_states
            and output.all_hidden_states is not None
        ):
            result().all_hidden_states = output.all_hidden_states.tolist()
        if generate_config.calculate_loss != 0 and output.loss is not None:
            result().loss = output.loss.tolist()
        if generate_config.return_logits and output.logits is not None:
            result().logits = output.logits.tolist()
        if generate_config.return_output_ids and output.output_ids is not None:
            result().output_ids = output.output_ids.tolist()
        if generate_config.return_input_ids and output.input_ids is not None:
            result().input_ids = output.input_ids.tolist()

        return final_result

    async def _update_single_status(
        self,
        status: StreamStatus,
        output: GenerateOutput,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        if status.finish_reason != None:
            return await self._create_empty_delta(status.output.aux_info)
        status.update_output(
            output,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and "\ufffd" == decoded_string[-1]:
                return await self._create_empty_delta(output.aux_info)
        else:
            while (len(decoded_string) > 0) and ("\ufffd" == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]
        if is_truncated(status.delta_output_string, stop_words_str, is_streaming):
            status.finish_reason = FinisheReason.stop
            return await self._create_empty_delta(output.aux_info)
        if not is_truncated(
            status.delta_output_string, stop_word_slice_list, is_streaming, True
        ):
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=await self._generate_log_probs(status, output),
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
                reuse_length=output.aux_info.reuse_len,
            )
            status.delta_output_string = ""
            return delta
        else:
            return await self._create_empty_delta(output.aux_info)

    async def _generate_first(self, n: int):
        return StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(
                        role=RoleEnum.assistant,
                        content="",
                    ),
                )
                for i in range(n)
            ]
        )

    def _split_reasoning_text_and_content(
        self, item: OutputDelta, think_status: ThinkStatus
    ):
        if isinstance(item.output_str, str):
            processing_index, output_len = 0, len(item.output_str)

            if output_len == 0:
                return DeltaMessage(content="")

            reasoning_text, content = "", ""
            update_think_tokens = think_status.in_think_mode
            while processing_index < output_len:
                if think_status.in_think_mode:
                    think_status.think_buffer += item.output_str[processing_index]
                    if think_status.think_buffer.startswith(THINK_START_TAG):
                        think_status.think_buffer = think_status.think_buffer[
                            len(THINK_START_TAG) :
                        ]

                    if think_status.think_buffer.endswith(THINK_END_TAG):
                        reasoning_text = think_status.think_buffer[
                            : -len(THINK_END_TAG)
                        ]
                        think_status.think_buffer = ""
                        think_status.in_think_mode = False
                    elif has_overlap_kmp(
                        think_status.think_buffer, THINK_END_TAG
                    ) or THINK_START_TAG.startswith(think_status.think_buffer):
                        pass
                    else:
                        reasoning_text = think_status.think_buffer
                    processing_index += 1
                else:
                    content += item.output_str[processing_index:]
                    processing_index = output_len

            if think_status.in_think_mode:
                if has_overlap_kmp(
                    think_status.think_buffer, THINK_END_TAG
                ) or THINK_START_TAG.startswith(think_status.think_buffer):
                    reasoning_text = ""
                else:
                    think_status.think_buffer = ""

            if think_status.enable_think_mode and update_think_tokens:
                if not think_status.is_streaming:
                    think_status.think_tokens = item.output_length - len(
                        self.tokenizer.tokenize(content or "")
                    )
                else:
                    think_status.think_tokens = item.output_length
            return DeltaMessage(
                reasoning_content=reasoning_text or "", content=content or ""
            )

        elif isinstance(item.output_str, DeltaMessage):
            # 对于已经是 DeltaMessage的情况, 则代表下层已经处理好了tool_calls和reasoning_content
            if not think_status.is_streaming:
                think_status.think_tokens = len(
                    self.tokenizer.tokenize(item.output_str.reasoning_content or "")
                )
            else:
                has_content_or_tool_calls = (
                    item.output_str.content or item.output_str.tool_calls
                )
                has_reasoning = item.output_str.reasoning_content
                if has_reasoning and not has_content_or_tool_calls:
                    think_status.think_tokens = item.output_length
            return item.output_str

        else:
            raise Exception(f"undefined output_str type[{type(item.output_str)}]")

    async def _generate_stream_response(
        self, items: List[OutputDelta], think_status_list: List[ThinkStatus]
    ) -> StreamResponseObject:
        if len(items) == 0:
            raise Exception("output items length should not be 0")
        input_lengths = items[0].input_length
        output_lengths = sum([x.output_length for x in items])
        reuse_lengths = items[0].reuse_length
        all_choices = []
        for i, item in enumerate(items):
            delta = self._split_reasoning_text_and_content(item, think_status_list[i])
            all_choices.append(
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=delta,
                    logprobs=(
                        ChoiceLogprobs(
                            content=[item.logprobs] if item.logprobs != None else None,
                            refusal=None,
                        )
                        if item.logprobs != None
                        else None
                    ),
                )
            )

        return StreamResponseObject(
            choices=all_choices,
            usage=UsageInfo(
                prompt_tokens=input_lengths,
                total_tokens=input_lengths + output_lengths,
                completion_tokens=output_lengths,
                completion_tokens_details=(
                    CompletionTokensDetails(
                        reasoning_tokens=sum(
                            [x.think_tokens for x in think_status_list]
                        )
                    )
                    if think_status_list[0].enable_think_mode > 0
                    else None
                ),
                prompt_tokens_details=(
                    PromptTokensDetails(cached_tokens=reuse_lengths)
                    if reuse_lengths > 0
                    else None
                ),
            ),
            # TODO(zhangjianning.zjn): merge all extra outputs for streaming request
            extra_outputs=items[-1].extra_outputs,
        )

    async def _flush_buffer(
        self,
        buffer_list: List[StreamStatus],
        stop_words_str: List[str],
        is_streaming: bool,
        think_status_list: List[ThinkStatus],
    ):
        output_items: List[OutputDelta] = []
        for buffer in buffer_list:
            if buffer.output is None:
                raise Exception("last output should not be None")
            aux_info = buffer.output.aux_info
            trunc_string = truncate_response_with_stop_words(
                buffer.delta_output_string, stop_words_str, is_streaming
            )
            output_items.append(
                OutputDelta(
                    trunc_string,
                    await self._generate_log_probs(buffer, buffer.output),
                    aux_info.input_len,
                    aux_info.output_len,
                    aux_info.reuse_len,
                )
            )
        return await self._generate_stream_response(output_items, think_status_list)

    async def _generate_final(
        self,
        buffer_list: List[StreamStatus],
        request: ChatCompletionRequest,
        think_status_list: List[ThinkStatus],
    ):
        input_token_length = 0
        output_token_length = 0
        reuse_length = 0
        aux_info = None
        for i, buffer in enumerate(buffer_list):
            if buffer.output is None:
                raise Exception("buffer last output should not be None")
            # 延迟引入, 避免循环import
            from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
                ReasoningToolStreamStatus,
            )

            # 判断buffer有无generating_tool_call这个属性
            if (
                isinstance(buffer, ReasoningToolStreamStatus)
                and buffer.generating_tool_call
            ):
                buffer.finish_reason = FinisheReason.tool_calls

            if buffer.finish_reason == None:
                logging.debug("output %s found no stop reason! use stop as default.", i)
                buffer.finish_reason = FinisheReason.stop
            if i == 0:
                input_token_length = buffer.output.aux_info.input_len
                reuse_length = buffer.output.aux_info.reuse_len
                aux_info = buffer.output.aux_info if request.aux_info else None
            output_token_length += buffer.output.aux_info.output_len
        return StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(
                        content="",
                    ),
                    finish_reason=buffer.finish_reason,
                )
                for i, buffer in enumerate(buffer_list)
            ],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length,
                completion_tokens_details=(
                    CompletionTokensDetails(
                        reasoning_tokens=sum(
                            [x.think_tokens for x in think_status_list]
                        )
                    )
                    if think_status_list[0].enable_think_mode
                    else None
                ),
                prompt_tokens_details=(
                    PromptTokensDetails(cached_tokens=reuse_length)
                    if reuse_length > 0
                    else None
                ),
            ),
            aux_info=aux_info,
        )

    async def _create_status_list(
        self, n: int, request: ChatCompletionRequest
    ) -> List[StreamStatus]:
        return [StreamStatus(request) for _ in range(n)]

    def in_think_mode(self, request: ChatCompletionRequest):
        global THINK_MODE
        return THINK_MODE

    def should_process_think(self, request: ChatCompletionRequest):
        # 留出方法给子类重写, 避免重复的think处理
        return self.in_think_mode(request)

    async def render_response_stream(
        self,
        output_generator: AsyncGenerator[GenerateOutputs, None],
        request: ChatCompletionRequest,
        generate_config: GenerateConfig,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        stop_word_slice_list = get_stop_word_slices(generate_config.stop_words_str)
        nums_output = request.n if request.n is not None else 1
        # FIXME(zhangjianning.zjn): for variable width beam search,
        # the num_ouput may not be the last num beams,
        # and is dependent to the length of sequence
        last_num_beams = (
            generate_config.variable_num_beams[-1]
            if len(generate_config.variable_num_beams) > 1
            else generate_config.num_beams
        )
        nums_output = last_num_beams if last_num_beams != 1 else nums_output
        status_list = await self._create_status_list(nums_output, request)
        index = 0
        think_status_list = [
            ThinkStatus(
                enable_think_mode=bool(self.in_think_mode(request)),
                in_think_mode=bool(self.should_process_think(request)),
                think_buffer="",
                think_tokens=0,
                is_streaming=generate_config.is_streaming,
            )
            for _ in range(nums_output)
        ]
        async for outputs in output_generator:
            if index == 0:
                yield await self._generate_first(nums_output)
            index += 1
            if len(outputs.generate_outputs) != nums_output:
                raise Exception(
                    f"output num {len(outputs.generate_outputs)} != nums_output {nums_output}"
                )
            delta_list: List[OutputDelta] = []
            for status, output in zip(status_list, outputs.generate_outputs):
                delta = await self._update_single_status(
                    status,
                    output,
                    generate_config.max_new_tokens,
                    generate_config.stop_words_str,
                    stop_word_slice_list,
                    generate_config.is_streaming,
                )
                if delta.extra_outputs is None:
                    delta.extra_outputs = await self._generate_extra_outputs(
                        output, generate_config
                    )
                delta_list.append(delta)
            yield await self._generate_stream_response(delta_list, think_status_list)
            if self._check_all_finished(status_list):
                break
        if index != 0:
            yield await self._flush_buffer(
                status_list,
                generate_config.stop_words_str,
                generate_config.is_streaming,
                think_status_list,
            )
            yield await self._generate_final(status_list, request, think_status_list)

    def _create_empty_delta_sync(self, input_len: int, output_len: int, reuse_len: int):
        return OutputDelta(
            output_str="",
            logprobs=None,
            input_length=input_len,
            output_length=output_len,
            reuse_length=reuse_len,
        )

    def _generate_log_probs_sync(
        self,
        status: StreamStatusSync,
        all_probs: Optional[torch.Tensor],
        output_ids: Optional[torch.Tensor],
    ) -> Optional[ChatCompletionTokenLogprob]:
        if not status.request.logprobs:
            return None
        prob_return_num = status.request.top_logprobs or 1
        all_probs = all_probs
        output_id = output_ids
        if output_id == None:
            return None
        selected_id = output_id[-1].item()
        if all_probs == None:
            raise Exception(
                "all_probs is None when logprobs is true. There should be a internal bug."
            )
        all_probs = all_probs.squeeze()
        non_zero_size = all_probs.nonzero().shape[0]
        prob_return_num = min(prob_return_num, non_zero_size)
        # 使用 topk 提高计算效率，只计算需要的前 k 个值
        probs, tokens = all_probs.topk(
            prob_return_num, dim=-1, largest=True, sorted=True
        )
        log_values = probs.log()

        selected_token = self.tokenizer.decode([selected_id])
        chat_logprob = ChatCompletionTokenLogprob(
            token=selected_token,
            bytes=list(selected_token.encode("utf-8", errors="replace")),
            logprob=all_probs[output_id].log().item(),
            top_logprobs=[],
        )
        for i in range(prob_return_num):
            token = self.tokenizer.decode(tokens[i].item())
            chat_logprob.top_logprobs.append(
                TopLogprob(
                    token=token,
                    logprob=log_values[i].item(),
                    bytes=list(token.encode("utf-8", errors="replace")),
                )
            )

        logging.debug("chat_logprob: %s", chat_logprob.model_dump_json(indent=4))

        return chat_logprob

    def _update_single_status_sync(
        self,
        status: StreamStatusSync,
        input_len: int,  # output.aux_info
        output_len: int,  # output.aux_info
        reuse_len: int,  # output.aux_info
        all_probs: torch.Tensor,
        output_ids: torch.Tensor,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        if status.finish_reason != None:
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)
        status.update_output_sync(
            output_ids,
            input_len,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and "\ufffd" == decoded_string[-1]:
                return self._create_empty_delta_sync(input_len, output_len, reuse_len)
        else:
            while (len(decoded_string) > 0) and ("\ufffd" == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]
        if is_truncated(status.delta_output_string, stop_words_str, is_streaming):
            status.finish_reason = FinisheReason.stop
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)
        if not is_truncated(
            status.delta_output_string, stop_word_slice_list, is_streaming, True
        ):
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=self._generate_log_probs_sync(status, all_probs, output_ids),
                input_length=input_len,
                output_length=output_len,
                reuse_length=reuse_len,
            )
            status.delta_output_string = ""
            return delta
        else:
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)

    def _generate_first_sync(self, n: int):
        return StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(
                        role=RoleEnum.assistant,
                        content="",
                    ),
                )
                for i in range(n)
            ]
        )

    def _generate_stream_response_sync(
        self, items: List[OutputDelta]
    ) -> StreamResponseObject:
        if len(items) == 0:
            raise Exception("output items length should not be 0")
        input_lengths = items[0].input_length
        output_lengths = sum([x.output_length for x in items])
        reuse_lengths = items[0].reuse_length
        return StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=(
                        DeltaMessage(
                            content=item.output_str,
                        )
                        if isinstance(item.output_str, str)
                        else item.output_str
                    ),
                    logprobs=(
                        ChoiceLogprobs(
                            content=[item.logprobs] if item.logprobs != None else None,
                            refusal=None,
                        )
                        if item.logprobs != None
                        else None
                    ),
                )
                for i, item in enumerate(items)
            ],
            usage=UsageInfo(
                prompt_tokens=input_lengths,
                total_tokens=input_lengths + output_lengths,
                completion_tokens=output_lengths,
                prompt_tokens_details=(
                    PromptTokensDetails(cached_tokens=reuse_lengths)
                    if reuse_lengths > 0
                    else None
                ),
            ),
        )

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
        output_items: List[OutputDelta] = []
        for buffer, input_len, output_len, reuse_len, all_probs, output_ids in zip(
            buffer_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            all_probs_list,
            output_ids_list,
        ):
            trunc_string = truncate_response_with_stop_words(
                buffer.delta_output_string, stop_words_str, is_streaming
            )
            output_items.append(
                OutputDelta(
                    trunc_string,
                    self._generate_log_probs_sync(buffer, all_probs, output_ids),
                    input_len,
                    output_len,
                    reuse_len,
                )
            )
        return self._generate_stream_response_sync(output_items)

    def _generate_final_sync(
        self,
        buffer_list: List[StreamStatusSync],
        input_len_list,
        output_len_list,
        reuse_len_list,
    ):
        input_token_length = 0
        output_token_length = 0
        reuse_length = 0
        aux_info = None
        for i, (buffer, input_len, output_len, reuse_len) in enumerate(
            zip(buffer_list, input_len_list, output_len_list, reuse_len_list)
        ):
            if buffer.finish_reason == None:
                logging.debug("output %s found no stop reason! use stop as default.", i)
                buffer.finish_reason = FinisheReason.stop
            if i == 0:
                input_token_length = input_len
                reuse_length = reuse_len
            output_token_length += output_len
        return StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(
                        content="",
                    ),
                    finish_reason=buffer.finish_reason,
                )
                for i, buffer in enumerate(buffer_list)
            ],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length,
                prompt_tokens_details=(
                    PromptTokensDetails(cached_tokens=reuse_length)
                    if reuse_length > 0
                    else None
                ),
            ),
            aux_info=aux_info,
        )

    def _create_status_list_sync(self, n: int, body: str) -> List[StreamStatusSync]:
        request = self.getRequest(body)
        return [StreamStatusSync(request) for _ in range(n)]

    def render_stream_response_first(self, n: int, debug_info: str):
        stream_response = self._generate_first_sync(n)
        chat_response = ChatCompletionStreamResponse(
            choices=stream_response.choices,
            usage=stream_response.usage,
            aux_info=stream_response.aux_info,
            debug_info=debug_info,
        )
        return chat_response.model_dump_json(exclude_none=True)

    def render_stream_response_refactor(
        self,
        status_list: StreamStatusSync,  # pass in from cpp
        input_len_list,  # output.aux_info
        output_len_list,  # output.aux_info
        reuse_len_list,  # output.aux_info
        all_probs_list,  # GenerateOutput
        output_ids_list,  # GenerateOutput
        max_new_tokens,  # GenerateConfig
        stop_words_str,  # GenerateConfig
        is_streaming,
    ):
        stop_word_slice_list = get_stop_word_slices(
            stop_words_str
        )  # move into cpp, then pass in
        delta_list: List[OutputDelta] = []
        for status, input_len, output_len, reuse_len, all_probs, output_ids in zip(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,  # AuxInfo
            all_probs_list,
            output_ids_list,  # GenerateOutput
        ):
            delta_list.append(
                self._update_single_status_sync(
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
            )
        stream_response = self._generate_stream_response_sync(delta_list)
        chat_response = ChatCompletionStreamResponse(
            choices=stream_response.choices,
            usage=stream_response.usage,
            aux_info=stream_response.aux_info,
        )
        return chat_response.model_dump_json(exclude_none=True)

    def render_stream_response_flush(
        self,
        status_list,
        input_len_list,
        output_len_list,
        reuse_len_list,
        all_probs_list,
        output_ids_list,
        stop_words_str,
        is_streaming,
    ):
        stream_response = self._flush_buffer_sync(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            all_probs_list,
            output_ids_list,
            stop_words_str,
            is_streaming,
        )
        chat_response = ChatCompletionStreamResponse(
            choices=stream_response.choices,
            usage=stream_response.usage,
            aux_info=stream_response.aux_info,
        )
        return chat_response.model_dump_json(exclude_none=True)

    def render_stream_response_final(
        self, status_list, input_len_list, output_len_list, reuse_len_list
    ):
        stream_response = self._generate_final_sync(
            status_list, input_len_list, output_len_list, reuse_len_list
        )
        chat_response = ChatCompletionStreamResponse(
            choices=stream_response.choices,
            usage=stream_response.usage,
            aux_info=stream_response.aux_info,
        )
        return chat_response.model_dump_json(exclude_none=True)

    def render_stream_response_first_blocking(self, n: int):
        stream_response = self._generate_first_sync(n)
        return stream_response

    def render_stream_response_blocking(
        self,
        status_list: StreamStatusSync,  # pass in from cpp
        input_len_list,  # output.aux_info
        output_len_list,  # output.aux_info
        reuse_len_list,  # output.aux_info
        all_probs_list,  # GenerateOutput
        output_ids_list,  # GenerateOutput
        max_new_tokens,  # GenerateConfig
        stop_words_str,  # GenerateConfig
        is_streaming,
    ):
        stop_word_slice_list = get_stop_word_slices(
            stop_words_str
        )  # move into cpp, then pass in
        delta_list: List[OutputDelta] = []
        for status, input_len, output_len, reuse_len, all_probs, output_ids in zip(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,  # AuxInfo
            all_probs_list,
            output_ids_list,  # GenerateOutput
        ):
            delta_list.append(
                self._update_single_status_sync(
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
            )
        stream_response = self._generate_stream_response_sync(delta_list)
        return stream_response

    def render_stream_response_flush_blocking(
        self,
        status_list,
        input_len_list,
        output_len_list,
        reuse_len_list,
        all_probs_list,
        output_ids_list,
        stop_words_str,
        is_streaming,
    ):
        stream_response = self._flush_buffer_sync(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            all_probs_list,
            output_ids_list,
            stop_words_str,
            is_streaming,
        )
        return stream_response

    def render_stream_response_final_blocking(
        self, status_list, input_len_list, output_len_list, reuse_len_list
    ):
        stream_response = self._generate_final_sync(
            status_list, input_len_list, output_len_list, reuse_len_list
        )
        return stream_response

    def collect_complete_response(self, choice_generator):
        all_choices = []
        usage = None
        aux_info = None

        def split_think_tag(text: Optional[str]):
            if text is None:
                return None, None
            text_results = text.split(THINK_END_TAG, 1)
            reasoning_content = text_results[0] if len(text_results) == 2 else None
            content = text_results[1] if len(text_results) == 2 else text
            return content, reasoning_content

        for response in choice_generator:

            if len(response.choices) != len(all_choices):
                if all_choices == []:
                    for i, choice in enumerate(response.choices):
                        content, reasoning_content = split_think_tag(
                            choice.delta.content
                        )
                        all_choices.append(
                            ChatCompletionResponseChoice(
                                index=i,
                                message=ChatMessage(
                                    role=choice.delta.role or RoleEnum.assistant,
                                    content=content or None,
                                    reasoning_content=reasoning_content or None,
                                    function_call=choice.delta.function_call or None,
                                ),
                                finish_reason=choice.finish_reason,
                                logprobs=choice.logprobs,
                            )
                        )
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
                    content, reasoning_content = split_think_tag(
                        all_choices[i].message.content
                    )
                    all_choices[i].message.content = content
                    all_choices[i].message.reasoning_content = reasoning_content
                    all_choices[i].message.role = (
                        response.choices[i].delta.role or all_choices[i].message.role
                    )
                    all_choices[i].message.function_call = (
                        response.choices[i].delta.function_call
                        or all_choices[i].message.function_call
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

        if usage == None:
            logging.warning(f"No usage returned from stream response. use empty value.")
            usage = UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0)
        chat_response = ChatCompletionResponse(
            choices=all_choices,
            usage=usage,
            aux_info=aux_info,
            model="AsyncModel",
        )
        return chat_response.model_dump_json(exclude_none=True)

    def _check_finish_reason(
        self, token_ids: List[int], input_token_length: int, max_new_tokens: int = -1
    ) -> Optional[FinisheReason]:
        stop_word_ids_list_all = (
            self.get_all_extra_stop_word_ids_list() + self.stop_words_id_list
        )
        if max_new_tokens > 0 and len(token_ids) >= max_new_tokens:
            return FinisheReason.length
        if len(token_ids) + input_token_length >= self.max_seq_len:
            return FinisheReason.length
        if token_ids and token_ids[-1] == self.eos_token_id:
            return FinisheReason.stop
        for stop_word_ids in stop_word_ids_list_all:
            if (len(token_ids) >= len(stop_word_ids)) and (
                token_ids[-len(stop_word_ids) :] == stop_word_ids
            ):
                return FinisheReason.stop
        return None

    def _remove_stop_word_ids(
        self, output_ids: List[int], delta_output_ids: List[int]
    ) -> List[int]:
        stop_word_ids_list_all = (
            self.get_all_extra_stop_word_ids_list() + self.stop_words_id_list
        )
        start_pos = len(output_ids) - len(delta_output_ids)
        end_pos = len(output_ids)
        if start_pos >= 0:
            for i in range(start_pos, end_pos):
                if output_ids[i] == self.eos_token_id:
                    output_ids = output_ids[:i]
                    break
        for stop_word_ids in stop_word_ids_list_all:
            #  此处应该从最大的范围开始判断
            # 有可能会有stopword_ids 重复的情况，比如[144575, 14098, 144575]
            # 若从1开始判断会导致 去除了最后一个 144575 就退出了
            for i in range(len(stop_word_ids) + 1, 1, -1):
                if output_ids[-i:] == stop_word_ids[:i]:
                    output_ids = output_ids[:-i]
                    break
        return output_ids
