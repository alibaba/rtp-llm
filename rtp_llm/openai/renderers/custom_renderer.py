import copy
import functools
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
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
from rtp_llm.server.request_headers import normalize_request_headers
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


def _merge_choice_logprob_tensors(
    outputs_list: List[GenerateOutputs], choice_index: int
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    merged_tensors: List[Optional[torch.Tensor]] = []
    for field_name in (
        "token_logprobs",
        "top_logprob_token_ids",
        "top_logprobs",
    ):
        chunks = []
        for outputs in outputs_list:
            if choice_index >= len(outputs.generate_outputs):
                continue
            value = getattr(outputs.generate_outputs[choice_index], field_name)
            if value is not None:
                chunks.append(value)
        merged_tensors.append(torch.cat(chunks, dim=0) if chunks else None)

    return merged_tensors[0], merged_tensors[1], merged_tensors[2]


def _get_think_config(generate_env_config):
    """Get thinking configuration from generate_env_config.

    Args:
        generate_env_config: GenerateEnvConfig object.

    Returns:
        Tuple of (think_mode, think_start_tag, think_end_tag)
    """
    think_mode = generate_env_config.think_mode
    think_start_tag = generate_env_config.think_start_tag.encode("utf-8").decode(
        "unicode_escape"
    )
    think_end_tag = generate_env_config.think_end_tag.encode("utf-8").decode(
        "unicode_escape"
    )
    return think_mode, think_start_tag, think_end_tag


class StreamStatus:
    index: int = 0
    request: ChatCompletionRequest
    output: Optional[GenerateOutput] = None
    output_ids: List[int] = []
    output_ids_list: List[int] = []
    last_output_ids: List[int] = []
    last_token_length: int = 0
    finish_reason = None
    tokenizer = None
    responded_string = ""
    delta_output_string = ""

    def __init__(self, request: ChatCompletionRequest):
        self.request = request
        self.pending_logprobs: List[ChatCompletionTokenLogprob] = []
        self.emitted_logprob_token_count = 0

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
        self.pending_logprobs: List[ChatCompletionTokenLogprob] = []
        self.emitted_logprob_token_count = 0

    def update_output_sync(
        self,
        output_ids,
        input_len,
        check_finish_func,
        remove_stop_word_ids_func,
    ):
        self.index += 1
        delta_output_ids = output_ids.cpu().flatten().tolist()
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


class TemplateType(Enum):
    """Template type for different model types."""

    chat = "chat"
    vqa = "vqa"
    base = "image"


@dataclass
class RendererParams:
    model_type: str
    max_seq_len: int
    eos_token_id: int
    stop_word_ids_list: List[List[int]]
    template_type: TemplateType = TemplateType.chat
    ckpt_path: str = ""


@dataclass
class OutputDelta:
    output_str: Union[str, DeltaMessage]
    logprobs: Optional[List[ChatCompletionTokenLogprob]]
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
    raw_content_for_logprobs: bool = False


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

        for url, type, config in zip(input_urls, input_urls_type, preprocess_configs):
            self.multimodal_inputs.append(MultimodalInput(url, type, config))


class CustomChatRenderer:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config: GenerateEnvConfig,
        render_config: Optional[RenderConfig] = None,
        ckpt_path: Optional[str] = None,
        misc_config: Optional[Any] = None,
        vit_config: Optional[Any] = None,
    ):
        # Get think config from generate_env_config
        self.think_mode, self.think_start_tag, self.think_end_tag = _get_think_config(
            generate_env_config
        )

        # Store configs for subclasses
        self.ckpt_path = ckpt_path
        self.misc_config = misc_config
        self.vit_config = vit_config
        self.render_config = render_config

        # Create a minimal model_config-like object for renderers that access self.model_config.checkpoint_path
        # This is only for backward compatibility with existing renderer code that accesses model_config attributes
        class MinimalModelConfig:
            def __init__(self, ckpt_path: str, misc_config: Any, vit_config: Any):
                self.ckpt_path = ckpt_path
                self.checkpoint_path = ckpt_path
                self.misc_config = misc_config
                self.vit_config = vit_config

        self.model_config = MinimalModelConfig(ckpt_path or "", misc_config, vit_config)

        self.tokenizer = tokenizer
        self.model_type = renderer_params.model_type
        self.max_seq_len = renderer_params.max_seq_len
        self.eos_token_id = renderer_params.eos_token_id
        self.stop_words_id_list = renderer_params.stop_word_ids_list
        self.stop_words_str_list = [
            self.tokenizer.decode(stop_word_ids)
            for stop_word_ids in self.stop_words_id_list
        ]
        self._tokenizer_byte_candidates: Optional[List[Any]] = None
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

    def apply_chat_completion_constraints(
        self, request: ChatCompletionRequest, generate_config: GenerateConfig
    ) -> None:
        if request.logprobs and self.should_process_think(request):
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS,
                "logprobs is not supported when the renderer parses thinking output",
            )
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is None or tool_choice in ("auto", "none"):
            return
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS,
            f"tool_choice={tool_choice!r} is not supported by "
            f"{self.__class__.__name__}",
        )

    async def generate_choice(
        self,
        request_id: int,
        input_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
        request: ChatCompletionRequest,
        headers: Optional[Dict[str, str]] = None,
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
                    headers=normalize_request_headers(headers),
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
            (
                merged_output.token_logprobs,
                merged_output.top_logprob_token_ids,
                merged_output.top_logprobs,
            ) = _merge_choice_logprob_tensors(collected_outputs_list, i)

            merged_generate_outputs.append(merged_output)

        return GenerateOutputs(generate_outputs=merged_generate_outputs)

    async def _create_empty_delta(
        self,
        aux_info: AuxInfo,
        logprobs: Optional[List[ChatCompletionTokenLogprob]] = None,
    ):
        return OutputDelta(
            output_str="",
            logprobs=logprobs,
            input_length=aux_info.input_len,
            output_length=aux_info.output_len,
            reuse_length=aux_info.reuse_len,
        )

    def _token_id_to_bytes(
        self, token_id: int, decoded_token: str
    ) -> Optional[List[int]]:
        """Return exact token bytes when the tokenizer exposes them.

        Byte-level tokenizers can decode an individual, incomplete UTF-8 token as
        U+FFFD (or as an empty string). Encoding that display string would invent
        replacement bytes which were never sampled. Prefer raw-token APIs and
        decoder tables; only fall back to UTF-8 encoding when the decoded token is
        complete. ``None`` means the original bytes cannot be recovered safely.
        """

        # A complete per-token decode already identifies the exact UTF-8 bytes
        # represented by the API's token string. Raw-backend discovery is only
        # needed for byte fragments that decode to U+FFFD or an empty string.
        if decoded_token and "\ufffd" not in decoded_token:
            try:
                return list(decoded_token.encode("utf-8"))
            except UnicodeEncodeError:
                return None

        def normalize_raw_bytes(value: Any) -> Optional[List[int]]:
            if isinstance(value, (bytes, bytearray, memoryview)):
                return list(bytes(value))
            if (
                isinstance(value, (list, tuple))
                and value
                and all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and 0 <= item <= 255
                    for item in value
                )
            ):
                return list(value)
            return None

        # BaseTokenizer wraps a HuggingFace tokenizer, which may itself wrap a
        # tiktoken/tokenizers/SentencePiece backend. Walk the small adapter graph
        # so decode_single_token_bytes(), byte decoder tables, and byte-fallback
        # pieces remain usable without requiring one tokenizer implementation.
        candidates = self._tokenizer_byte_candidates
        if candidates is None:
            candidates = []
            pending: List[Any] = [self.tokenizer]
            seen: set[int] = set()
            while pending and len(candidates) < 16:
                candidate = pending.pop(0)
                if candidate is None or id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                candidates.append(candidate)

                get_real_tokenizer = getattr(candidate, "get_real_tokenizer", None)
                if callable(get_real_tokenizer):
                    try:
                        pending.append(get_real_tokenizer())
                    except Exception:
                        pass
                for attribute in (
                    "tokenizer",
                    "_tokenizer",
                    "backend_tokenizer",
                    "model",
                    "sp_model",
                ):
                    try:
                        pending.append(getattr(candidate, attribute, None))
                    except Exception:
                        pass
            self._tokenizer_byte_candidates = candidates

        token_pieces: List[str] = []
        for candidate in candidates:
            for method_name in ("token_id_to_bytes", "decode_single_token_bytes"):
                method = getattr(candidate, method_name, None)
                if not callable(method):
                    continue
                try:
                    value = method(token_id)
                except Exception:
                    continue
                raw_bytes = normalize_raw_bytes(value)
                if raw_bytes is not None:
                    return raw_bytes
                if isinstance(value, str):
                    token_pieces.append(value)

            decoder = getattr(candidate, "decoder", None)
            decoder_get = getattr(decoder, "get", None)
            if callable(decoder_get):
                try:
                    value = decoder_get(token_id)
                except Exception:
                    value = None
                raw_bytes = normalize_raw_bytes(value)
                if raw_bytes is not None:
                    return raw_bytes
                if isinstance(value, str):
                    token_pieces.append(value)

            for method_name in (
                "convert_ids_to_tokens",
                "_convert_id_to_token",
                "id_to_token",
                "id_to_piece",
                "IdToPiece",
            ):
                method = getattr(candidate, method_name, None)
                if not callable(method):
                    continue
                try:
                    value = method(token_id)
                except Exception:
                    continue
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                raw_bytes = normalize_raw_bytes(value)
                if raw_bytes is not None:
                    return raw_bytes
                if isinstance(value, str):
                    token_pieces.append(value)

        for piece in token_pieces:
            if (
                len(piece) == 6
                and piece.startswith(("<0x", "<0X"))
                and piece.endswith(">")
            ):
                try:
                    return [int(piece[3:5], 16)]
                except ValueError:
                    pass

            if not piece:
                continue
            for candidate in candidates:
                byte_decoder = getattr(candidate, "byte_decoder", None)
                decoder_get = getattr(byte_decoder, "get", None)
                if not callable(decoder_get):
                    continue
                try:
                    values = [decoder_get(char) for char in piece]
                except Exception:
                    continue
                if all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value <= 255
                    for value in values
                ):
                    return list(values)

        return None

    def _generate_log_probs_from_tensors(
        self,
        request: ChatCompletionRequest,
        output_ids: Optional[torch.Tensor],
        token_logprobs: Optional[torch.Tensor],
        top_logprob_token_ids: Optional[torch.Tensor],
        top_logprobs: Optional[torch.Tensor],
        token_count: Optional[int] = None,
    ) -> Optional[List[ChatCompletionTokenLogprob]]:
        if not request.logprobs:
            return None
        if output_ids is None:
            return None
        if token_logprobs is None:
            raise RuntimeError(
                "token_logprobs is None when logprobs is true; this is an internal bug"
            )

        if output_ids.dtype == torch.bool or output_ids.is_floating_point():
            raise RuntimeError("output_ids must use an integer dtype")
        if not token_logprobs.is_floating_point():
            raise RuntimeError("token_logprobs must use a floating-point dtype")

        output_id_values = output_ids.detach().cpu().reshape(-1)
        raw_token_count = output_id_values.numel()
        if token_count is not None:
            if token_count < 0 or token_count > raw_token_count:
                raise RuntimeError(
                    f"invalid logprob token_count={token_count}, output token count={raw_token_count}"
                )
            output_id_values = output_id_values[:token_count]

        num_tokens = output_id_values.numel()
        if num_tokens == 0:
            return None

        token_logprob_values = token_logprobs.detach().cpu().reshape(-1)
        if token_logprob_values.numel() != raw_token_count:
            raise RuntimeError(
                "token_logprobs must align one-to-one with output_ids: "
                f"got {token_logprob_values.numel()} values for {raw_token_count} tokens"
            )
        token_logprob_values = token_logprob_values[:num_tokens]

        requested_k = request.top_logprobs if request.top_logprobs is not None else 0
        actual_k = 0
        top_token_values: Optional[torch.Tensor] = None
        top_logprob_values: Optional[torch.Tensor] = None
        if requested_k > 0:
            if top_logprob_token_ids is None or top_logprobs is None:
                raise RuntimeError(
                    "top-logprob tensors are missing when top_logprobs is positive"
                )
            if (
                top_logprob_token_ids.dtype == torch.bool
                or top_logprob_token_ids.is_floating_point()
            ):
                raise RuntimeError("top_logprob_token_ids must use an integer dtype")
            if not top_logprobs.is_floating_point():
                raise RuntimeError("top_logprobs must use a floating-point dtype")
            if top_logprob_token_ids.shape != top_logprobs.shape:
                raise RuntimeError(
                    "top-logprob token ids and values must have the same shape"
                )
            if top_logprob_token_ids.dim() < 2:
                raise RuntimeError(
                    "top-logprob tensors must have shape [num_output_tokens, top_logprobs]"
                )
            actual_k = top_logprob_token_ids.shape[-1]
            if actual_k > requested_k:
                raise RuntimeError(
                    f"backend returned top_logprobs={actual_k}, exceeding requested {requested_k}"
                )
            leading_token_count = top_logprob_token_ids.numel() // max(actual_k, 1)
            if actual_k == 0:
                leading_token_count = 1
                for dimension in top_logprob_token_ids.shape[:-1]:
                    leading_token_count *= dimension
            if leading_token_count != raw_token_count:
                raise RuntimeError(
                    "top-logprob tensors must align with the output token dimension"
                )
            top_token_values = (
                top_logprob_token_ids.detach()
                .cpu()
                .reshape(raw_token_count, actual_k)[:num_tokens]
            )
            top_logprob_values = (
                top_logprobs.detach()
                .cpu()
                .reshape(raw_token_count, actual_k)[:num_tokens]
            )

        result: List[ChatCompletionTokenLogprob] = []
        for token_index, (token_id_tensor, token_logprob_tensor) in enumerate(
            zip(output_id_values, token_logprob_values)
        ):
            token_id = token_id_tensor.item()
            token = self.tokenizer.decode([token_id])
            token_result = ChatCompletionTokenLogprob(
                token=token,
                bytes=self._token_id_to_bytes(token_id, token),
                logprob=token_logprob_tensor.item(),
                top_logprobs=[],
            )
            if actual_k > 0:
                assert top_token_values is not None
                assert top_logprob_values is not None
                for top_token_id, top_logprob in zip(
                    top_token_values[token_index], top_logprob_values[token_index]
                ):
                    top_token = self.tokenizer.decode([top_token_id.item()])
                    token_result.top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            logprob=top_logprob.item(),
                            bytes=self._token_id_to_bytes(
                                top_token_id.item(), top_token
                            ),
                        )
                    )
            result.append(token_result)

        logging.debug("chat_logprobs: %s", result)
        return result

    def _accumulate_log_probs_from_tensors(
        self,
        status: Union[StreamStatus, StreamStatusSync],
        output_ids: Optional[torch.Tensor],
        token_logprobs: Optional[torch.Tensor],
        top_logprob_token_ids: Optional[torch.Tensor],
        top_logprobs: Optional[torch.Tensor],
    ) -> None:
        """Append logprobs for retained tokens and keep them pending until visible.

        A streamed chunk may contain several speculative tokens.  Stop-word
        handling can retain only a prefix of that chunk, while UTF-8 and partial
        stop-word handling can delay the corresponding text.  Keeping the
        records on the status makes logprobs follow the same visibility rules as
        text and prevents them from being emitted twice by the final flush.
        """
        if not status.request.logprobs:
            return

        retained_token_count = len(status.output_ids)
        emitted_token_count = status.emitted_logprob_token_count
        if retained_token_count < emitted_token_count:
            # This can only happen if a multi-token stop word retracts a prefix
            # that was already exposed.  Text cannot be retracted either, so keep
            # the status internally consistent and make the condition visible.
            logging.warning(
                "retained token count %s is smaller than emitted logprob count %s",
                retained_token_count,
                emitted_token_count,
            )
            status.pending_logprobs = []
            status.emitted_logprob_token_count = retained_token_count
            emitted_token_count = retained_token_count

        allowed_pending_count = retained_token_count - emitted_token_count
        if len(status.pending_logprobs) > allowed_pending_count:
            status.pending_logprobs = status.pending_logprobs[:allowed_pending_count]

        accounted_token_count = emitted_token_count + len(status.pending_logprobs)
        new_retained_token_count = retained_token_count - accounted_token_count
        if new_retained_token_count == 0:
            return
        if output_ids is None or new_retained_token_count > output_ids.numel():
            raise RuntimeError(
                "retained logprob token count does not align with the output chunk"
            )

        current_records = self._generate_log_probs_from_tensors(
            status.request,
            output_ids,
            token_logprobs,
            top_logprob_token_ids,
            top_logprobs,
            new_retained_token_count,
        )
        if current_records:
            status.pending_logprobs.extend(current_records)

    def _trim_pending_logprobs_to_visible_text(
        self,
        status: Union[StreamStatus, StreamStatusSync],
        visible_delta: str,
    ) -> None:
        """Drop pending records that do not contribute to the visible delta.

        Token-level stop words are already removed from ``status.output_ids``.
        This extra pass covers both a stop string found by the decoded-text path
        and an incomplete UTF-8 suffix discarded when generation terminates.
        """
        if not status.pending_logprobs:
            return

        if not visible_delta:
            status.pending_logprobs = []
            return

        output_ids = list(status.output_ids)
        context_count = status.emitted_logprob_token_count
        pending_count = len(status.pending_logprobs)
        if context_count + pending_count > len(output_ids):
            return

        context_ids = output_ids[:context_count]
        pending_ids = output_ids[context_count : context_count + pending_count]
        context_text = self.tokenizer.decode(context_ids)
        found_context_preserving_prefix = False
        keep_count = 0
        for index in range(1, pending_count + 1):
            candidate_text = self.tokenizer.decode(context_ids + pending_ids[:index])
            if not candidate_text.startswith(context_text):
                continue
            found_context_preserving_prefix = True
            candidate_delta = candidate_text[len(context_text) :]
            if candidate_delta == visible_delta or candidate_delta.startswith(
                visible_delta
            ):
                # The first matching prefix is the smallest token set that
                # accounts for all visible text.  ``candidate_delta`` may be
                # longer when the truncated UTF-8/stop suffix shares its final
                # token with visible text; that token's logprob must remain.
                keep_count = index
                break
            if visible_delta.startswith(candidate_delta):
                # This prefix is wholly visible, but later tokens may still be
                # required (for example byte-fallback pieces that only decode
                # after the following token arrives).
                keep_count = index

        # Some context-sensitive tokenizers do not preserve the decoded prefix.
        # In that case, do not guess away valid records. If decoding was stable
        # but no pending prefix is visible, every pending record belongs to the
        # truncated suffix.
        if found_context_preserving_prefix:
            status.pending_logprobs = status.pending_logprobs[:keep_count]

    @staticmethod
    def _prepare_decoded_string_for_emit(
        status: Union[StreamStatus, StreamStatusSync],
        decoded_string: str,
        is_streaming: bool,
    ) -> Tuple[Optional[str], bool]:
        """Handle an incomplete decoded UTF-8 suffix before emitting a chunk.

        Without logprobs, preserve the renderer's historical behavior: streaming
        waits for a trailing replacement character, while non-streaming removes
        it and emits the valid prefix.  With logprobs, tokens must stay pending
        until they either become decodable or generation ends.  At termination,
        emit the valid prefix and report that the invisible suffix was removed so
        its pending logprob records can be trimmed as well.

        Returns ``(None, False)`` when the caller should keep buffering.  The
        boolean is true only when a terminal invisible suffix was removed.
        """
        if not status.request.logprobs:
            if is_streaming and decoded_string.endswith("\ufffd"):
                return None, False
            if not is_streaming:
                decoded_string = decoded_string.rstrip("\ufffd")
            return decoded_string, False

        has_uncommitted_tokens = len(status.output_ids) > len(status.last_output_ids)
        if not has_uncommitted_tokens:
            return decoded_string, False

        has_incomplete_suffix = not decoded_string or decoded_string.endswith("\ufffd")
        if not has_incomplete_suffix:
            return decoded_string, False
        if status.finish_reason is None:
            return None, False
        return decoded_string.rstrip("\ufffd"), True

    @staticmethod
    def _take_pending_logprobs(
        status: Union[StreamStatus, StreamStatusSync],
    ) -> Optional[List[ChatCompletionTokenLogprob]]:
        if not status.pending_logprobs:
            return None
        result = status.pending_logprobs
        status.pending_logprobs = []
        status.emitted_logprob_token_count += len(result)
        return result

    async def _generate_log_probs(
        self,
        status: StreamStatus,
        output: Optional[GenerateOutput],
        token_count: Optional[int] = None,
    ) -> Optional[List[ChatCompletionTokenLogprob]]:
        assert output is not None
        return self._generate_log_probs_from_tensors(
            status.request,
            output.output_ids,
            output.token_logprobs,
            output.top_logprob_token_ids,
            output.top_logprobs,
            token_count,
        )

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

    def _process_stop_words(
        self,
        delta_string: str,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
        status: StreamStatus,
    ) -> Tuple[str, bool]:
        """
        Process stop words in decoded text: truncate at complete stop words and detect partial ones.

        This method operates on string-level stop words AFTER token-level truncation has been
        performed by _remove_stop_word_ids(). It handles cases where stop words appear within
        decoded strings but not at token boundaries.

        Args:
            delta_string: The decoded text to process
            stop_words_str: List of complete stop word strings to truncate at
            stop_word_slice_list: List of partial stop word prefixes for buffering detection
            is_streaming: Whether in streaming mode (affects buffering behavior)
            status: Stream status object (finish_reason will be updated if truncated)

        Returns:
            (truncated_string, should_buffer):
            - truncated_string: Text truncated at first complete stop word (if found)
            - should_buffer: True if should buffer this chunk in streaming mode
                            (text ends with partial stop word but no complete stop word found)

        Side effects:
            - Sets status.finish_reason = FinisheReason.stop if complete stop word found

        Example scenarios:
            1. Complete stop word: "hello<|observation|>" with stop_words_str=["<|observation|>"]
               -> Returns ("hello", False), sets finish_reason=stop

            2. Partial stop word (streaming): "hello<|obs" with stop_word_slice_list=["<|observation|>"]
               -> Returns ("hello<|obs", True), buffers for next chunk

            3. No stop word: "hello world"
               -> Returns ("hello world", False)
        """
        if not delta_string:
            return delta_string, False

        # Truncate at complete stop words
        truncated = delta_string
        if stop_words_str:
            truncated = truncate_response_with_stop_words(
                delta_string, stop_words_str, is_streaming
            )
            if len(truncated) < len(delta_string):
                status.finish_reason = FinisheReason.stop

        # Check if should buffer (only if didn't truncate at complete stop word)
        # In non-streaming mode, never buffer since all tokens arrive at once
        should_buffer = (
            is_streaming
            and status.finish_reason != FinisheReason.stop
            and is_truncated(truncated, stop_word_slice_list, is_streaming, True)
        )

        return truncated, should_buffer

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
        self._accumulate_log_probs_from_tensors(
            status,
            output.output_ids,
            output.token_logprobs,
            output.top_logprob_token_ids,
            output.top_logprobs,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        decoded_string, trimmed_incomplete_utf8 = self._prepare_decoded_string_for_emit(
            status, decoded_string, is_streaming
        )
        if decoded_string is None:
            return await self._create_empty_delta(output.aux_info)
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]

        # Process stop words: truncate complete stop words, detect partial stop words
        untruncated_delta = status.delta_output_string
        status.delta_output_string, should_buffer = self._process_stop_words(
            untruncated_delta,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
            status,
        )
        if trimmed_incomplete_utf8 or len(status.delta_output_string) < len(
            untruncated_delta
        ):
            self._trim_pending_logprobs_to_visible_text(
                status, status.delta_output_string
            )

        if should_buffer:
            return await self._create_empty_delta(output.aux_info)

        # Build delta output
        if len(status.delta_output_string) > 0:
            current_logprobs = self._take_pending_logprobs(status)
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=current_logprobs,
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
                reuse_length=output.aux_info.reuse_len,
            )
            status.delta_output_string = ""
            return delta
        else:
            current_logprobs = self._take_pending_logprobs(status)
            if current_logprobs:
                status.update_result()
            return await self._create_empty_delta(output.aux_info, current_logprobs)

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
                    if think_status.think_buffer.startswith(self.think_start_tag):
                        think_status.think_buffer = think_status.think_buffer[
                            len(self.think_start_tag) :
                        ]

                    if think_status.think_buffer.endswith(self.think_end_tag):
                        reasoning_text = think_status.think_buffer[
                            : -len(self.think_end_tag)
                        ]
                        think_status.think_buffer = ""
                        think_status.in_think_mode = False
                    elif has_overlap_kmp(
                        think_status.think_buffer, self.think_end_tag
                    ) or self.think_start_tag.startswith(think_status.think_buffer):
                        pass
                    else:
                        reasoning_text = think_status.think_buffer
                    processing_index += 1
                else:
                    content += item.output_str[processing_index:]
                    processing_index = output_len

            if think_status.in_think_mode:
                if has_overlap_kmp(
                    think_status.think_buffer, self.think_end_tag
                ) or self.think_start_tag.startswith(think_status.think_buffer):
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
            # OpenAI exposes token logprobs only for ``content``. Keep the raw
            # model text in that field when probabilities are requested so the
            # token strings and ``delta.content`` stay one-to-one. The C++ sync
            # renderer already follows this contract; without this guard the
            # Python async path moved thinking text to ``reasoning_content`` but
            # still attached all of its probabilities to ``logprobs.content``.
            if think_status_list[i].raw_content_for_logprobs and isinstance(
                item.output_str, str
            ):
                delta = DeltaMessage(content=item.output_str)
            else:
                delta = self._split_reasoning_text_and_content(
                    item, think_status_list[i]
                )
            all_choices.append(
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=delta,
                    logprobs=(
                        ChoiceLogprobs(
                            content=item.logprobs,
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

    def _should_yield_stream_response(
        self, response: StreamResponseObject, is_final: bool = False
    ) -> bool:
        return True

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
                    self._take_pending_logprobs(buffer),
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
        return self.think_mode

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
        raw_content_for_logprobs = bool(generate_config.return_logprobs)
        think_status_list = [
            ThinkStatus(
                enable_think_mode=bool(self.in_think_mode(request)),
                # Raw logprob mode exposes thinking text as ordinary content so
                # its tokens remain one-to-one with ``logprobs.content``. Do not
                # simultaneously classify those same tokens as hidden reasoning.
                in_think_mode=(
                    False
                    if raw_content_for_logprobs
                    else bool(self.should_process_think(request))
                ),
                think_buffer="",
                think_tokens=0,
                is_streaming=generate_config.is_streaming,
                raw_content_for_logprobs=raw_content_for_logprobs,
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
            stream_response = await self._generate_stream_response(
                delta_list, think_status_list
            )
            if self._should_yield_stream_response(stream_response):
                yield stream_response
            if self._check_all_finished(status_list):
                break
        if index != 0:
            flush_response = await self._flush_buffer(
                status_list,
                generate_config.stop_words_str,
                generate_config.is_streaming,
                think_status_list,
            )
            if self._should_yield_stream_response(flush_response):
                yield flush_response
            final_response = await self._generate_final(
                status_list, request, think_status_list
            )
            if self._should_yield_stream_response(final_response, is_final=True):
                yield final_response

    def _create_empty_delta_sync(
        self,
        input_len: int,
        output_len: int,
        reuse_len: int,
        logprobs: Optional[List[ChatCompletionTokenLogprob]] = None,
    ):
        return OutputDelta(
            output_str="",
            logprobs=logprobs,
            input_length=input_len,
            output_length=output_len,
            reuse_length=reuse_len,
        )

    def _generate_log_probs_sync(
        self,
        status: StreamStatusSync,
        token_logprobs: Optional[torch.Tensor],
        top_logprob_token_ids: Optional[torch.Tensor],
        top_logprobs: Optional[torch.Tensor],
        output_ids: Optional[torch.Tensor],
        token_count: Optional[int] = None,
    ) -> Optional[List[ChatCompletionTokenLogprob]]:
        return self._generate_log_probs_from_tensors(
            status.request,
            output_ids,
            token_logprobs,
            top_logprob_token_ids,
            top_logprobs,
            token_count,
        )

    def _update_single_status_sync(
        self,
        status: StreamStatusSync,
        input_len: int,  # output.aux_info
        output_len: int,  # output.aux_info
        reuse_len: int,  # output.aux_info
        token_logprobs: Optional[torch.Tensor],
        top_logprob_token_ids: Optional[torch.Tensor],
        top_logprobs: Optional[torch.Tensor],
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
        self._accumulate_log_probs_from_tensors(
            status,
            output_ids,
            token_logprobs,
            top_logprob_token_ids,
            top_logprobs,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        decoded_string, trimmed_incomplete_utf8 = self._prepare_decoded_string_for_emit(
            status, decoded_string, is_streaming
        )
        if decoded_string is None:
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]

        # Process stop words: truncate complete stop words, detect partial stop words
        untruncated_delta = status.delta_output_string
        status.delta_output_string, should_buffer = self._process_stop_words(
            untruncated_delta,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
            status,
        )
        if trimmed_incomplete_utf8 or len(status.delta_output_string) < len(
            untruncated_delta
        ):
            self._trim_pending_logprobs_to_visible_text(
                status, status.delta_output_string
            )

        if should_buffer:
            return self._create_empty_delta_sync(input_len, output_len, reuse_len)

        # Build delta output
        if len(status.delta_output_string) > 0:
            current_logprobs = self._take_pending_logprobs(status)
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=current_logprobs,
                input_length=input_len,
                output_length=output_len,
                reuse_length=reuse_len,
            )
            status.delta_output_string = ""
            return delta
        else:
            current_logprobs = self._take_pending_logprobs(status)
            if current_logprobs:
                status.update_result()
            return self._create_empty_delta_sync(
                input_len, output_len, reuse_len, current_logprobs
            )

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
                            content=item.logprobs,
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
        token_logprobs_list,
        top_logprob_token_ids_list,
        top_logprobs_list,
        output_ids_list,
        stop_words_str: List[str],
        is_streaming: bool,
    ):
        output_items: List[OutputDelta] = []
        for buffer, input_len, output_len, reuse_len in zip(
            buffer_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
        ):
            trunc_string = truncate_response_with_stop_words(
                buffer.delta_output_string, stop_words_str, is_streaming
            )
            output_items.append(
                OutputDelta(
                    trunc_string,
                    self._take_pending_logprobs(buffer),
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

    @staticmethod
    def _parse_sync_response_args(render_args):
        """Accept the compact non-logprob and extended logprob ABI."""
        if len(render_args) == 4:
            output_ids_list, max_new_tokens, stop_words_str, is_streaming = render_args
            return (
                None,
                None,
                None,
                output_ids_list,
                max_new_tokens,
                stop_words_str,
                is_streaming,
            )
        if len(render_args) == 7:
            return render_args
        raise TypeError(
            "sync response expects compact output_ids/config arguments or "
            "the three logprob tensor lists followed by output_ids/config"
        )

    @staticmethod
    def _parse_sync_flush_args(render_args):
        """Accept the minimal, compact, and legacy extended flush ABIs."""
        if len(render_args) == 2:
            stop_words_str, is_streaming = render_args
            return None, None, None, None, stop_words_str, is_streaming
        if len(render_args) == 3:
            output_ids_list, stop_words_str, is_streaming = render_args
            return None, None, None, output_ids_list, stop_words_str, is_streaming
        if len(render_args) == 6:
            return render_args
        raise TypeError(
            "sync flush expects stop/config arguments, compact output_ids/config "
            "arguments, or the three logprob tensor lists followed by "
            "output_ids/config"
        )

    def _render_sync_delta_list(
        self,
        status_list,
        input_len_list,
        output_len_list,
        reuse_len_list,
        token_logprobs_list,
        top_logprob_token_ids_list,
        top_logprobs_list,
        output_ids_list,
        max_new_tokens,
        stop_words_str,
        is_streaming,
    ) -> List[OutputDelta]:
        expected_count = len(status_list)
        if not (
            len(input_len_list)
            == len(output_len_list)
            == len(reuse_len_list)
            == len(output_ids_list)
            == expected_count
        ):
            raise ValueError(
                "sync response core lists must match status_list length "
                f"{expected_count}"
            )

        has_token_logprobs = token_logprobs_list is not None
        has_top_token_ids = top_logprob_token_ids_list is not None
        has_top_logprobs = top_logprobs_list is not None
        if not (has_token_logprobs == has_top_token_ids == has_top_logprobs):
            raise ValueError(
                "sync response logprob lists must be supplied together or omitted"
            )
        if has_token_logprobs and not (
            len(token_logprobs_list)
            == len(top_logprob_token_ids_list)
            == len(top_logprobs_list)
            == expected_count
        ):
            raise ValueError(
                "sync response logprob lists must match status_list length "
                f"{expected_count}"
            )

        stop_word_slice_list = get_stop_word_slices(stop_words_str)
        delta_list: List[OutputDelta] = []
        for index, (status, input_len, output_len, reuse_len, output_ids) in enumerate(
            zip(
                status_list,
                input_len_list,
                output_len_list,
                reuse_len_list,
                output_ids_list,
            )
        ):
            token_logprobs = (
                None if token_logprobs_list is None else token_logprobs_list[index]
            )
            top_logprob_token_ids = (
                None
                if top_logprob_token_ids_list is None
                else top_logprob_token_ids_list[index]
            )
            top_logprobs = (
                None if top_logprobs_list is None else top_logprobs_list[index]
            )
            delta_list.append(
                self._update_single_status_sync(
                    status,
                    input_len,
                    output_len,
                    reuse_len,
                    token_logprobs,
                    top_logprob_token_ids,
                    top_logprobs,
                    output_ids,
                    max_new_tokens,
                    stop_words_str,
                    stop_word_slice_list,
                    is_streaming,
                )
            )
        return delta_list

    def render_stream_response_refactor(
        self,
        status_list: StreamStatusSync,  # pass in from cpp
        input_len_list,  # output.aux_info
        output_len_list,  # output.aux_info
        reuse_len_list,  # output.aux_info
        *render_args,
    ):
        (
            token_logprobs_list,
            top_logprob_token_ids_list,
            top_logprobs_list,
            output_ids_list,
            max_new_tokens,
            stop_words_str,
            is_streaming,
        ) = self._parse_sync_response_args(render_args)
        delta_list = self._render_sync_delta_list(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            token_logprobs_list,
            top_logprob_token_ids_list,
            top_logprobs_list,
            output_ids_list,
            max_new_tokens,
            stop_words_str,
            is_streaming,
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
        *render_args,
    ):
        (
            _token_logprobs_list,
            _top_logprob_token_ids_list,
            _top_logprobs_list,
            _output_ids_list,
            stop_words_str,
            is_streaming,
        ) = self._parse_sync_flush_args(render_args)
        stream_response = self._flush_buffer_sync(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            _token_logprobs_list,
            _top_logprob_token_ids_list,
            _top_logprobs_list,
            _output_ids_list,
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
        *render_args,
    ):
        (
            token_logprobs_list,
            top_logprob_token_ids_list,
            top_logprobs_list,
            output_ids_list,
            max_new_tokens,
            stop_words_str,
            is_streaming,
        ) = self._parse_sync_response_args(render_args)
        delta_list = self._render_sync_delta_list(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            token_logprobs_list,
            top_logprob_token_ids_list,
            top_logprobs_list,
            output_ids_list,
            max_new_tokens,
            stop_words_str,
            is_streaming,
        )
        stream_response = self._generate_stream_response_sync(delta_list)
        return stream_response

    def render_stream_response_flush_blocking(
        self,
        status_list,
        input_len_list,
        output_len_list,
        reuse_len_list,
        *render_args,
    ):
        (
            _token_logprobs_list,
            _top_logprob_token_ids_list,
            _top_logprobs_list,
            _output_ids_list,
            stop_words_str,
            is_streaming,
        ) = self._parse_sync_flush_args(render_args)
        stream_response = self._flush_buffer_sync(
            status_list,
            input_len_list,
            output_len_list,
            reuse_len_list,
            _token_logprobs_list,
            _top_logprob_token_ids_list,
            _top_logprobs_list,
            _output_ids_list,
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
            text_results = text.split(self.think_end_tag, 1)
            reasoning_content = text_results[0] if len(text_results) == 2 else None
            content = text_results[1] if len(text_results) == 2 else text
            return content, reasoning_content

        for response in choice_generator:

            if len(response.choices) != len(all_choices):
                if all_choices == []:
                    for i, choice in enumerate(response.choices):
                        all_choices.append(
                            ChatCompletionResponseChoice(
                                index=i,
                                message=ChatMessage(
                                    role=choice.delta.role or RoleEnum.assistant,
                                    content=choice.delta.content or None,
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

        for choice in all_choices:
            if choice.logprobs is None:
                content, reasoning_content = split_think_tag(choice.message.content)
                choice.message.content = content
                choice.message.reasoning_content = reasoning_content
            # Logprob mode intentionally leaves raw thinking content untouched
            # so every visible token stays aligned with one probability record.

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
        """
        Truncate token sequence at FIRST occurrence of stop word (eos or stop_word_ids).

        This is the first phase of stop word processing, operating at the token level.
        It handles cases where stop words appear as token sequences, especially important
        for speculative sampling (MTP) where multiple tokens are generated at once.

        Args:
            output_ids: Complete output token sequence to truncate
            delta_output_ids: New tokens in this chunk (unused in current implementation)

        Returns:
            Truncated output_ids list, with everything from first stop word removed

        Behavior:
            1. Check for eos_token_id anywhere in sequence, truncate at first occurrence
            2. Check for stop_word_ids sequences, truncate at first occurrence
            3. If multiple stop words found, truncate at the earliest position

        Why this is needed:
            - In speculative sampling, multiple tokens may be generated at once
            - Stop word could appear anywhere in the multi-token chunk, not just at end
            - Example: tokens [A, B, STOP, C, D] should truncate to [A, B]

        Note:
            String-level truncation is still needed after this (see _process_stop_words)
            because stop words may span multiple tokens or appear mid-token.
        """
        stop_word_ids_list_all = (
            self.get_all_extra_stop_word_ids_list() + self.stop_words_id_list
        )

        # Find earliest position of any stop token (EOS or stop word sequence)
        min_stop_pos = len(output_ids)

        # Check for eos token position
        if self.eos_token_id in output_ids:
            eos_pos = output_ids.index(self.eos_token_id)
            min_stop_pos = min(min_stop_pos, eos_pos)

        # Check for stop word sequences - find first occurrence of each
        for stop_word_ids in stop_word_ids_list_all:
            if not stop_word_ids:
                continue
            stop_len = len(stop_word_ids)
            # Scan through output_ids to find first occurrence
            for i in range(len(output_ids) - stop_len + 1):
                if output_ids[i : i + stop_len] == stop_word_ids:
                    min_stop_pos = min(min_stop_pos, i)
                    break

        # Truncate at earliest stop position found
        if min_stop_pos < len(output_ids):
            output_ids = output_ids[:min_stop_pos]

        return output_ids
