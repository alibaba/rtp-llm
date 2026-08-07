import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from typing_extensions import override

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    K3_MAX_IMAGE_FILE_SIZE_KB,
    KimiK3VisionProcessor,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest, DeltaMessage
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    StreamStatus,
)
from rtp_llm.multimodal.multimodal_util import MMUrlType, get_bytes_io_from_url

_K3_MEDIA_PREFLIGHT_CONCURRENCY = 4
_K3_MEDIA_EXECUTOR = ThreadPoolExecutor(
    max_workers=_K3_MEDIA_PREFLIGHT_CONCURRENCY,
    thread_name_prefix="kimi-k3-media",
)


class _KimiK3StreamStatus(StreamStatus):
    """Per-choice state for parsing K3's generated XTML channels."""

    def __init__(self, request: ChatCompletionRequest):
        super().__init__(request)
        self.xtml_pending = ""
        self.in_reasoning = not request.disable_thinking()
        self.response_closed = False


class KimiK3Renderer(CustomChatRenderer):
    """Render Kimi K3's Python-defined XTML and collect image inputs.

    K3 deliberately has no Jinja ``chat_template``.  Its remote tokenizer
    renders a sequence of trusted structural segments and untrusted text
    segments, then encodes the two with different special-token policies.
    Calling ``encode`` on the final debug string would lose that distinction,
    so this renderer consumes the tokenizer's tokenized result directly.
    """

    MAX_IMAGES_PER_REQUEST = 16
    MAX_IMAGE_BYTES = K3_MAX_IMAGE_FILE_SIZE_KB * 1024
    MAX_TOTAL_IMAGE_BYTES = 128 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_processor = KimiK3VisionProcessor()
        self.add_extra_stop_words(["<|end_of_msg|>"])

    @staticmethod
    def _split_marker_prefix(text: str, marker: str) -> tuple[str, str]:
        """Keep the longest suffix that may be a split XTML marker."""

        max_prefix = min(len(text), len(marker) - 1)
        for length in range(max_prefix, 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length], text[-length:]
        return text, ""

    @classmethod
    def _parse_xtml_delta(
        cls, status: _KimiK3StreamStatus, text: str, flush: bool = False
    ) -> DeltaMessage:
        """Split K3 reasoning/content channels and remove their XTML envelope.

        In thinking mode the generation prompt has already opened the
        ``think`` channel. The model then emits this transition before the
        visible answer::

            <|close|>think<|sep|><|open|>response<|sep|>

        Both thinking and non-thinking modes finish the visible channel with
        ``<|close|>response<|sep|>``. ``<|end_of_msg|>`` is removed by the
        generic stop path, but the other XTML tokens are ordinary generated
        tokens. Parse the exact channel boundaries here so they never leak
        into OpenAI ``content`` and the reasoning text is exposed through
        ``reasoning_content``. Partial markers are buffered across streaming
        chunks.
        """

        if status.response_closed:
            return DeltaMessage(reasoning_content="", content="")

        think_to_response = "<|close|>think<|sep|><|open|>response<|sep|>"
        response_closure = "<|close|>response<|sep|>"
        combined = status.xtml_pending + text
        status.xtml_pending = ""
        reasoning = ""
        content = ""

        if status.in_reasoning:
            transition_at = combined.find(think_to_response)
            if transition_at < 0:
                if flush:
                    reasoning = combined
                else:
                    reasoning, status.xtml_pending = cls._split_marker_prefix(
                        combined, think_to_response
                    )
                return DeltaMessage(reasoning_content=reasoning, content="")
            reasoning = combined[:transition_at]
            combined = combined[transition_at + len(think_to_response) :]
            status.in_reasoning = False

        closure_at = combined.find(response_closure)
        if closure_at >= 0:
            content = combined[:closure_at]
            status.response_closed = True
        elif flush:
            content = combined
        else:
            content, status.xtml_pending = cls._split_marker_prefix(
                combined, response_closure
            )

        return DeltaMessage(reasoning_content=reasoning, content=content)

    @override
    async def _create_status_list(
        self, n: int, request: ChatCompletionRequest
    ) -> List[StreamStatus]:
        return [_KimiK3StreamStatus(request) for _ in range(n)]

    @override
    async def _update_single_status(
        self,
        status: StreamStatus,
        output,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        delta = await super()._update_single_status(
            status,
            output,
            max_new_tokens,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
        )
        if isinstance(status, _KimiK3StreamStatus) and isinstance(
            delta.output_str, str
        ):
            delta.output_str = self._parse_xtml_delta(
                status,
                delta.output_str,
                flush=status.finish_reason is not None,
            )
        return delta

    @override
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        return not request.disable_thinking()

    @override
    def should_process_think(self, request: ChatCompletionRequest) -> bool:
        del request
        # _parse_xtml_delta returns DeltaMessage with the channels already
        # separated; the generic <think> tag parser must not process it again.
        return False

    @staticmethod
    def _request_dict(request: ChatCompletionRequest) -> Dict[str, Any]:
        return request.model_dump(exclude_none=True, mode="json")

    @staticmethod
    def _collect_and_rewrite(
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], PromptWithMMInput]:
        urls: List[str] = []
        types: List[MMUrlType] = []
        rewritten: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, list):
                rewritten.append(message)
                continue

            new_parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Kimi K3 message content parts must be objects")
                part_type = part.get("type")
                if part_type == "text":
                    new_parts.append({"type": "text", "text": part.get("text")})
                elif part_type == "image_url":
                    if role != "user":
                        raise ValueError(
                            "Kimi K3 supports image_url content only in user "
                            f"messages; got role {role!r}"
                        )
                    image_url = part.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                    if not isinstance(url, str) or not url:
                        raise ValueError(
                            "Kimi K3 image_url content requires a non-empty URL"
                        )
                    urls.append(url)
                    types.append(MMUrlType.IMAGE)
                    new_parts.append({"type": "image", "image": url})
                else:
                    raise ValueError(
                        "Kimi K3 supports only text and image_url content parts; "
                        f"got {part_type!r}"
                    )

            new_message = dict(message)
            new_message["content"] = new_parts
            rewritten.append(new_message)

        return rewritten, PromptWithMMInput(prompt="", urls=urls, mm_types=types)

    def _preflight_one(self, url: str) -> tuple[torch.Tensor, tuple[int, int]]:
        data = get_bytes_io_from_url(
            url,
            self.vit_config.download_headers,
            max_file_size_kb=self.MAX_IMAGE_BYTES // 1024,
        )
        raw = data.getbuffer()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
        return torch.frombuffer(raw, dtype=torch.uint8), (width, height)

    def _validate_image_count(self, urls: List[str]) -> None:
        if len(urls) > self.MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                "Kimi K3 image count exceeds the per-request limit: "
                f"{len(urls)} > {self.MAX_IMAGES_PER_REQUEST}"
            )

    def _append_preflight_batch(
        self,
        results: List[tuple[torch.Tensor, tuple[int, int]]],
        tensors: List[torch.Tensor],
        metadata: List[tuple[int, int]],
        total_bytes: int,
    ) -> int:
        for tensor, size in results:
            total_bytes += tensor.numel()
            if total_bytes > self.MAX_TOTAL_IMAGE_BYTES:
                raise ValueError(
                    "Kimi K3 image bytes exceed the per-request limit: "
                    f"{total_bytes} > {self.MAX_TOTAL_IMAGE_BYTES}"
                )
            tensors.append(tensor)
            metadata.append(size)
        return total_bytes

    def _preflight_media(
        self, urls: List[str]
    ) -> tuple[List[torch.Tensor], List[tuple[int, int]]]:
        self._validate_image_count(urls)
        tensors: List[torch.Tensor] = []
        metadata: List[tuple[int, int]] = []
        total_bytes = 0
        for offset in range(0, len(urls), _K3_MEDIA_PREFLIGHT_CONCURRENCY):
            batch = urls[offset : offset + _K3_MEDIA_PREFLIGHT_CONCURRENCY]
            results = list(_K3_MEDIA_EXECUTOR.map(self._preflight_one, batch))
            total_bytes = self._append_preflight_batch(
                results, tensors, metadata, total_bytes
            )
        return tensors, metadata

    async def _preflight_media_async(
        self, urls: List[str]
    ) -> tuple[List[torch.Tensor], List[tuple[int, int]]]:
        self._validate_image_count(urls)
        loop = asyncio.get_running_loop()
        tensors: List[torch.Tensor] = []
        metadata: List[tuple[int, int]] = []
        total_bytes = 0
        for offset in range(0, len(urls), _K3_MEDIA_PREFLIGHT_CONCURRENCY):
            batch = urls[offset : offset + _K3_MEDIA_PREFLIGHT_CONCURRENCY]
            results = await asyncio.gather(
                *(
                    loop.run_in_executor(
                        _K3_MEDIA_EXECUTOR,
                        self._preflight_one,
                        url,
                    )
                    for url in batch
                )
            )
            total_bytes = self._append_preflight_batch(
                results, tensors, metadata, total_bytes
            )
        return tensors, metadata

    @staticmethod
    def _tools(request_dict: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        tools = request_dict.get("tools")
        if tools:
            return tools
        functions = request_dict.get("functions")
        if functions:
            return [
                {"type": "function", "function": function} for function in functions
            ]
        return None

    @staticmethod
    def _template_kwargs(
        request: ChatCompletionRequest, request_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if request.chat_template_kwargs:
            kwargs.update(request.chat_template_kwargs)
        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
        ):
            kwargs.update(request.extra_configs.chat_template_kwargs)

        # RTP's public spelling is enable_thinking; K3's tokenizer spelling is
        # simply thinking.  An explicit request flag wins over template kwargs.
        if "enable_thinking" in kwargs and "thinking" not in kwargs:
            kwargs["thinking"] = bool(kwargs.pop("enable_thinking"))
        else:
            kwargs.pop("enable_thinking", None)
        kwargs.setdefault("thinking", not request.disable_thinking())
        if request.enable_thinking is not None:
            kwargs["thinking"] = request.enable_thinking

        if request.reasoning_effort is not None:
            kwargs["thinking_effort"] = request.reasoning_effort
        if request_dict.get("tool_choice") is not None:
            kwargs["tool_choice"] = request_dict["tool_choice"]
        if request_dict.get("response_format") is not None:
            kwargs["response_format"] = request_dict["response_format"]
        return kwargs

    @staticmethod
    def _as_token_ids(value: Any) -> List[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError(
                    "Kimi K3 renderer expected one conversation, got a token batch"
                )
            value = value[0]
        if not isinstance(value, list) or not all(
            isinstance(token_id, int) for token_id in value
        ):
            raise TypeError(
                "Kimi K3 tokenizer.apply_chat_template must return List[int] "
                f"for a single request, got {type(value).__name__}"
            )
        return value

    def _validate_visual_token_budget(
        self,
        token_ids: List[int],
        metadata: List[tuple[int, int]],
    ) -> None:
        if not self.max_seq_len or not metadata:
            return
        visual_tokens = sum(
            self._image_processor.resize_config_for_size(width, height)["num_tokens"]
            for width, height in metadata
        )
        expanded_input_length = len(token_ids) - len(metadata) + visual_tokens
        if expanded_input_length > self.max_seq_len:
            raise ValueError(
                "Kimi K3 expanded multimodal input exceeds max_seq_len: "
                f"{expanded_input_length} > {self.max_seq_len}"
            )

    def _render_preflighted(
        self,
        request: ChatCompletionRequest,
        request_dict: Dict[str, Any],
        messages: List[Dict[str, Any]],
        mm_input: PromptWithMMInput,
        tensors: List[torch.Tensor],
        metadata: List[tuple[int, int]],
    ) -> RenderedInputs:
        image_prompts = [
            KimiK3VisionProcessor.make_image_prompt(width, height)
            for width, height in metadata
        ]
        tools = self._tools(request_dict)
        template_kwargs = self._template_kwargs(request, request_dict)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            image_prompts=image_prompts,
            **template_kwargs,
        )
        token_ids = self._as_token_ids(input_ids)
        self._validate_visual_token_budget(token_ids, metadata)
        logging.debug("Kimi K3 rendered %d XTML prompt tokens", len(token_ids))
        # ``rendered_prompt`` is left empty on purpose: rendering K3's XTML template a
        # second time with tokenize=False is not cheap, and the string only feeds
        # debug/logging.  Both endpoints fall back to decoding ``input_ids`` when it is
        # empty (openai_endpoint.py ``_get_debug_info``, OpenaiEndpoint.cc
        # ``getDebugInfo``), so the prompt text is produced on demand instead.
        return RenderedInputs(
            input_ids=token_ids,
            input_urls=mm_input.urls,
            input_urls_type=mm_input.mm_types,
            input_tensors=tensors,
        )

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        request_dict = self._request_dict(request)
        messages, mm_input = self._collect_and_rewrite(request_dict["messages"])
        tensors, metadata = self._preflight_media(mm_input.urls)
        return self._render_preflighted(
            request,
            request_dict,
            messages,
            mm_input,
            tensors,
            metadata,
        )

    @override
    async def render_chat_async(
        self, request: ChatCompletionRequest
    ) -> RenderedInputs:
        request_dict = self._request_dict(request)
        messages, mm_input = self._collect_and_rewrite(request_dict["messages"])
        tensors, metadata = await self._preflight_media_async(mm_input.urls)
        return self._render_preflighted(
            request,
            request_dict,
            messages,
            mm_input,
            tensors,
            metadata,
        )

    @override
    def apply_chat_completion_constraints(
        self, request: ChatCompletionRequest, generate_config: GenerateConfig
    ) -> None:
        del generate_config
        tool_choice = request.tool_choice
        if tool_choice is None or tool_choice in ("auto", "none", "required"):
            return
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS,
            "Kimi K3 currently supports tool_choice='auto', 'none', or "
            "'required'; named tool_choice is not implemented",
        )


register_renderer("kimi_k3", KimiK3Renderer)
