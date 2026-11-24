import copy
import json
import re
from uuid import uuid4

from jinja2 import Environment
from typing_extensions import override

from rtp_llm.multimodal.glm54_mock_vit_input import (
    GLM54_MOCK_IMAGE_TOKEN,
    UNSUPPORTED_IMAGE_OPTIONS,
    reject_image_option,
    validate_image_url,
    validate_pixel_bounds,
)
from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPart,
    ContentPartTypeEnum,
    MMPreprocessConfigPart,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.custom_renderer import (
    RenderedInputs,
    StreamStatus,
    ThinkStatus,
)
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolStreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.ops import MMPreprocessConfig
from rtp_llm.utils.base_model_datatypes import MMUrlType


class _Glm54DirectAnswerParser(ReasoningParser):
    """Remove leading closing tags echoed after the prompt closed thinking."""

    _OPEN = "<think>"
    _CLOSE = "</think>"
    _LEADING_CLOSES = re.compile(r"(?:\s*</think>)+")

    def __init__(self):
        super().__init__(model_type="glm45")
        self._leading_text = ""
        self._at_start = True

    def _strip_leading_closes(self, text: str) -> str:
        match = self._LEADING_CLOSES.match(text)
        return text[match.end() :] if match else text

    def parse_non_stream(self, full_text: str) -> tuple[str, str]:
        return super().parse_non_stream(self._strip_leading_closes(full_text))

    def parse_stream_chunk(self, chunk_text: str) -> tuple[str, str]:
        if self._at_start:
            self._leading_text = self._strip_leading_closes(
                self._leading_text + chunk_text
            )
            prefix = self._leading_text.lstrip()
            if self._CLOSE.startswith(prefix) or (
                self._OPEN.startswith(prefix) and prefix != self._OPEN
            ):
                return "", ""
            chunk_text = (
                prefix if prefix.startswith(self._OPEN) else self._leading_text
            )
            self._leading_text = ""
            self._at_start = False
        return super().parse_stream_chunk(chunk_text)

    def finish_stream(self) -> str:
        pending_text = self._leading_text
        self._leading_text = ""
        self._at_start = False
        return pending_text


class Glm54MockVitRenderer(ChatGlm45Renderer):
    """Flash-template renderer for the GLM5.4 mock image checkpoint."""

    @override
    def _customize_jinja_env(self, env: Environment) -> None:
        # The GLM-5.3 checkpoint serializes individual tool fields with tojson.
        # Strings must retain JSON quotes and escaping in those fields.
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, sort_keys=False, ensure_ascii=kwargs.get("ensure_ascii", False)
        )

    @override
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        # This checkpoint's template unconditionally opens every generated
        # assistant turn with ``<think>``; it does not consume
        # ``enable_thinking``.  Keep the parser installed even when a caller
        # asks to disable thinking, otherwise the hidden preamble and literal
        # ``</think>`` leak into ``content``.
        return True

    @override
    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        prompt = super()._build_prompt(request)
        if request.disable_thinking() and prompt.endswith("<think>"):
            # GLM-5.3-Flash's checkpoint template always opens the generated
            # assistant turn with <think> and does not consume
            # enable_thinking.  Close only that final generation prefix so an
            # explicit opt-out starts on the direct-answer path.  If a newer
            # checkpoint template already handles the flag, leave it intact.
            prompt += "</think>"
        return prompt

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> ReasoningParser | None:
        if request.disable_thinking():
            return _Glm54DirectAnswerParser()
        return super()._create_reasoning_parser(request)

    @override
    async def _flush_buffer(
        self,
        buffer_list: list[StreamStatus],
        stop_words_str: list[str],
        is_streaming: bool,
        think_status_list: list[ThinkStatus],
    ):
        if is_streaming:
            for status in buffer_list:
                if isinstance(status, ReasoningToolStreamStatus) and isinstance(
                    status.reasoning_parser, _Glm54DirectAnswerParser
                ):
                    # The stream ended before this prefix could become a full tag.
                    status.delta_output_string = (
                        status.reasoning_parser.finish_stream()
                        + status.delta_output_string
                    )
        return await super()._flush_buffer(
            buffer_list, stop_words_str, is_streaming, think_status_list
        )

    @staticmethod
    def _preprocess_config(
        config: MMPreprocessConfigPart | None,
    ) -> MMPreprocessConfig:
        if config is None:
            return MMPreprocessConfig()
        for name, _ in UNSUPPORTED_IMAGE_OPTIONS:
            if getattr(config, name) not in (None, -1):
                reject_image_option(name)
        if config.crop_positions:
            reject_image_option("crop_positions")

        min_pixels = -1 if config.min_pixels is None else config.min_pixels
        max_pixels = -1 if config.max_pixels is None else config.max_pixels
        validate_pixel_bounds(min_pixels, max_pixels)
        if not -(2**31) <= config.mm_timeout_ms < 2**31:
            raise_mm("mm_timeout_ms must fit in a 32-bit integer")
        return MMPreprocessConfig(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            mm_timeout_ms=config.mm_timeout_ms or -1,
        )

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        request = copy.deepcopy(request)
        marker_prefix = None
        images_by_marker = {}
        urls = []
        preprocess_configs = []

        for message in request.messages:
            if not isinstance(message.content, list):
                continue
            rewritten = []
            for part in message.content:
                if part.type == ContentPartTypeEnum.image_url:
                    if part.image_url is None:
                        raise_mm(MMErr.URL_INVALID)
                    validate_image_url(part.image_url.url)
                    if marker_prefix is None:
                        marker_prefix = f"__RTPLLM_IMAGE_{uuid4().hex}_"
                    marker = f"{marker_prefix}{len(images_by_marker)}__"
                    images_by_marker[marker] = (
                        part.image_url.url,
                        self._preprocess_config(part.preprocess_config),
                    )
                    rewritten.append(
                        ContentPart(
                            type=ContentPartTypeEnum.text,
                            text=marker + "\n",
                        )
                    )
                elif part.type == ContentPartTypeEnum.text:
                    if part.text is None:
                        raise_mm("Text content requires a text field")
                    rewritten.append(part)
                else:
                    raise_mm(
                        f"GLM5.4 mock ViT does not support {part.type.value} content"
                    )
            message.content = rewritten

        prompt = self._build_prompt(request)
        if GLM54_MOCK_IMAGE_TOKEN in prompt:
            raise_mm("GLM5.4 mock ViT image placeholders are reserved for image inputs")

        def restore_image(match: re.Match[str]) -> str:
            image = images_by_marker.pop(match[0], None)
            if image is None:
                raise_mm("GLM5.4 mock ViT template repeated an image placeholder")
            url, preprocess_config = image
            urls.append(url)
            preprocess_configs.append(preprocess_config)
            return GLM54_MOCK_IMAGE_TOKEN

        # The Flash template can reorder tool results. Use their rendered order
        # to associate the image embeddings with the corresponding placeholders.
        if marker_prefix is not None:
            prompt = re.sub(
                re.escape(marker_prefix) + r"[0-9]+__", restore_image, prompt
            )
        if images_by_marker:
            raise_mm("GLM5.4 mock ViT image placeholders do not match the image inputs")
        return RenderedInputs(
            input_ids=self.tokenizer.encode(prompt),
            input_urls=urls,
            rendered_prompt=prompt,
            input_urls_type=[MMUrlType.IMAGE] * len(urls),
            preprocess_configs=preprocess_configs,
        )


register_renderer("glm5_4_mock_vit", Glm54MockVitRenderer)
