import copy
import json
import logging
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment
from typing_extensions import override

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPart,
    ContentPartTypeEnum,
    MMPreprocessConfigPart,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.utils.base_model_datatypes import MMUrlType


class ChatGlm45Renderer(ReasoningToolBaseRenderer):
    """ChatGLM45Renderer 使用 GLM4MoeDetector 进行工具调用解析"""

    @override
    def _setup_stop_words(self):
        """设置GLM45特定的停止词"""
        self.add_extra_stop_words(["<|user|>", "<|observation|>"])

    @override
    def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
        """预处理消息，确保 tool_calls 中的 arguments 是字典对象"""
        processed_messages = []
        for message in messages:
            processed_message = message.copy()
            if "tool_calls" in processed_message and processed_message["tool_calls"]:
                processed_tool_calls = []
                for tool_call in processed_message["tool_calls"]:
                    processed_tool_call = tool_call.copy()
                    if "function" in processed_tool_call:
                        function = processed_tool_call["function"].copy()
                        if "arguments" in function and isinstance(
                            function["arguments"], str
                        ):
                            try:
                                function["arguments"] = json.loads(
                                    function["arguments"]
                                )
                            except json.JSONDecodeError:
                                function["arguments"] = {}
                        processed_tool_call["function"] = function
                    elif "arguments" in processed_tool_call and isinstance(
                        processed_tool_call["arguments"], str
                    ):
                        try:
                            processed_tool_call["arguments"] = json.loads(
                                processed_tool_call["arguments"]
                            )
                        except json.JSONDecodeError:
                            processed_tool_call["arguments"] = {}
                    processed_tool_calls.append(processed_tool_call)
                processed_message["tool_calls"] = processed_tool_calls
            processed_messages.append(processed_message)
        return processed_messages

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """创建GLM45检测器"""
        if request.tools:
            return Glm4MoeDetector()
        else:
            return None

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        if not self.in_think_mode(request):
            return None

        try:
            rendered_result = self.render_chat(request)
            if rendered_result.rendered_prompt.endswith("<think>"):
                return ReasoningParser(model_type="glm45", force_reasoning=True)
        except Exception as e:
            logging.error(f"Failed to render chat in _create_reasoning_parser: {e}")

        return ReasoningParser(model_type="glm45")

    @override
    def _customize_jinja_env(self, env: Environment) -> None:
        """
        自定义Jinja2环境，子类可以重写此方法来添加自定义过滤器、函数等

        Args:
            env: Jinja2环境对象
            request: 聊天完成请求
            context: 模板渲染上下文
        """
        # 设置默认的tojson过滤器
        env.filters["tojson"] = lambda value, **kwargs: (
            value
            if isinstance(value, str)
            else json.dumps(
                value, sort_keys=False, ensure_ascii=kwargs.get("ensure_ascii", False)
            )
        )


class Glm53FlashRenderer(ChatGlm45Renderer):
    """GLM-5.3 renderer that preserves media inputs for the ViT pipeline."""

    _IMAGE_TOKEN = "<|begin_of_image|><|image|><|end_of_image|>"
    _VIDEO_TOKEN = "<|begin_of_video|><|video|><|end_of_video|>"

    @override
    def _customize_jinja_env(self, env: Environment) -> None:
        # The GLM-5.3 checkpoint serializes individual tool fields with tojson.
        # Strings must retain JSON quotes and escaping in those fields.
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, sort_keys=False, ensure_ascii=kwargs.get("ensure_ascii", False)
        )

    @override
    def apply_chat_completion_constraints(
        self, request: ChatCompletionRequest, generate_config: GenerateConfig
    ) -> None:
        super().apply_chat_completion_constraints(request, generate_config)
        if "top_p" in request.model_fields_set and request.top_p is not None:
            return
        if (
            request.extra_configs is not None
            and "top_p" in request.extra_configs.model_fields_set
        ):
            generate_config.top_p = request.extra_configs.top_p
            return

        # The API schema's 1.0 default must not mask the checkpoint's nucleus
        # sampling default when the client omits top_p.
        if not hasattr(self, "_checkpoint_top_p"):
            ckpt_path = getattr(self, "ckpt_path", None)
            path = Path(ckpt_path) / "generation_config.json" if ckpt_path else None
            defaults = json.loads(path.read_text()) if path and path.is_file() else {}
            self._checkpoint_top_p = defaults.get("top_p")
        if self._checkpoint_top_p is not None:
            top_p = float(self._checkpoint_top_p)
            if not 0 < top_p <= 1:
                raise ValueError("checkpoint top_p must be in (0, 1]")
            generate_config.top_p = top_p

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

    @staticmethod
    def _preprocess_config(content_part, media_url):
        config = (
            content_part.preprocess_config.model_copy()
            if content_part.preprocess_config is not None
            else MMPreprocessConfigPart()
        )
        for field in ("max_long_side_pixel", "fps"):
            value = getattr(media_url, field, None)
            if value is None:
                value = getattr(content_part, field, None)
            if value is not None:
                setattr(config, field, value)
        return get_preprocess_config(config)

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        request = copy.deepcopy(request)
        urls: List[str] = []
        types: List[MMUrlType] = []
        preprocess_configs = []

        for message in request.messages:
            if not isinstance(message.content, list):
                continue
            rewritten = []
            for part in message.content:
                if part.type == ContentPartTypeEnum.image_url:
                    assert part.image_url is not None
                    urls.append(part.image_url.url)
                    types.append(MMUrlType.IMAGE)
                    preprocess_configs.append(
                        self._preprocess_config(part, part.image_url)
                    )
                    rewritten.append(
                        ContentPart(
                            type=ContentPartTypeEnum.text,
                            text=self._IMAGE_TOKEN + "\n",
                        )
                    )
                elif part.type == ContentPartTypeEnum.video_url:
                    assert part.video_url is not None
                    urls.append(part.video_url.url)
                    types.append(MMUrlType.VIDEO)
                    preprocess_configs.append(
                        self._preprocess_config(part, part.video_url)
                    )
                    rewritten.append(
                        ContentPart(
                            type=ContentPartTypeEnum.text,
                            text=self._VIDEO_TOKEN + "\n",
                        )
                    )
                else:
                    rewritten.append(part)
            message.content = rewritten

        prompt = self._build_prompt(request)
        return RenderedInputs(
            input_ids=self.tokenizer.encode(prompt),
            input_urls=urls,
            rendered_prompt=prompt,
            input_urls_type=types,
            preprocess_configs=preprocess_configs,
        )


register_renderer("glm4_moe", ChatGlm45Renderer)
register_renderer("glm_5", ChatGlm45Renderer)
register_renderer("glm5_3_flash", Glm53FlashRenderer)
