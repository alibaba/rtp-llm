import json
from typing import Optional

from typing_extensions import override

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
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
        # 模板注入了 think 锚点就意味着模型会输出思考内容，此时即便请求侧
        # thinking_mode 为 DISABLED 也必须建解析器，否则思考块会泄漏进可见回复。
        anchored = self._resolve_think_anchor(request)
        if not anchored and not self.in_think_mode(request):
            return None

        return ReasoningParser(model_type="glm45", force_reasoning=anchored)


register_renderer("glm4_moe", ChatGlm45Renderer)
register_renderer("glm_5", ChatGlm45Renderer)
