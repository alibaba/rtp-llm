from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)
from rtp_llm.openai.renderers.tool_base_renderer import ToolBaseRenderer


class QwenToolRenderer(ToolBaseRenderer):
    """QwenToolRenderer 使用 Qwen25Detector 进行工具调用解析"""

    def _create_detector(self) -> BaseFormatDetector:
        return Qwen25Detector()

    def in_think_mode(self, request: ChatCompletionRequest):
        if request.disable_thinking():
            return False
        return super().in_think_mode(request)


register_renderer("qwen_tool", QwenToolRenderer)
register_renderer("qwen_3_tool", QwenToolRenderer)
