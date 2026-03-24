from typing import Optional

from typing_extensions import override

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm47_moe_detector import (
    Glm47MoeDetector,
)


class ChatGlm47Renderer(ChatGlm45Renderer):
    """ChatGLM45Renderer 使用 GLM4MoeDetector 进行工具调用解析"""

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """创建GLM47检测器"""
        if request.tools:
            return Glm47MoeDetector()
        else:
            return None


register_renderer("glm47_moe", ChatGlm47Renderer)
