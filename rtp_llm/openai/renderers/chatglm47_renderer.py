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
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        # GLM-5.3's checkpoint template always opens a <think> channel for the
        # assistant turn, so its reasoning must be parsed even when the global
        # THINK_MODE switch was not set for the service.
        if self.model_type == "glm5_3_flash":
            return True
        return super().in_think_mode(request)

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
register_renderer("glm5_3_flash", ChatGlm47Renderer)
