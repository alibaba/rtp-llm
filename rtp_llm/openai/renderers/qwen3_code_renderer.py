import json
from typing import Optional

from jinja2 import Environment
from typing_extensions import override

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen3_coder_detector import (
    Qwen3CoderDetector,
)


class Qwen3CoderRenderer(ReasoningToolBaseRenderer):
    """Qwen3CoderRenderer 使用 Qwen3CoderDetector 进行工具调用解析"""

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """创建Qwen3Coder检测器"""
        if request.tools:
            return Qwen3CoderDetector()
        else:
            return None

    def _customize_jinja_env(self, env: Environment) -> None:
        """自定义Jinja2环境，添加smart items过滤器"""
        # 先调用父类方法设置默认的tojson过滤器
        super()._customize_jinja_env(env)

        # 自定义 items 过滤器，智能处理字符串和字典
        def smart_items(value):
            """
            智能 items 过滤器：
            - 如果是字符串，尝试解析为 JSON 后返回 items
            - 如果是字典，直接返回 items
            - 其他情况返回空迭代器
            """
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, dict):
                        return parsed_value.items()
                    else:
                        return []
                except json.JSONDecodeError:
                    return []
            elif isinstance(value, dict):
                return value.items()
            else:
                return []

        env.filters["items"] = smart_items


register_renderer("qwen3_coder_moe", Qwen3CoderRenderer)
register_renderer("qwen35_moe", Qwen3CoderRenderer)
