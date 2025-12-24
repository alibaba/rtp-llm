import logging
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
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


class QwenReasoningToolRenderer(ReasoningToolBaseRenderer):
    """QwenToolRenderer 使用 Qwen25Detector 进行工具调用解析"""

    def _setup_stop_words(self):
        """设置额外的停止词，子类可以重写"""
        self.add_extra_stop_word_ids([[151643]])  # <|endoftext|>

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        if request.tools:
            detector = Qwen25Detector()
            # 对于qwen3-thinking的模型，注意到tool_call_separator需要设置为"\n\n"
            try:
                rendered_result = self.render_chat(request)
                if rendered_result.rendered_prompt.endswith(self.think_start_tag):
                    detector.tool_call_separator = "\n\n"
            except Exception as e:
                logging.error(f"Failed to render chat in _create_detector: {e}")
            return detector
        else:
            return None

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        """当处于think模式时，默认创建qwen3的ReasoningParser, 但是对于qwen3-thinking的模型，需要创建qwen3-thinking的ReasoningParser, 判断方式为渲染的prompt是否以think_start_tag结尾"""
        if not self.in_think_mode(request):
            return None

        model_type = "qwen3"
        try:
            rendered_result = self.render_chat(request)
            if rendered_result.rendered_prompt.endswith(self.think_start_tag):
                model_type = "qwen3-thinking"
        except Exception as e:
            logging.error(f"Failed to render chat in _create_reasoning_parser: {e}")
        return ReasoningParser(model_type=model_type)


register_renderer("qwen_tool", QwenReasoningToolRenderer)
register_renderer("qwen_3_tool", QwenReasoningToolRenderer)
register_renderer("qwen_3", QwenReasoningToolRenderer)
register_renderer("qwen_3_moe", QwenReasoningToolRenderer)
