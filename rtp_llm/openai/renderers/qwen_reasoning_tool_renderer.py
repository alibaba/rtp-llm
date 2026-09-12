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
        # 旧实现写死 151643（151K 词表上的 <|endoftext|>），词表不同会注册错 token。
        self.add_extra_stop_word_ids(self.encode_extra_stop_words(["<|endoftext|>"]))

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        if request.tools:
            detector = Qwen25Detector()
            # 对于qwen3-thinking的模型，注意到tool_call_separator需要设置为"\n\n"
            if self._resolve_think_anchor(request):
                detector.tool_call_separator = "\n\n"
            return detector
        else:
            return None

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        """默认创建 qwen3 的 ReasoningParser；若渲染的 prompt 以 think_start_tag 结尾，
        说明模板已注入思考锚点，改创建 qwen3-thinking 的 ReasoningParser。"""
        # 锚点存在即意味着模型会输出思考内容，此时即便请求侧 thinking_mode 为
        # DISABLED 也必须建解析器，否则思考块会泄漏进可见回复。
        anchored = self._resolve_think_anchor(request)
        if not anchored and not self.in_think_mode(request):
            return None

        return ReasoningParser(model_type="qwen3-thinking" if anchored else "qwen3")


register_renderer("qwen_tool", QwenReasoningToolRenderer)
register_renderer("qwen_3_tool", QwenReasoningToolRenderer)
register_renderer("qwen_3", QwenReasoningToolRenderer)
register_renderer("qwen_3_moe", QwenReasoningToolRenderer)
register_renderer("qwen3_next", QwenReasoningToolRenderer)
