import logging

from typing_extensions import override

from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.kimik2_detector import (
    KimiK2Detector,
)
from rtp_llm.openai.renderers.tool_base_renderer import ToolBaseRenderer


class KimiK2Renderer(ToolBaseRenderer):
    """KimiK2渲染器，使用 KimiK2Detector 进行工具调用解析"""

    @override
    def _setup_stop_words(self):
        """设置KimiK2特定的停止词"""
        self.add_extra_stop_words(["<|im_end|>"])

    def _create_detector(self) -> BaseFormatDetector:
        """创建KimiK2检测器"""
        return KimiK2Detector()

    @override
    def _clean_stop_words(self, text: str) -> str:
        """清理KimiK2特定的停止词"""
        if not text:
            return text

        stop_words = ["<|im_end|>"]
        original_text = text

        for stop_word in stop_words:
            text = text.replace(stop_word, "")

        text = text.strip()

        if original_text != text:
            logging.debug(f"清理停止词: '{original_text}' -> '{text}'")

        return text


register_renderer("kimi_k2", KimiK2Renderer)
