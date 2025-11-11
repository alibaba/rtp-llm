import logging
import re
from typing import List, Optional, Set

from typing_extensions import override

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    GPTToolDefinition,
    ToolCall,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.format_convert_helper import (
    rtp_tools_to_sglang_tools,
    streaming_parse_result_to_tool_calls,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.kimik2_detector import (
    KimiK2Detector,
)


class KimiK2Renderer(ReasoningToolBaseRenderer):
    """KimiK2渲染器，使用 KimiK2Detector 进行工具调用解析"""

    @override
    def _setup_stop_words(self):
        """设置KimiK2特定的停止词"""
        self.add_extra_stop_words(["<|im_end|>"])

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        if request.tools:
            return KimiK2Detector()
        else:
            return None

    @override
    def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
        """预处理消息，确保 tool_call_id 和 tool_calls 中的 id 符合格式，并验证对应关系"""
        # ID 格式验证正则表达式：functions.{name}:{index}
        id_pattern = re.compile(r"^functions\.[\w\.-]+:\d+$")

        processed_messages = []
        tool_call_ids_used: Set[str] = set()  # 收集所有使用的 tool_call_id
        tool_call_ids_returned: Set[str] = set()  # 收集所有返回的 tool_call_id

        for message in messages:
            processed_message = message.copy()

            # 处理 tool_calls 中的 id
            if "tool_calls" in processed_message and processed_message["tool_calls"]:
                processed_tool_calls = []
                for tool_call in processed_message["tool_calls"]:
                    processed_tool_call = tool_call.copy()

                    # 确保 id 符合 functions.{name}:{index} 格式
                    if "id" in processed_tool_call:
                        tool_call_id = processed_tool_call["id"]
                        if not tool_call_id.startswith("functions."):
                            tool_call_id = f"functions.{tool_call_id}"
                            processed_tool_call["id"] = tool_call_id

                        # 验证 ID 格式
                        if not id_pattern.match(tool_call_id):
                            raise ValueError(
                                f"Invalid tool_call id format: '{tool_call_id}'. Expected format: 'functions.{{name}}:{{index}}'"
                            )

                        # 收集使用的 tool_call_id
                        tool_call_ids_used.add(tool_call_id)

                    processed_tool_calls.append(processed_tool_call)
                processed_message["tool_calls"] = processed_tool_calls

            # 处理 tool_call_id
            if "tool_call_id" in processed_message:
                tool_call_id = processed_message["tool_call_id"]
                if not tool_call_id.startswith("functions."):
                    tool_call_id = f"functions.{tool_call_id}"
                    processed_message["tool_call_id"] = tool_call_id

                # 验证 ID 格式
                if not id_pattern.match(tool_call_id):
                    raise ValueError(
                        f"Invalid tool_call_id format: '{tool_call_id}'. Expected format: 'functions.{{name}}:{{index}}'"
                    )

                # 收集返回的 tool_call_id
                tool_call_ids_returned.add(tool_call_id)

            processed_messages.append(processed_message)

        # 验证每个使用的 tool_call_id 都有对应的返回
        missing_returns = tool_call_ids_used - tool_call_ids_returned
        if missing_returns:
            raise ValueError(
                f"Missing tool_call_id returns for: {', '.join(missing_returns)}"
            )

        # 可选：验证没有多余的返回（如果需要严格对应）
        extra_returns = tool_call_ids_returned - tool_call_ids_used
        if extra_returns:
            raise ValueError(
                f"Unexpected tool_call_id returns: {', '.join(extra_returns)}"
            )

        return processed_messages

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
            logging.debug("清理停止词: '%s' -> '%s'", original_text, text)

        return text

    @override
    async def _extract_tool_calls_content(
        self,
        detector: Optional[BaseFormatDetector],
        tools: Optional[List[GPTToolDefinition]],
        text: str,
        is_streaming: bool,
    ) -> tuple[Optional[List[ToolCall]], str]:
        """
        支持kimi_k2的tool_call_id类似于functions.get_current_weather:1这样的返回结果
        在拿到streaming_parse_result_to_tool_calls返回值后, 需要进行部分定制重写
        """

        # 如果没有工具定义，直接返回
        if not detector:
            return None, text

        # 转换工具格式
        sglang_tools = rtp_tools_to_sglang_tools(tools)

        try:
            if is_streaming:
                parse_result = detector.parse_streaming_increment(text, sglang_tools)
            else:
                # 兼容kimik2在非流式的情况下可能返回结果中有以<|im_end|>的结果
                cleaned_text = self._clean_stop_words(text)
                parse_result = detector.detect_and_parse(cleaned_text, sglang_tools)

            # 有工具调用时，使用格式转换函数
            tool_calls, remaining_text = streaming_parse_result_to_tool_calls(
                parse_result
            )

            # 需要在这里单独的调整tool_call.id
            if is_streaming:
                for tool_call in tool_calls:
                    if tool_call.function.name is not None:
                        tool_call.id = (
                            f"{tool_call.function.name}:{detector.function_idx}"
                        )
            else:
                for tool_call in tool_calls:
                    tool_call.id = f"{tool_call.function.name}:{tool_call.index}"

            # 统一为非流式场景的tool_call重新分配index以兼容sglang中非流式大部分index为-1的情况
            if not is_streaming:
                for i, tool_call in enumerate(tool_calls):
                    tool_call.index = i

            return tool_calls, remaining_text

        except Exception as e:
            logging.error(f"工具调用解析失败: {e}, 当前文本: {text}")
            return None, text


register_renderer("kimi_k2", KimiK2Renderer)
