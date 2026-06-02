import uuid
from typing import List

from rtp_llm.openai.api_datatype import FunctionCall, GPTToolDefinition, ToolCall
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
)


def streaming_parse_result_to_tool_calls(
    result: StreamingParseResult,
) -> tuple[List[ToolCall], str]:
    """
    将 StreamingParseResult 转换为 ToolCall 列表

    Args:
        result: StreamingParseResult 对象，包含普通文本和工具调用信息

    Returns:
        tuple[List[ToolCall], str]: (工具调用列表, 剩余的普通文本)
    """
    # Many detectors may emit multiple ToolCallItem entries for the same tool_index in
    # a single chunk (e.g. "{" then "\"location\": " then "\"杭州"). For downstream
    # consumers it's much easier (and matches common production assumptions) to have
    # at most one ToolCall per index per chunk. We squash by tool_index and sort the
    # output by tool_index, since tool_index represents the model's intended ordering.
    tool_call_map: dict[int, ToolCall] = {}

    for call in result.calls:
        tool_index = call.tool_index
        tool_call = tool_call_map.get(tool_index)
        if tool_call is None:
            call_id = f"call_{uuid.uuid4().hex[:24]}" if call.name else None
            tool_call = ToolCall(
                index=tool_index,
                id=call_id,
                type="function",
                function=FunctionCall(name=call.name, arguments=""),
            )
            tool_call_map[tool_index] = tool_call
        else:
            if call.name and not tool_call.function.name:
                tool_call.function.name = call.name
                if not tool_call.id:
                    tool_call.id = f"call_{uuid.uuid4().hex[:24]}"

        if call.parameters:
            tool_call.function.arguments = (
                tool_call.function.arguments or ""
            ) + call.parameters

    return [tool_call_map[i] for i in sorted(tool_call_map)], result.normal_text


def rtp_tools_to_sglang_tools(rtp_tools: List[GPTToolDefinition]) -> List[Tool]:
    """
    将 RTP 格式的工具定义转换为 SGLang 格式

    Args:
        rtp_tools: RTP 格式的工具定义列表

    Returns:
        List[Tool]: SGLang 格式的工具定义列表
    """
    sglang_tools: List[Tool] = []
    for rtp_tool in rtp_tools:
        if rtp_tool.type == "function" and rtp_tool.function:
            sglang_tool = Tool(
                type="function",
                function=Function(
                    name=rtp_tool.function.name,
                    description=rtp_tool.function.description,
                    parameters=rtp_tool.function.parameters,
                ),
            )
            sglang_tools.append(sglang_tool)
    return sglang_tools
