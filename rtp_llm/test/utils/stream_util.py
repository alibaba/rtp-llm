from typing import Any, Dict, List

from rtp_llm.openai.api_datatype import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    FunctionCall,
    ToolCall,
)


class StreamResponseMerger:
    """专门用于合并 ChatCompletionStreamResponse 的工具类"""

    def __init__(self):
        self.choice_states = {}  # key: choice_index, value: choice合并状态

    def merge_stream_responses(
        self, responses: List[ChatCompletionStreamResponse]
    ) -> ChatCompletionStreamResponse:
        """
        合并多个流式响应为一个完整的响应
        Args:
            responses: 流式响应列表
        Returns:
            合并后的 ChatCompletionStreamResponse
        """
        if not responses:
            raise ValueError("响应列表不能为空")
        # 重置状态
        self.choice_states = {}
        # 使用第一个响应作为基础模板
        base_response = responses[0]
        # 逐个处理响应
        for response in responses:
            self._merge_single_response(response)
        # 构建合并后的choices
        merged_choices = []
        for choice_index in sorted(self.choice_states.keys()):
            merged_choice = self._build_merged_choice(choice_index)
            merged_choices.append(merged_choice)
        # 合并 extra_outputs
        # TODO(zhangjianning.zjn) implement proper merge procedure for extra)_outputs
        extra_outputs = None
        for response in responses:
            if response.extra_outputs is not None:
                extra_outputs = response.extra_outputs
                break
        # 构建最终的响应（使用基础响应的元数据）
        return ChatCompletionStreamResponse(
            id=base_response.id,
            object=base_response.object,
            created=base_response.created,
            model=base_response.model,
            choices=merged_choices,
            usage=base_response.usage,  # 可以用最后一个响应的usage，但这里简化处理
            debug_info=base_response.debug_info,
            aux_info=base_response.aux_info,
            extra_outputs=extra_outputs,
        )

    def _merge_single_response(self, response: ChatCompletionStreamResponse):
        """合并单个流式响应"""
        for choice in response.choices:
            choice_index = choice.index
            # 初始化choice状态
            if choice_index not in self.choice_states:
                self._initialize_choice_state(choice_index)
            # 合并choice内容
            self._merge_choice_content(choice_index, choice)

    def _initialize_choice_state(self, choice_index: int):
        """初始化choice状态"""
        self.choice_states[choice_index] = {
            "content": "",
            "reasoning_content": "",
            "role": None,
            "function_call": None,
            "tool_calls": {},  # key: tool_call_index, value: ToolCall
            "finish_reason": None,
            "logprobs": None,
        }

    def _merge_choice_content(
        self, choice_index: int, choice: ChatCompletionResponseStreamChoice
    ):
        """合并choice内容"""
        choice_state = self.choice_states[choice_index]
        # 合并delta内容
        if choice.delta:
            self._merge_delta_content(choice_state, choice.delta)
        # 更新finish_reason（通常在最后一个chunk中出现）
        if choice.finish_reason:
            choice_state["finish_reason"] = choice.finish_reason
        # 更新logprobs
        if choice.logprobs:
            choice_state["logprobs"] = choice.logprobs

    def _merge_delta_content(self, choice_state: Dict[str, Any], delta: DeltaMessage):
        """合并delta消息内容"""
        # 合并role（通常只在第一个delta中出现）
        if delta.role and choice_state["role"] is None:
            choice_state["role"] = delta.role
        # 合并content
        if delta.content:
            choice_state["content"] += delta.content
        # 合并reasoning_content
        if delta.reasoning_content:
            choice_state["reasoning_content"] += delta.reasoning_content
        # 合并function_call
        if delta.function_call:
            self._merge_function_call(choice_state, delta.function_call)
        # 合并tool_calls
        if delta.tool_calls:
            self._merge_tool_calls(choice_state, delta.tool_calls)

    def _merge_function_call(
        self, choice_state: Dict[str, Any], function_call: FunctionCall
    ):
        """合并function_call"""
        if choice_state["function_call"] is None:
            choice_state["function_call"] = FunctionCall(
                name=function_call.name or "", arguments=function_call.arguments or ""
            )
        else:
            # 合并name（通常只在第一个chunk中出现）
            if function_call.name and not choice_state["function_call"].name:
                choice_state["function_call"].name = function_call.name
            # 累积arguments
            if function_call.arguments:
                if choice_state["function_call"].arguments is None:
                    choice_state["function_call"].arguments = ""
                choice_state["function_call"].arguments += function_call.arguments

    def _merge_tool_calls(
        self, choice_state: Dict[str, Any], tool_calls: List[ToolCall]
    ):
        """合并tool_calls"""
        for tool_call in tool_calls:
            index = tool_call.index
            if index not in choice_state["tool_calls"]:
                # 创建新的tool_call
                choice_state["tool_calls"][index] = ToolCall(
                    index=tool_call.index,
                    id=tool_call.id or "",
                    type=tool_call.type,
                    function=FunctionCall(
                        name=tool_call.function.name or "",
                        arguments=tool_call.function.arguments or "",
                    ),
                )
            else:
                # 合并现有的tool_call
                existing = choice_state["tool_calls"][index]
                # 更新id（只在第一次设置时更新）
                if tool_call.id and not existing.id:
                    existing.id = tool_call.id
                # 更新type
                if tool_call.type:
                    existing.type = tool_call.type
                # 合并function
                if tool_call.function:
                    if tool_call.function.name and not existing.function.name:
                        existing.function.name = tool_call.function.name
                    if tool_call.function.arguments:
                        if existing.function.arguments is None:
                            existing.function.arguments = ""
                        existing.function.arguments += tool_call.function.arguments

    def _build_merged_choice(
        self, choice_index: int
    ) -> ChatCompletionResponseStreamChoice:
        """构建合并后的ChatCompletionResponseStreamChoice"""
        choice_state = self.choice_states[choice_index]
        # 构建合并后的delta
        merged_delta = DeltaMessage(
            role=choice_state["role"],
            content=choice_state["content"] if choice_state["content"] else None,
            reasoning_content=(
                choice_state["reasoning_content"]
                if choice_state["reasoning_content"]
                else None
            ),
            function_call=choice_state["function_call"],
            tool_calls=(
                list(choice_state["tool_calls"].values())
                if choice_state["tool_calls"]
                else None
            ),
        )
        # 构建合并后的choice
        return ChatCompletionResponseStreamChoice(
            index=choice_index,
            delta=merged_delta,
            finish_reason=choice_state["finish_reason"],
            logprobs=choice_state["logprobs"],
        )


def merge_stream_responses(
    responses: List[ChatCompletionStreamResponse],
) -> ChatCompletionStreamResponse:
    """
    便捷函数：合并流式响应
    Args:
        responses: 流式响应列表
    Returns:
        合并后的ChatCompletionStreamResponse
    """
    merger = StreamResponseMerger()
    return merger.merge_stream_responses(responses)


def is_valid_tool_call_chunk(chunk: ChatCompletionStreamResponse) -> bool:
    """
    检查单个chunk中的tool_call结构是否有效
    检查规则：
    1. ToolCall.index 不能为 None
    2. ToolCall.type 不能为空
    3. ToolCall.function 不能为 None
    4. 有name时id不能为空
    5. 有name时不能有arguments
    Args:
        chunk: 单个流式响应
    Returns:
        bool: 是否有效
    """
    if not chunk or not chunk.choices:
        return True
    for choice in chunk.choices:
        if not choice.delta or not choice.delta.tool_calls:
            continue
        for tool_call in choice.delta.tool_calls:
            # 检查基本字段
            if tool_call.index is None:
                return False
            if not tool_call.type:
                return False
            if not tool_call.function:
                return False
            # 检查name和id、arguments的关系
            has_name = tool_call.function.name and tool_call.function.name.strip()
            has_id = tool_call.id and tool_call.id.strip()
            has_arguments = (
                tool_call.function.arguments and tool_call.function.arguments.strip()
            )
            # 规则：有name时id不能为空
            if has_name and not has_id:
                return False
            # 规则：有name时不能有arguments
            if has_name and has_arguments:
                return False
    return True
