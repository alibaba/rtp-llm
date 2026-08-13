import json
import logging
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
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


class ChatGlm45Renderer(ReasoningToolBaseRenderer):
    """ChatGLM45Renderer 使用 GLM4MoeDetector 进行工具调用解析"""

    @override
    def _setup_stop_words(self):
        """设置GLM45特定的停止词"""
        self.add_extra_stop_words(["<|user|>", "<|observation|>"])

    @override
    def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize tool arguments and associate tool results with their calls."""
        processed_messages = []
        message_index = 0
        while message_index < len(messages):
            message = messages[message_index]
            processed_message = message.copy()
            tool_calls = processed_message.get("tool_calls") or []
            if tool_calls:
                processed_tool_calls = []
                for tool_call in tool_calls:
                    processed_tool_call = tool_call.copy()
                    if "function" in processed_tool_call:
                        function = processed_tool_call["function"].copy()
                        if "arguments" in function and isinstance(
                            function["arguments"], str
                        ):
                            try:
                                function["arguments"] = json.loads(
                                    function["arguments"]
                                )
                            except json.JSONDecodeError:
                                function["arguments"] = {}
                        processed_tool_call["function"] = function
                    elif "arguments" in processed_tool_call and isinstance(
                        processed_tool_call["arguments"], str
                    ):
                        try:
                            processed_tool_call["arguments"] = json.loads(
                                processed_tool_call["arguments"]
                            )
                        except json.JSONDecodeError:
                            processed_tool_call["arguments"] = {}
                    processed_tool_calls.append(processed_tool_call)
                processed_message["tool_calls"] = processed_tool_calls

            if processed_message.get("role") == "tool":
                raise ValueError(
                    "Tool result must immediately follow an assistant tool call"
                )

            processed_messages.append(processed_message)
            message_index += 1

            if not tool_calls:
                continue
            if processed_message.get("role") != "assistant":
                raise ValueError("Only assistant messages may contain tool calls")

            tool_results = []
            while (
                message_index < len(messages)
                and messages[message_index].get("role") == "tool"
            ):
                tool_results.append(messages[message_index].copy())
                message_index += 1
            processed_messages.extend(
                self._order_tool_results(processed_tool_calls, tool_results)
            )

        return processed_messages

    @staticmethod
    def _order_tool_results(
        tool_calls: list[dict], tool_results: list[dict]
    ) -> list[dict]:
        if len(tool_calls) != len(tool_results):
            raise ValueError(
                "Every assistant tool call must have exactly one adjacent tool result"
            )

        call_ids = [
            ChatGlm45Renderer._optional_tool_id(tool_call, "id")
            for tool_call in tool_calls
        ]
        result_ids = [
            ChatGlm45Renderer._optional_tool_id(tool_result, "tool_call_id")
            for tool_result in tool_results
        ]

        all_ids = call_ids + result_ids
        if len(tool_calls) == 1 and all(tool_id is None for tool_id in all_ids):
            return tool_results
        if any(tool_id is None for tool_id in all_ids):
            raise ValueError("Tool history must provide every tool call id")

        if len(set(call_ids)) != len(call_ids):
            raise ValueError("Assistant tool call ids must be unique")
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("Tool result ids must be unique")

        results_by_id = dict(zip(result_ids, tool_results))
        if set(call_ids) != set(result_ids):
            raise ValueError("Tool results do not match the assistant tool calls")
        return [results_by_id[tool_id] for tool_id in call_ids]

    @staticmethod
    def _optional_tool_id(item: dict, field: str) -> Optional[str]:
        tool_id = item.get(field)
        if tool_id is None:
            return None
        if not isinstance(tool_id, str) or not tool_id:
            raise ValueError(f"{field} must be a non-empty string")
        return tool_id

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """创建GLM45检测器"""
        if request.tools:
            return Glm4MoeDetector()
        else:
            return None

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        if not self.in_think_mode(request):
            return None

        force_reasoning = request._force_reasoning_from_rendered_prompt
        if force_reasoning is None:
            try:
                self.render_chat(request)
                force_reasoning = request._force_reasoning_from_rendered_prompt
            except Exception as e:
                logging.error(f"Failed to render chat in _create_reasoning_parser: {e}")
                request._force_reasoning_from_rendered_prompt = False

        return ReasoningParser(
            model_type="glm45", force_reasoning=force_reasoning is True
        )

    @override
    def _update_request_from_rendered_prompt(
        self, request: ChatCompletionRequest, rendered_prompt: str
    ) -> None:
        request._force_reasoning_from_rendered_prompt = rendered_prompt.endswith(
            "<think>"
        )

    @override
    def _customize_jinja_env(self, env: Environment) -> None:
        """
        自定义Jinja2环境，子类可以重写此方法来添加自定义过滤器、函数等

        Args:
            env: Jinja2环境对象
            request: 聊天完成请求
            context: 模板渲染上下文
        """
        # 设置默认的tojson过滤器
        env.filters["tojson"] = lambda value, **kwargs: (
            value
            if isinstance(value, str)
            else json.dumps(
                value, sort_keys=False, ensure_ascii=kwargs.get("ensure_ascii", False)
            )
        )


register_renderer("glm4_moe", ChatGlm45Renderer)
