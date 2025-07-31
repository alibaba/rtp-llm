import json
import logging

from jinja2 import BaseLoader, Environment
from typing_extensions import override

from rtp_llm.openai.api_datatype import ChatCompletionRequest, RoleEnum
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
)
from rtp_llm.openai.renderers.tool_base_renderer import ToolBaseRenderer


class ChatGlm45Renderer(ToolBaseRenderer):
    """ChatGLM45Renderer 使用 GLM4MoeDetector 进行工具调用解析"""

    def _preprocess_messages(self, messages):
        """预处理消息，确保 tool_calls 中的 arguments 是字典对象"""
        processed_messages = []
        for message in messages:
            processed_message = message.copy()
            if "tool_calls" in processed_message and processed_message["tool_calls"]:
                processed_tool_calls = []
                for tool_call in processed_message["tool_calls"]:
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
            processed_messages.append(processed_message)
        return processed_messages

    def _create_detector(self) -> BaseFormatDetector:
        """创建GLM45检测器"""
        return Glm4MoeDetector()

    @override
    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        """
        构建提示文本
        Args:
            request: 聊天完成请求
        Returns:
            str: 格式化后的提示文本
        """
        context = request.model_dump(exclude_none=True)

        # 只要不是已经有assistant消息, 则需要添加生成提示
        if request.messages[-1].role != RoleEnum.assistant:
            context["add_generation_prompt"] = True

        messages = self._preprocess_messages(context["messages"])
        context.update({"messages": messages})

        # 合并chat_template_kwargs
        if request.chat_template_kwargs is not None:
            context.update(request.chat_template_kwargs)

        if (
            request.extend_fields is not None
            and "chat_template_kwargs" in request.extend_fields
            and isinstance(request.extend_fields["chat_template_kwargs"], dict)
        ):
            context.update(request.extend_fields["chat_template_kwargs"])

        # 创建Jinja2环境
        env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

        # 允许子类自定义环境
        self._customize_jinja_env(env)

        try:
            template = env.from_string(self.tokenizer.chat_template)
            rendered_prompt = template.render(**context)
            return rendered_prompt
        except Exception as e:
            logging.error(f"构建提示文本失败: {str(e)}")
            raise ValueError(f"Error rendering prompt template: {str(e)}")

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


register_renderer("chatglm45", ChatGlm45Renderer)
