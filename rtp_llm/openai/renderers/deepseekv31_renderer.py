import logging
from typing import Optional

from jinja2 import BaseLoader, Environment
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from rtp_llm.openai.api_datatype import ChatCompletionRequest, RoleEnum
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv31_detector import (
    DeepSeekV31Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser

TOOL_CHAT_TEMPLATE_DEEPSEEKV31 = """{% if not add_generation_prompt is defined %}
  {% set add_generation_prompt = false %}
{% endif %}
{% if not thinking is defined %}
  {% set thinking = false %}
{% endif %}
{% set ns = namespace(is_first=false, is_tool=false, system_prompt='', is_first_sp=true, is_last_user=false) %}
{%- for message in messages %}
  {%- if message['role'] == 'system' %}
    {%- if ns.is_first_sp %}
      {% set ns.system_prompt = ns.system_prompt + message['content'] %}
      {% set ns.is_first_sp = false %}
    {%- else %}
      {% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}
    {%- endif %}
  {%- endif %}
{%- endfor %}

{% if tools is defined and tools is not none %}
  {% set tool_ns = namespace(text='## Tools\nYou have access to the following tools:\n') %}
  {% for tool in tools %}
    {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + '\nDescription: ' + tool.function.description + '\n\nParameters: ' + (tool.function.parameters | tojson) + '\n' %}
  {% endfor %}
  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\nWhere:\n\n- `tool_call_name` must be an exact match to one of the available tools\n- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema\n- For multiple tool calls, chain them directly without separators or spaces\n" %}
  {% set ns.system_prompt = ns.system_prompt + '\n\n' + tool_ns.text %}
{% endif %}

{{ bos_token }}{{ ns.system_prompt }}
{%- for message in messages %}
  {%- if message['role'] == 'user' %}
    {%- set ns.is_tool = false -%}
    {%- set ns.is_first = false -%}
    {%- set ns.is_last_user = true -%}
    {{'<｜User｜>' + message['content']}}
  {%- endif %}
  {%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}
    {%- if ns.is_last_user %}
      {{'<｜Assistant｜></think>'}}
    {%- endif %}
    {%- set ns.is_last_user = false -%}
    {%- set ns.is_first = false %}
    {%- set ns.is_tool = false -%}
    {%- for tool in message['tool_calls'] %}
      {%- if not ns.is_first %}
        {%- if message['content'] is none %}
          {{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'}}
        {%- else %}
          {{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'}}
        {%- endif %}
        {%- set ns.is_first = true -%}
      {%- else %}
        {{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'}}
      {%- endif %}
    {%- endfor %}
    {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
  {%- endif %}
  {%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none) %}
    {%- if ns.is_last_user %}
      {{'<｜Assistant｜>'}}
      {%- if message['prefix'] is defined and message['prefix'] and thinking %}
        {{'<think>'}}
      {%- else %}
        {{'</think>'}}
      {%- endif %}
    {%- endif %}
    {%- set ns.is_last_user = false -%}
    {%- if ns.is_tool %}
      {{message['content'] + '<｜end▁of▁sentence｜>'}}
      {%- set ns.is_tool = false -%}
    {%- else %}
      {%- set content = message['content'] -%}
      {%- if '</think>' in content %}
        {%- set content = content.split('</think>', 1)[1] -%}
      {%- endif %}
      {{content + '<｜end▁of▁sentence｜>'}}
    {%- endif %}
  {%- endif %}
  {%- if message['role'] == 'tool' %}
    {%- set ns.is_last_user = false -%}
    {%- set ns.is_tool = true -%}
    {{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
  {%- endif %}
{%- endfor -%}
{%- if add_generation_prompt and ns.is_last_user and not ns.is_tool %}
  {{'<｜Assistant｜>'}}
  {%- if not thinking %}
    {{'</think>'}}
  {%- else %}
    {{'<think>'}}
  {%- endif %}
{% endif %}"""


class DeepseekV31Renderer(ReasoningToolBaseRenderer):
    """DeepSeekV31渲染器
    这里需要注意的点包括:
    1. DeepseekV31默认在工具调用情况下, 不支持深度思考, 所以渲染prompt时, 有tools就不开启thinking
    2. 兼容之前的enable_thinking的行为, 让用户指定enable_thinking时, thinking也能生效
    3. ReasoningParser只有在渲染的prompt以<think>结尾时 且在think_mode下, 才会创建reasoning解析
    """

    @override
    def _setup_chat_template(self):
        self.chat_template = TOOL_CHAT_TEMPLATE_DEEPSEEKV31

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

        # 合并chat_template_kwargs
        if request.chat_template_kwargs is not None:
            context.update(request.chat_template_kwargs)

        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
        ):
            context.update(request.extra_configs.chat_template_kwargs)

        # 兼容一下enable_thinking的行为, 让用户指定enable_thinking时, thinking也能生效
        if context.get("enable_thinking") == True:
            context["thinking"] = context["enable_thinking"]

        if self.tokenizer.bos_token:
            context["bos_token"] = self.tokenizer.bos_token

        # 带有tools的情况默认不开启thinking
        if request.tools:
            context["thinking"] = False

        # 创建Jinja2环境
        env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

        # 允许子类自定义环境
        self._customize_jinja_env(env)

        try:
            template = env.from_string(self.chat_template)
            rendered_prompt = template.render(**context)
            return rendered_prompt
        except Exception as e:
            logging.error(f"构建提示文本失败: {str(e)}")
            raise ValueError(f"Error rendering prompt template: {str(e)}")

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        if request.tools:
            return DeepSeekV31Detector()
        else:
            return None

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        if not self.in_think_mode(request):
            return None

        try:
            rendered_result = self.render_chat(request)
            if rendered_result.rendered_prompt.endswith("<think>"):
                return ReasoningParser(model_type="deepseek-v3", force_reasoning=True)
        except Exception as e:
            logging.error(f"Failed to render chat in _create_reasoning_parser: {e}")
            return None

        return None


register_renderer("deepseek_v31", DeepseekV31Renderer)
