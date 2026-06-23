import json
import os
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerFast

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    register_tokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer as QwenTokenizerOrigin,
)

# Official Qwen3.5 chat_template.jinja fallback from Qwen/Qwen3.5-35B-A3B.
_QWEN35_DEFAULT_CHAT_TEMPLATE = r"""{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"""


class QWenTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        self.tokenizer = QwenTokenizerOrigin.from_pretrained(tokenizer_path)
        self.tokenizer.decoder.update(
            {v: k for k, v in self.tokenizer.special_tokens.items()}
        )

    @property
    def im_start_id(self):
        return self.tokenizer.im_start_id

    @property
    def im_end_id(self):
        return self.tokenizer.im_end_id

    @property
    def stop_words_id_list(self):
        return [[self.im_end_id], [self.im_start_id]]


class QWenV2Tokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json: Dict[str, Any] = {}):
        try:
            super().init_tokenizer(tokenizer_path, config_json)
        except ValueError as e:
            if "TokenizersBackend" not in str(e):
                raise
            self.tokenizer = self._load_fast_tokenizer_from_tokenizer_json(
                tokenizer_path, config_json
            )
        self.tokenizer.im_start_id = self.tokenizer.encode("<|im_start|>")[0]
        self.tokenizer.im_end_id = self.tokenizer.encode("<|im_end|>")[0]

    def _load_fast_tokenizer_from_tokenizer_json(
        self, tokenizer_path: str, config_json: Dict[str, Any]
    ):
        tokenizer_config = {}
        tokenizer_config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as reader:
                tokenizer_config = json.load(reader)

        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        tokenizer_kwargs = {
            key: tokenizer_config[key]
            for key in [
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
                "additional_special_tokens",
                "model_max_length",
            ]
            if key in tokenizer_config
        }
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file, **tokenizer_kwargs
        )
        chat_template = self._load_chat_template(
            tokenizer_path, tokenizer_config, config_json
        )
        if chat_template:
            tokenizer.chat_template = chat_template
        return tokenizer

    def _load_chat_template(
        self,
        tokenizer_path: str,
        tokenizer_config: Dict[str, Any],
        config_json: Dict[str, Any],
    ):
        if tokenizer_config.get("chat_template"):
            return tokenizer_config["chat_template"]

        chat_template_path = os.path.join(tokenizer_path, "chat_template.jinja")
        if os.path.exists(chat_template_path):
            with open(chat_template_path, "r", encoding="utf-8") as reader:
                return reader.read()

        if self._is_qwen35_config(config_json, tokenizer_path):
            return _QWEN35_DEFAULT_CHAT_TEMPLATE
        return None

    def _is_qwen35_config(self, config_json: Dict[str, Any], tokenizer_path: str):
        text_config = config_json.get("text_config", {})
        model_type = str(
            config_json.get("model_type") or text_config.get("model_type") or ""
        )
        if model_type.startswith("qwen3_5"):
            return True
        lower_path = tokenizer_path.lower()
        return any(name in lower_path for name in ["qwen35", "qwen3.5", "qwen3.6"])

    @property
    def im_start_id(self):
        return self.tokenizer.im_start_id

    @property
    def im_end_id(self):
        return self.tokenizer.im_end_id

    @property
    def stop_words_id_list(self):
        return [[self.im_end_id], [self.im_start_id]]


register_tokenizer(["qwen", "qwen_7b", "qwen_13b", "qwen_1b8"], QWenTokenizer)
register_tokenizer(
    [
        "qwen_2",
        "qwen_agent",
        "qwen_2_embedding",
        "qwen_tool",
        "qwen_2-mtp",
        "qwen_3",
        "qwen_3_tool",
        "qwen35_dense",
        "qwen35_moe",
        "qwen35_moe_mtp",
    ],
    QWenV2Tokenizer,
)
