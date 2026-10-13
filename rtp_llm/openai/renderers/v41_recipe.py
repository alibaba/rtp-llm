"""Python request conversion and rendering from deepseek-recipe 8cadfede.

The upstream MIT notice is in v41_recipe.LICENSE. No native recipe code is used.
"""

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from rtp_llm.openai.reasoning_effort import normalize_v41_reasoning_effort

RECIPE_REVISION = "8cadfede7063c896b944e7bae05daa3549ae97ea"
BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"
SYSTEM = "<\uff5cSystem\uff5c>"
USER = "<\uff5cUser\uff5c>"
ASSISTANT = "<\uff5cAssistant\uff5c>"
IMAGE = "<\uff5cdeepseek_image\uff5c>"
DSML = "\uff5cDSML\uff5c"
CALLS = f"<{DSML} calls>"

_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml} calls>" block like the following:

<{dsml} calls>
<{dsml} invoke name="$TOOL_NAME">
<{dsml} parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml} parameter>
...
</{dsml} invoke>
<{dsml} invoke name="$TOOL_NAME2">
...
</{dsml} invoke>
</{dsml} calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


@dataclass(frozen=True)
class RecipeRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    thinking: bool
    effort: int
    force_tools: bool
    json_output: bool
    stop: tuple[str, ...]


def resolve_thinking(request: dict, default: bool) -> tuple[bool, int]:
    kwargs = request.get("chat_template_kwargs") or {}
    extra = request.get("extra_configs") or {}
    kwargs = {**kwargs, **(extra.get("chat_template_kwargs") or {})}
    effort = request.get("reasoning_effort")
    if effort is None:
        effort = kwargs.get("reasoning_effort")
    budget = normalize_v41_reasoning_effort(effort)
    explicit = request.get("thinking")
    if explicit is not None:
        if explicit.get("type") not in ("enabled", "disabled"):
            raise ValueError("thinking.type must be enabled or disabled")
        thinking = explicit["type"] == "enabled"
    elif request.get("enable_thinking") is not None:
        thinking = request["enable_thinking"]
    elif kwargs.get("enable_thinking") is not None:
        thinking = kwargs["enable_thinking"]
    elif "thinking_mode" in kwargs:
        if kwargs["thinking_mode"] not in ("thinking", "chat"):
            raise ValueError("thinking_mode must be thinking or chat")
        thinking = kwargs["thinking_mode"] == "thinking"
    elif effort == "none":
        thinking = False
    elif effort is not None:
        thinking = True
    else:
        thinking = default
    # The existing explicit zero budget remains a hard request-level limit.
    if request.get("thinking_budget") == 0 or extra.get("max_thinking_tokens") == 0:
        thinking = False
    return bool(thinking), budget


def _validate_text(text: str) -> str:
    if IMAGE in text:
        raise ValueError("literal V4.1 image placeholders are not valid API text")
    return text


def _json_dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _reject_json_constant(value):
    raise ValueError(f"{value} is not a finite JSON number")


def _content(value, role: str) -> tuple[str, list[dict]]:
    if value is None:
        return "", []
    if isinstance(value, str):
        return _validate_text(value), []
    parts, images = [], []
    for part in value:
        if any(v is not None for v in (part.get("preprocess_config") or {}).values()):
            raise ValueError("V4.1 image preprocessing is fixed by the model config")
        if part.get("type") == "text":
            parts.append(_validate_text(part.get("text") or ""))
        elif part.get("type") == "image_url":
            if role not in ("user", "tool"):
                raise ValueError(f"images are not supported in {role} messages")
            source = part["image_url"]
            if not source.get("url"):
                raise ValueError("image_url.url must be non-empty")
            images.append({"type": "image", **source})
            parts.append(IMAGE)
        else:
            raise ValueError("V4.1 supports text and image_url content parts only")
    return "\n\n".join(parts), images


def _messages(raw_messages: list[dict]) -> list[dict]:
    if not raw_messages:
        raise ValueError("Empty input messages")
    messages = []
    index = 0
    while index < len(raw_messages):
        raw = raw_messages[index]
        role = raw["role"]
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unsupported V4.1 message role: {role}")
        if role == "tool":
            raise ValueError("tool messages must follow their assistant tool_calls")
        content, images = _content(raw.get("content"), role)
        message = {"role": role, "content": content, "images": images}
        if role == "system" and not content:
            index += 1
            continue
        calls = raw.get("tool_calls")
        if role == "assistant":
            if raw.get("content") is None and calls is None:
                raise ValueError("assistant content or tool_calls must be set")
            message["reasoning_content"] = _validate_text(
                raw.get("reasoning_content") or ""
            )
        messages.append(message)
        index += 1
        if calls is None:
            continue
        if role != "assistant" or not calls:
            raise ValueError("tool_calls must be a non-empty assistant array")
        ids = [call.get("id") for call in calls]
        if any(not identity for identity in ids) or len(set(ids)) != len(ids):
            raise ValueError("historical tool calls require unique non-empty IDs")
        message["tool_calls"] = copy.deepcopy(calls)
        for call in message["tool_calls"]:
            function = call["function"]
            _validate_text(function["name"])
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = _json_dumps(arguments)
            function["arguments"] = _validate_text(arguments)
        results = {}
        for _ in calls:
            if index >= len(raw_messages) or raw_messages[index]["role"] != "tool":
                raise ValueError(
                    "every tool_call must have an immediately following tool result"
                )
            raw_result = raw_messages[index]
            identity = raw_result.get("tool_call_id")
            if identity not in ids or identity in results:
                raise ValueError("unexpected or duplicate tool_call_id")
            text, images = _content(raw_result.get("content"), "tool")
            results[identity] = {"role": "tool", "content": text, "images": images}
            index += 1
        messages.extend(results[identity] for identity in ids)
    return messages


def convert_request(request: dict, *, default_thinking: bool = False) -> RecipeRequest:
    thinking, effort = resolve_thinking(request, default_thinking)
    tools = []
    names = set()
    for raw in request.get("tools") or []:
        function = raw["function"]
        name = function["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise ValueError("tool names require 1..128 ASCII letters, digits, _ or -")
        if name in names:
            raise ValueError("tool names must be unique")
        names.add(name)
        parameters = function.get("parameters") or {}
        if parameters and parameters.get("type") != "object":
            raise ValueError("tool parameters must be a JSON Schema of type object")
        tools.append(
            {
                "name": name,
                "description": function.get("description") or "",
                "parameters": parameters,
                "strict": function.get("strict"),
            }
        )
    _validate_text(_json_dumps(tools))
    choice = request.get("tool_choice")
    forced = choice == "required" or isinstance(choice, dict)
    if choice == "none":
        tools = []
    elif isinstance(choice, dict):
        name = choice["function"]["name"]
        if name not in names:
            raise ValueError("tool_choice function is not in tools")
        tools = [tool for tool in tools if tool["name"] == name]
    if forced and not tools:
        raise ValueError("tool_choice requires at least one tool")
    if forced and thinking:
        raise ValueError("Thinking mode does not support this tool_choice")
    messages = _messages(request["messages"])
    response_format = request.get("response_format") or {}
    json_output = response_format.get("type") == "json_object"
    if json_output and not any("json" in msg["content"].lower() for msg in messages):
        raise ValueError("json_object requires the word json in an input message")
    stop = request.get("stop") or []
    stop = [stop] if isinstance(stop, str) else stop
    if len(stop) > 16:
        raise ValueError("at most 16 stop sequences are supported")
    if any(not isinstance(value, str) for value in stop):
        raise ValueError("stop must be a string or string array")
    return RecipeRequest(
        messages,
        tools,
        thinking,
        effort,
        forced,
        json_output,
        tuple(value for value in stop if value),
    )


def _render_calls(calls: list[dict]) -> str:
    rendered = []
    for call in calls:
        function = call["function"]
        raw = function["arguments"]
        try:
            arguments = json.loads(raw, parse_constant=_reject_json_constant)
        except (ValueError, TypeError):
            arguments = None
        if not isinstance(arguments, dict):
            arguments = {"arguments": raw}
        parameters = []
        for key, value in arguments.items():
            string = isinstance(value, str)
            text = value if string else _json_dumps(value)
            parameters.append(
                f'<{DSML} parameter name="{key}" string="{str(string).lower()}">{text}</{DSML} parameter>'
            )
        values = "\n".join(parameters)
        rendered.append(
            f'<{DSML} invoke name="{function["name"]}">\n{values}\n</{DSML} invoke>'
        )
    return f"{CALLS}\n" + "\n".join(rendered) + f"\n</{DSML} calls>"


def render_request(request: RecipeRequest) -> tuple[str, list[dict]]:
    messages = copy.deepcopy(request.messages)
    if request.tools or request.json_output:
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": "", "images": []})
        if request.tools:
            schemas = "\n".join(
                _json_dumps(
                    {key: tool[key] for key in ("name", "description", "parameters")}
                )
                for tool in request.tools
            )
            messages[0]["content"] += "\n\n" + _TOOLS_TEMPLATE.format(
                dsml=DSML, schemas=schemas
            )
        if request.json_output:
            messages[0][
                "content"
            ] += '\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{"type": "json_object"}'
    result, images, previous = [BOS], [], None
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if index == 0 and (request.thinking or role == "system"):
            result.append(SYSTEM)
        if index == 0 and request.thinking:
            result.append(
                f"Reasoning Effort: {request.effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
            )
        if role == "system":
            result.append((SYSTEM if index else "") + content)
        elif role in ("user", "tool"):
            result.append("\n\n" if previous in ("user", "tool") else USER)
            result.append(
                f"<tool_result>{content}</tool_result>" if role == "tool" else content
            )
        elif role == "assistant":
            result.append(ASSISTANT)
            if request.thinking and index:
                result.extend(
                    ["<think>", message.get("reasoning_content", ""), "</think>"]
                )
            else:
                result.append("</think>")
            result.append(content)
            if message.get("tool_calls"):
                result.extend(["\n\n", _render_calls(message["tool_calls"])])
            result.append(EOS)
        images.extend(message["images"])
        previous = role
    result.extend([ASSISTANT, "<think>" if request.thinking else "</think>"])
    if request.force_tools:
        result.append(f"\n\n{CALLS}\n")
    return "".join(result), images
