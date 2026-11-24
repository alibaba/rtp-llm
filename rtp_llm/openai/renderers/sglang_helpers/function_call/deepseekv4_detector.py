import json
import logging
import os
import re
from typing import Any, Dict, List

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import Tool
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


def _longest_suffix_prefix_len(text: str, tag: str) -> int:
    max_k = min(len(text), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def _strip_trailing_tag_prefix(text: str, tag: str) -> str:
    k = _longest_suffix_prefix_len(text, tag)
    return text[:-k] if k else text


def _dsv4_detector_debug_enabled() -> bool:
    return os.environ.get("RTP_LLM_DSV4_RENDERER_DEBUG", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _preview_text(text: str, limit: int = 512) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}...[{len(text)} chars]...{text[-half:]}"


class DeepSeekV4Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V4 model tool call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit tool calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
        </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
        </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    def __init__(self, encoding_module=None, thinking_mode: str = "chat"):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        self.partial_parameter_regex = (
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
        self.invoke_regex = (
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)'
        )
        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
        self.current_tool_id = -1
        self.encoding_module = encoding_module
        self.thinking_mode = thinking_mode
        self._in_tool_calls_block = False
        self._param_end_tag = "</｜DSML｜parameter>"
        self._invoke_open_re = re.compile(r'<｜DSML｜invoke\s+name="([^"]+)"\s*>')
        self._param_open_re = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>'
        )
        self._stream_state = "OUTSIDE"
        self._cur_func_name = ""
        self._cur_param_config: Dict[str, Any] = {}
        self._cur_param_name = ""
        self._cur_param_string_attr = "true"
        self._cur_defer_repair = False
        self._cur_invoke_param_count = 0
        self._cur_invoke_opened_json = False
        self._cur_deferred_params: Dict[str, tuple[str, str]] = {}
        self._cur_param_chunks: List[str] = []
        self._cur_scan_pos = 0
        self._emitted_normal_text = False

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses DSML tool calls in the provided text.

        Args:
            text: The complete text to parse
            tools: List of available tools

        Returns:
            StreamingParseResult with parsed tool calls
        """
        if self.bot_token not in text:
            if _dsv4_detector_debug_enabled():
                logger.info(
                    "[DeepSeekV4RendererDebug] detector_no_bot_token "
                    "text_len=%d has_invoke=%s preview=%s",
                    len(text),
                    "<｜DSML｜invoke" in text,
                    _preview_text(text),
                )
            return StreamingParseResult(normal_text=text, calls=[])

        if _dsv4_detector_debug_enabled():
            logger.info(
                "[DeepSeekV4RendererDebug] detector_detect_and_parse "
                "thinking_mode=%s text_len=%d bot_pos=%d eot_pos=%d preview=%s",
                self.thinking_mode,
                len(text),
                text.find(self.bot_token),
                text.find(self.eot_token),
                _preview_text(text),
            )

        if _dsv4_detector_debug_enabled():
            logger.info("[DeepSeekV4RendererDebug] detector_dsml_block_parse_start")
        return self._parse_with_regex(text, tools)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v4 format tool call."""
        return self.bot_token in text or "<｜DSML｜invoke" in text

    def _get_param_config(self, func_name: str, tools: List[Tool]) -> Dict[str, Any]:
        for tool in tools:
            if tool.function.name != func_name:
                continue
            schema = tool.function.parameters
            if isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
                return schema["properties"]
            return {}
        return {}

    def _convert_param_value_checked(self, value: str, param_type: str) -> Any:
        if value.lower() == "null":
            return None

        param_type = param_type.lower()
        if param_type in ("string", "str", "text"):
            return value
        if param_type in ("integer", "int"):
            return int(value)
        if param_type in ("number", "float"):
            parsed = float(value)
            return parsed if parsed != int(parsed) else int(parsed)
        if param_type in ("boolean", "bool"):
            value = value.strip()
            if value.lower() not in ("false", "0", "true", "1"):
                raise ValueError("Invalid boolean value")
            return value.lower() in ("true", "1")
        if param_type in ("object", "array"):
            return json.loads(value)
        return json.loads(value)

    def _convert_param_value(self, value: str, param_type: Any) -> Any:
        if not isinstance(param_type, list):
            param_type = [param_type]
        for current_type in param_type:
            try:
                return self._convert_param_value_checked(value, current_type)
            except Exception:
                continue
        return value

    def _repair_param_dict(
        self, parameters: Dict[str, Any], param_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        allowed = set(param_config.keys())
        for wrapper in ("arguments", "input"):
            if set(parameters.keys()) != {wrapper} or wrapper in allowed:
                continue
            inner = parameters[wrapper]
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    return parameters
            if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
                return inner
        return parameters

    def _convert_raw_parameters(
        self,
        raw_parameters: Dict[str, tuple[str, str]],
        param_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        parameters = {
            param_name: self._convert_raw_param_value(
                param_name, param_value, string_attr, param_config
            )
            for param_name, (param_value, string_attr) in raw_parameters.items()
        }
        return self._repair_param_dict(parameters, param_config)

    def _parse_parameters_from_xml(
        self,
        invoke_content: str,
        allow_partial: bool = False,
        param_config: Dict[str, Any] | None = None,
    ) -> dict:
        """
        Parse parameters from either XML-like format or JSON format to dict.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()

        if invoke_content_stripped.startswith("{") and invoke_content_stripped.endswith(
            "}"
        ):
            try:
                parameters = json.loads(invoke_content_stripped)
                if isinstance(parameters, dict):
                    return parameters
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, fall through to XML parsing
                pass

        # Fall back to XML parameter tag parsing (original format)
        raw_parameters: Dict[str, tuple[str, str]] = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        for match in param_matches:
            param_name = match.group(1)
            string_attr = match.group(2)
            param_value = match.group(3)
            last_match_end = match.end()
            raw_parameters[param_name] = (param_value, string_attr)

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            remaining_content = invoke_content[last_match_end:]

            # Strip only an actual trailing prefix of the closing parameter tag
            # (never character-set rstrip, which corrupts ending characters).
            remaining_content = _strip_trailing_tag_prefix(
                remaining_content, "</｜DSML｜parameter>"
            )

            # Match start of a parameter tag + value (potentially incomplete)
            # Only emit partial values for string parameters so typed literals
            # (boolean/integer/array/object) do not corrupt JSON mid-stream.
            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )

            if (
                partial_match
                and partial_match.group(2) == "true"
                and (param_value := partial_match.group(3))
            ):
                param_name = partial_match.group(1)
                raw_parameters[param_name] = (param_value, partial_match.group(2))

        return self._convert_raw_parameters(raw_parameters, param_config or {})

    def _parse_with_regex(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Fallback regex-based parsing when official parser is not available."""
        idx = text.find(self.bot_token)
        normal_text = text[:idx] if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            # Extract content between function_calls tags
            function_calls_match = re.search(
                self.function_calls_regex,
                text,
                re.DOTALL,
            )
            if not function_calls_match:
                if _dsv4_detector_debug_enabled():
                    logger.warning(
                        "[DeepSeekV4RendererDebug] detector_regex_no_complete_block "
                        "normal_len=%d text_preview=%s",
                        len(normal_text),
                        _preview_text(text),
                    )
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            invoke_matches = re.findall(
                self.invoke_regex, function_calls_content, re.DOTALL
            )
            if _dsv4_detector_debug_enabled():
                logger.info(
                    "[DeepSeekV4RendererDebug] detector_regex_matches "
                    "normal_len=%d block_len=%d invokes=%d",
                    len(normal_text),
                    len(function_calls_content),
                    len(invoke_matches),
                )

            for func_name, invoke_content, _ in invoke_matches:
                # Parse parameters from XML format
                func_args = self._parse_parameters_from_xml(
                    invoke_content,
                    param_config=self._get_param_config(func_name, tools),
                )
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(
                    self.parse_base_json(match_result, tools, start_index=len(calls))
                )

            if _dsv4_detector_debug_enabled():
                logger.info(
                    "[DeepSeekV4RendererDebug] detector_regex_return calls=%d",
                    len(calls),
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def _convert_raw_param_value(
        self,
        param_name: str,
        param_value: str,
        string_attr: str,
        param_config: Dict[str, Any],
    ) -> Any:
        if string_attr == "true":
            return param_value
        param_type: Any = "string"
        if param_name in param_config and isinstance(param_config[param_name], dict):
            param_type = param_config[param_name].get("type", "string")
        return self._convert_param_value(param_value, param_type)

    def _emit_tool_args_fragment(
        self, fragment: str, all_calls: List[ToolCallItem]
    ) -> None:
        if not fragment:
            return
        all_calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                name=None,
                parameters=fragment,
            )
        )
        self.streamed_args_for_tool[self.current_tool_id] += fragment

    def _start_invoke(
        self, func_name: str, tools: List[Tool], all_calls: List[ToolCallItem]
    ) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        if not self.current_tool_name_sent:
            all_calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=func_name,
                    parameters="",
                )
            )
            self.current_tool_name_sent = True

        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": {},
        }
        self._cur_func_name = func_name
        self._cur_param_config = self._get_param_config(func_name, tools)
        self._cur_param_name = ""
        self._cur_param_string_attr = "true"
        self._cur_defer_repair = False
        self._cur_invoke_param_count = 0
        self._cur_invoke_opened_json = False
        self._cur_deferred_params = {}
        self._cur_param_chunks = []
        self._stream_state = "IN_INVOKE"

    def _reset_invoke_state(self) -> None:
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self._cur_func_name = ""
        self._cur_param_config = {}
        self._cur_param_name = ""
        self._cur_param_string_attr = "true"
        self._cur_defer_repair = False
        self._cur_invoke_param_count = 0
        self._cur_invoke_opened_json = False
        self._cur_deferred_params = {}
        self._cur_param_chunks = []
        self._stream_state = "IN_BLOCK"

    def _flush_deferred_before_next_param(self, all_calls: List[ToolCallItem]) -> None:
        if not self._cur_deferred_params:
            return
        for pname, (pval, sattr) in self._cur_deferred_params.items():
            converted = self._convert_raw_param_value(
                pname, pval, sattr, self._cur_param_config
            )
            self.prev_tool_call_arr[self.current_tool_id].setdefault("arguments", {})[
                pname
            ] = converted
            prefix = "{" if not self._cur_invoke_opened_json else ", "
            self._cur_invoke_opened_json = True
            fragment = (
                prefix
                + json.dumps(pname, ensure_ascii=False)
                + ": "
                + json.dumps(converted, ensure_ascii=False)
            )
            self._emit_tool_args_fragment(fragment, all_calls)
        self._cur_deferred_params = {}
        self._cur_defer_repair = False

    def _finish_invoke_call(self, all_calls: List[ToolCallItem]) -> None:
        if self._cur_deferred_params:
            params = self._convert_raw_parameters(
                self._cur_deferred_params, self._cur_param_config
            )
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": self._cur_func_name,
                "arguments": params,
            }
            self._emit_tool_args_fragment(
                json.dumps(params, ensure_ascii=False), all_calls
            )
        elif self._cur_invoke_opened_json:
            self._emit_tool_args_fragment("}", all_calls)
        else:
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": self._cur_func_name,
                "arguments": {},
            }
            self._emit_tool_args_fragment("{}", all_calls)
        self._reset_invoke_state()

    def _finalize_buffered_param(
        self, raw_val: str, all_calls: List[ToolCallItem]
    ) -> None:
        if self._cur_defer_repair:
            self._cur_deferred_params[self._cur_param_name] = (
                raw_val,
                self._cur_param_string_attr,
            )
            self._cur_defer_repair = False
        else:
            converted = self._convert_raw_param_value(
                self._cur_param_name,
                raw_val,
                self._cur_param_string_attr,
                self._cur_param_config,
            )
            self.prev_tool_call_arr[self.current_tool_id].setdefault("arguments", {})[
                self._cur_param_name
            ] = converted
            prefix = "{" if not self._cur_invoke_opened_json else ", "
            self._cur_invoke_opened_json = True
            fragment = (
                prefix
                + json.dumps(self._cur_param_name, ensure_ascii=False)
                + ": "
                + json.dumps(converted, ensure_ascii=False)
            )
            self._emit_tool_args_fragment(fragment, all_calls)
        self._cur_param_name = ""
        self._cur_param_chunks = []
        self._stream_state = "IN_INVOKE"

    def _strip_stray_end_tags(self, text: str) -> str:
        for e_token in (self.eot_token, self.invoke_end_token):
            if e_token in text:
                text = text.replace(e_token, "")
        return text

    def _find_earliest_tag(self, text: str, tags: tuple[str, ...]) -> tuple[int, str]:
        best_pos = -1
        best_tag = ""
        for tag in tags:
            pos = text.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_tag = tag
        return best_pos, best_tag

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Linear-time streaming incremental parser for DeepSeekV4 DSML tool calls.
        Maintains an explicit state machine over incoming chunks so typed/non-string
        parameters never corrupt JSON mid-stream and large tool arguments run in O(N).
        """
        self._buffer += new_text
        normal_parts: List[str] = []
        all_calls: List[ToolCallItem] = []

        outside_tags = (
            self.bot_token,
            "<｜DSML｜invoke",
            self.eot_token,
            self.invoke_end_token,
        )
        block_tags = ("<｜DSML｜invoke", self.eot_token)
        param_boundary_tags = (
            self._param_end_tag,
            self.invoke_end_token,
            self.eot_token,
        )
        direct_json_end_tags = (self.invoke_end_token, self.eot_token)

        try:
            while self._buffer:
                if self._stream_state == "OUTSIDE":
                    bot_pos = self._buffer.find(self.bot_token)
                    invoke_match = self._invoke_open_re.search(self._buffer)
                    if bot_pos != -1 and (
                        invoke_match is None or bot_pos <= invoke_match.start()
                    ):
                        prefix = self._strip_stray_end_tags(self._buffer[:bot_pos])
                        if prefix and (self._emitted_normal_text or prefix.strip()):
                            normal_parts.append(prefix)
                            self._emitted_normal_text = True
                        self._buffer = self._buffer[bot_pos + len(self.bot_token) :]
                        self._in_tool_calls_block = True
                        self._stream_state = "IN_BLOCK"
                        continue
                    if invoke_match is not None:
                        prefix = self._strip_stray_end_tags(
                            self._buffer[: invoke_match.start()]
                        )
                        if prefix and (self._emitted_normal_text or prefix.strip()):
                            normal_parts.append(prefix)
                            self._emitted_normal_text = True
                        self._buffer = self._buffer[invoke_match.start() :]
                        self._in_tool_calls_block = True
                        self._stream_state = "IN_BLOCK"
                        continue

                    inv_prefix_pos = self._buffer.find("<｜DSML｜invoke")
                    if inv_prefix_pos != -1:
                        prefix = self._strip_stray_end_tags(
                            self._buffer[:inv_prefix_pos]
                        )
                        if prefix and (self._emitted_normal_text or prefix.strip()):
                            normal_parts.append(prefix)
                            self._emitted_normal_text = True
                        self._buffer = self._buffer[inv_prefix_pos:]
                        self._in_tool_calls_block = True
                        self._stream_state = "IN_BLOCK"
                        continue

                    hold_k = max(
                        _longest_suffix_prefix_len(self._buffer, tag)
                        for tag in outside_tags
                    )
                    safe_len = len(self._buffer) - hold_k
                    if safe_len <= 0:
                        break
                    emit = self._strip_stray_end_tags(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                    if emit:
                        normal_parts.append(emit)
                        self._emitted_normal_text = True
                    continue

                if self._stream_state == "IN_BLOCK":
                    invoke_match = self._invoke_open_re.search(self._buffer)
                    eot_pos = self._buffer.find(self.eot_token)
                    if eot_pos != -1 and (
                        invoke_match is None or eot_pos < invoke_match.start()
                    ):
                        self._buffer = self._buffer[eot_pos + len(self.eot_token) :]
                        self._in_tool_calls_block = False
                        self._stream_state = "OUTSIDE"
                        continue
                    if invoke_match is not None:
                        func_name = invoke_match.group(1).strip()
                        self._buffer = self._buffer[invoke_match.end() :]
                        self._start_invoke(func_name, tools, all_calls)
                        continue

                    hold_k = max(
                        _longest_suffix_prefix_len(self._buffer, tag)
                        for tag in block_tags
                    )
                    if hold_k > 0:
                        self._buffer = self._buffer[len(self._buffer) - hold_k :]
                    elif "<" in self._buffer:
                        self._buffer = self._buffer[self._buffer.rfind("<") :]
                    else:
                        self._buffer = ""
                    break

                if self._stream_state == "IN_INVOKE":
                    stripped = self._buffer.lstrip()
                    if not stripped:
                        self._buffer = ""
                        break
                    if stripped.startswith("{"):
                        self._buffer = stripped
                        self._cur_param_chunks = []
                        self._stream_state = "IN_DIRECT_JSON"
                        continue

                    param_match = self._param_open_re.search(self._buffer)
                    next_invoke_match = self._invoke_open_re.search(self._buffer)
                    end_pos, end_tag = self._find_earliest_tag(
                        self._buffer, (self.invoke_end_token, self.eot_token)
                    )
                    earliest_close_pos = end_pos
                    earliest_close_kind = (
                        "invoke_end" if end_tag == self.invoke_end_token else "eot"
                    )
                    if next_invoke_match is not None and (
                        earliest_close_pos == -1
                        or next_invoke_match.start() < earliest_close_pos
                    ):
                        earliest_close_pos = next_invoke_match.start()
                        earliest_close_kind = "next_invoke"

                    if earliest_close_pos != -1 and (
                        param_match is None or earliest_close_pos < param_match.start()
                    ):
                        if earliest_close_kind == "invoke_end":
                            self._buffer = self._buffer[
                                earliest_close_pos + len(self.invoke_end_token) :
                            ]
                        else:
                            self._buffer = self._buffer[earliest_close_pos:]
                        self._finish_invoke_call(all_calls)
                        continue

                    if param_match is not None:
                        self._flush_deferred_before_next_param(all_calls)
                        param_name = param_match.group(1)
                        string_attr = param_match.group(2)
                        self._buffer = self._buffer[param_match.end() :]
                        self._cur_param_name = param_name
                        self._cur_param_string_attr = string_attr
                        self._cur_defer_repair = (
                            self._cur_invoke_param_count == 0
                            and param_name in ("arguments", "input")
                            and param_name not in self._cur_param_config
                        )
                        self._cur_invoke_param_count += 1
                        self._cur_param_chunks = []

                        if string_attr == "true" and not self._cur_defer_repair:
                            prefix = (
                                ("{" if not self._cur_invoke_opened_json else ", ")
                                + json.dumps(param_name, ensure_ascii=False)
                                + ': "'
                            )
                            self._cur_invoke_opened_json = True
                            self._emit_tool_args_fragment(prefix, all_calls)
                            self._stream_state = "IN_STRING_PARAM"
                        else:
                            self._stream_state = "IN_BUFFERED_PARAM"
                        continue

                    # Wait for complete <parameter ...> or </invoke> tag; discard only leading whitespace.
                    self._buffer = stripped
                    break

                if self._stream_state == "IN_STRING_PARAM":
                    boundary_pos, boundary_tag = self._find_earliest_tag(
                        self._buffer, param_boundary_tags
                    )
                    if boundary_pos != -1:
                        raw_tail = self._buffer[:boundary_pos]
                        escaped_tail = json.dumps(raw_tail, ensure_ascii=False)[1:-1]
                        self._emit_tool_args_fragment(escaped_tail + '"', all_calls)
                        if boundary_tag == self._param_end_tag:
                            self._buffer = self._buffer[
                                boundary_pos + len(self._param_end_tag) :
                            ]
                        else:
                            # Malformed output closed invoke/block without </｜DSML｜parameter>.
                            self._buffer = self._buffer[boundary_pos:]
                        self._cur_param_name = ""
                        self._stream_state = "IN_INVOKE"
                        continue

                    hold_k = max(
                        _longest_suffix_prefix_len(self._buffer, tag)
                        for tag in param_boundary_tags
                    )
                    safe_len = len(self._buffer) - hold_k
                    if safe_len <= 0:
                        break
                    emit_raw = self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    escaped_emit = json.dumps(emit_raw, ensure_ascii=False)[1:-1]
                    self._emit_tool_args_fragment(escaped_emit, all_calls)
                    continue

                if self._stream_state == "IN_BUFFERED_PARAM":
                    boundary_pos, boundary_tag = self._find_earliest_tag(
                        self._buffer, param_boundary_tags
                    )
                    if boundary_pos != -1:
                        self._cur_param_chunks.append(self._buffer[:boundary_pos])
                        raw_val = "".join(self._cur_param_chunks)
                        if boundary_tag == self._param_end_tag:
                            self._buffer = self._buffer[
                                boundary_pos + len(self._param_end_tag) :
                            ]
                        else:
                            self._buffer = self._buffer[boundary_pos:]
                        self._finalize_buffered_param(raw_val, all_calls)
                        continue

                    hold_k = max(
                        _longest_suffix_prefix_len(self._buffer, tag)
                        for tag in param_boundary_tags
                    )
                    safe_len = len(self._buffer) - hold_k
                    if safe_len <= 0:
                        break
                    self._cur_param_chunks.append(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                    continue

                if self._stream_state == "IN_DIRECT_JSON":
                    close_pos, close_tag = self._find_earliest_tag(
                        self._buffer, direct_json_end_tags
                    )
                    if close_pos != -1:
                        self._cur_param_chunks.append(self._buffer[:close_pos])
                        raw_json = "".join(self._cur_param_chunks).strip()
                        self._cur_param_chunks = []
                        if close_tag == self.invoke_end_token:
                            self._buffer = self._buffer[
                                close_pos + len(self.invoke_end_token) :
                            ]
                        else:
                            self._buffer = self._buffer[close_pos:]
                        params = self._parse_parameters_from_xml(
                            raw_json,
                            allow_partial=False,
                            param_config=self._cur_param_config,
                        )
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": self._cur_func_name,
                            "arguments": params,
                        }
                        self._emit_tool_args_fragment(
                            json.dumps(params, ensure_ascii=False), all_calls
                        )
                        self._reset_invoke_state()
                        continue

                    hold_k = max(
                        _longest_suffix_prefix_len(self._buffer, tag)
                        for tag in direct_json_end_tags
                    )
                    safe_len = len(self._buffer) - hold_k
                    if safe_len <= 0:
                        break
                    self._cur_param_chunks.append(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                    continue

            return StreamingParseResult(
                normal_text="".join(normal_parts), calls=all_calls
            )
        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            remaining = self._buffer
            self._buffer = ""
            return StreamingParseResult(
                normal_text="".join(normal_parts) + remaining, calls=all_calls
            )

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger=f"<｜DSML｜invoke",
        )

    def tool_call_structural_tag(
        self,
        tools: List[Tool],
        *,
        stop_after_first: bool,
    ) -> Dict[str, Any]:
        """Build DeepSeek-V4 native structural_tag for forced tool-call decoding."""
        get_structure_info = self.structure_info()
        invoke_tags = []
        for tool in tools:
            function = tool.function
            info = get_structure_info(function.name)
            invoke_tags.append(
                {
                    "type": "tag",
                    "begin": info.begin,
                    "content": {
                        "type": "json_schema",
                        "json_schema": function.parameters or {},
                    },
                    "end": info.end,
                }
            )

        return {
            "format": {
                "type": "tag",
                "begin": f"{self.bot_token}\n",
                "content": {
                    "type": "tags_with_separator",
                    "tags": invoke_tags,
                    "separator": "\n",
                    "at_least_one": True,
                    "stop_after_first": stop_after_first,
                },
                "end": f"\n{self.eot_token}",
            }
        }
