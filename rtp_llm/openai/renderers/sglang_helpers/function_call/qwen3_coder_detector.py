import ast
import json
import logging
import re
from typing import Any, List, Optional

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import Tool
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # Buffer for whitespace between consecutive tool calls.
        # After </tool_call>, whitespace is buffered here. If <tool_call> follows,
        # the buffer is discarded. If real content follows, the buffer is flushed
        # as normal text.
        self._inter_tool_whitespace_buffer: str = ""
        # Flag to track if we've completed at least one tool call.
        # Whitespace is only buffered AFTER a </tool_call>, not before the first tool.
        self._has_completed_tool_call: bool = False

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None
        self._current_args_config: dict = {}
        self._current_tool_arguments: dict[str, Any] = {}

        # Streaming parameter state (only used for string-type parameters)
        self._streaming_param_name: Optional[str] = None
        self._streaming_param_value_parts: list[str] = []
        self._streaming_param_pending_newline: bool = False
        self._streaming_param_leading_newline_checked: bool = False
        # We delay emitting the opening quote until we know the value isn't a JSON null.
        # This preserves the legacy behavior where raw "null" converts to JSON null even
        # for string-typed parameters.
        self._streaming_param_quote_emitted: bool = False
        self._streaming_param_undecided_buffer: str = ""

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[list[Tool]]
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            try:
                config_type = config.type
                config_function = config.function
                config_function_name = config_function.name
            except AttributeError:
                continue

            if config_type == "function" and config_function_name == func_name:
                try:
                    params = config_function.parameters
                except AttributeError:
                    return {}

                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _is_string_param(self, param_name: str) -> bool:
        """Return True if param should be streamed as a JSON string."""
        if (
            not isinstance(self._current_args_config, dict)
            or not self._current_args_config
        ):
            return True

        if param_name not in self._current_args_config:
            return True

        cfg = self._current_args_config.get(param_name)
        if isinstance(cfg, dict) and "type" in cfg:
            param_type = str(cfg["type"]).strip().lower()
        else:
            param_type = "string"

        return param_type in ["string", "str", "text", "varchar", "char", "enum"]

    def _ensure_tool_tracking_capacity(self, tool_id: int) -> None:
        while len(self.prev_tool_call_arr) <= tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= tool_id:
            self.streamed_args_for_tool.append("")

    def _emit_arg_fragment(self, calls: list[ToolCallItem], fragment: str) -> None:
        if not fragment:
            return
        calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters=fragment))
        if self.current_tool_id >= 0:
            self._ensure_tool_tracking_capacity(self.current_tool_id)
            self.streamed_args_for_tool[self.current_tool_id] += fragment

    def _escape_json_string_segment(self, text: str) -> str:
        # json.dumps returns a quoted JSON string, we need the content only.
        return json.dumps(text, ensure_ascii=False)[1:-1]

    def _flush_streaming_param_newline_if_needed(
        self, calls: list[ToolCallItem]
    ) -> None:
        if not self._streaming_param_pending_newline:
            return
        self._streaming_param_pending_newline = False
        self._streaming_param_value_parts.append("\n")
        if self._streaming_param_quote_emitted:
            self._emit_arg_fragment(calls, self._escape_json_string_segment("\n"))
        else:
            self._streaming_param_undecided_buffer += "\n"

    def _consume_streaming_param_value_text(
        self, text: str, calls: list[ToolCallItem]
    ) -> None:
        if not text:
            return

        if not self._streaming_param_leading_newline_checked:
            self._streaming_param_leading_newline_checked = True
            if text.startswith("\n"):
                text = text[1:]
                if not text:
                    return

        # Hold back a single trailing '\n' so we can drop it if it's immediately
        # followed by a closing tag (legacy strip behavior).
        if text.endswith("\n"):
            text, trailing = text[:-1], True
        else:
            trailing = False

        # We have content (even if it's empty after stripping trailing \n).
        # Flush any previous pending newline before we potentially hold back a new one.
        # This prevents losing interior newlines when consecutive \n chunks arrive.
        if trailing or text:
            self._flush_streaming_param_newline_if_needed(calls)

        if text:
            self._streaming_param_value_parts.append(text)

            if self._streaming_param_quote_emitted:
                self._emit_arg_fragment(calls, self._escape_json_string_segment(text))
            else:
                self._streaming_param_undecided_buffer += text
                if not "null".startswith(
                    self._streaming_param_undecided_buffer.lower()
                ):
                    self._emit_arg_fragment(calls, '"')
                    self._emit_arg_fragment(
                        calls,
                        self._escape_json_string_segment(
                            self._streaming_param_undecided_buffer
                        ),
                    )
                    self._streaming_param_quote_emitted = True
                    self._streaming_param_undecided_buffer = ""

        if trailing:
            self._streaming_param_pending_newline = True

    def _finish_streaming_param(self, calls: list[ToolCallItem]) -> None:
        if self._streaming_param_name is None:
            return

        # Drop one trailing '\n' if it was held back.
        self._streaming_param_pending_newline = False

        raw_value = "".join(self._streaming_param_value_parts)
        converted_value = (
            None if raw_value.lower() == "null" else raw_value
        )  # preserve legacy null handling
        self._current_tool_arguments[self._streaming_param_name] = converted_value

        if self._streaming_param_quote_emitted:
            self._emit_arg_fragment(calls, '"')
        else:
            if converted_value is None:
                self._emit_arg_fragment(calls, "null")
            else:
                self._emit_arg_fragment(calls, '"')
                if raw_value:
                    self._emit_arg_fragment(
                        calls, self._escape_json_string_segment(raw_value)
                    )
                self._emit_arg_fragment(calls, '"')

        self._streaming_param_name = None
        self._streaming_param_value_parts = []
        self._streaming_param_pending_newline = False
        self._streaming_param_leading_newline_checked = False
        self._streaming_param_quote_emitted = False
        self._streaming_param_undecided_buffer = ""

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    False if "." in param_value or "e" in param_value.lower() else True
                )
                param_value: float = float(param_value)
                if maybe_convert and param_value.is_integer():
                    param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            # Simple cleanup of the text to find tool calls
            # Note: This is a simplified regex approach consistent with vLLM
            raw_tool_calls = self.tool_call_regex.findall(text)
            if not raw_tool_calls:
                # Fallback: maybe the whole text is inside the tag or tags are stripped
                if self.tool_call_prefix in text:
                    raw_tool_calls = [text]

            tool_idx = 0
            for tool_content in raw_tool_calls:
                # Find function calls
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_match in funcs:
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    params_str = func_body[name_end + 1 :]

                    param_config = self._get_arguments_config(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        parsed_params[p_name] = self._convert_param_value(
                            p_val, p_name, param_config, func_name
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx,
                            name=func_name,
                            parameters=json.dumps(parsed_params, ensure_ascii=False),
                        )
                    )
                    tool_idx += 1

            # Determine normal text (text before the first tool call)
            start_idx = text.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = text.find(self.tool_call_prefix)
            normal_text = text[:start_idx] if start_idx > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser.
        """
        self._buffer += new_text

        # Guard against empty buffer
        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            # Working text slice
            current_slice = self._buffer[self.parsed_pos :]

            # Optimization: If almost empty, wait for more
            if not current_slice:
                break

            # -------------------------------------------------------
            # 0. Streaming parameter value mode (string parameters only)
            # -------------------------------------------------------
            if self._streaming_param_name is not None:
                if current_slice.startswith(self.parameter_end_token):
                    self._finish_streaming_param(calls)
                    self.parsed_pos += len(self.parameter_end_token)
                    continue
                implicit_end_tokens = [
                    self.parameter_prefix,
                    self.function_end_token,
                    self.tool_call_end_token,
                    self.tool_call_start_token,
                    self.tool_call_prefix,
                ]
                if any(current_slice.startswith(t) for t in implicit_end_tokens):
                    # Implicit parameter end (malformed output) - finalize value and
                    # let the next tag be parsed by the main loop.
                    self._finish_streaming_param(calls)
                    continue

                next_open_angle = current_slice.find("<")
                if next_open_angle == -1:
                    self._consume_streaming_param_value_text(current_slice, calls)
                    self.parsed_pos += len(current_slice)
                    continue
                if next_open_angle > 0:
                    self._consume_streaming_param_value_text(
                        current_slice[:next_open_angle], calls
                    )
                    self.parsed_pos += next_open_angle
                    continue

                # current_slice starts with '<' but doesn't form a known delimiter yet.
                possible_delimiters = [
                    self.parameter_end_token,
                    *implicit_end_tokens,
                ]
                if any(d.startswith(current_slice) for d in possible_delimiters):
                    break

                # Treat '<' as a literal character inside the value.
                self._consume_streaming_param_value_text("<", calls)
                self.parsed_pos += 1
                continue

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                # Discard any buffered inter-tool whitespace (newlines between tools)
                self._inter_tool_whitespace_buffer = ""
                continue

            # -------------------------------------------------------
            # 2. Function Name: <function=name>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = current_slice[len(self.tool_call_prefix) : end_angle]

                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self.current_func_name = func_name
                    self._current_args_config = self._get_arguments_config(
                        func_name, tools
                    )
                    self._current_tool_arguments = {}
                    self._ensure_tool_tracking_capacity(self.current_tool_id)
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    self.parsed_pos += end_angle + 1
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter=name>value...
            # -------------------------------------------------------
            if current_slice.startswith(self.parameter_prefix):
                name_end = current_slice.find(">")
                if name_end != -1:
                    param_name = current_slice[len(self.parameter_prefix) : name_end]

                    # JSON Construction
                    if not self.json_started:
                        self._emit_arg_fragment(calls, "{")
                        self.json_started = True

                    if self._is_string_param(param_name):
                        prefix = (
                            ", " if self.current_tool_param_count > 0 else ""
                        ) + f"{json.dumps(param_name)}: "
                        self._emit_arg_fragment(calls, prefix)
                        self.current_tool_param_count += 1

                        # Enter streaming mode for this parameter's value
                        self._streaming_param_name = param_name
                        self._streaming_param_value_parts = []
                        self._streaming_param_pending_newline = False
                        self._streaming_param_leading_newline_checked = False
                        self._streaming_param_quote_emitted = False
                        self._streaming_param_undecided_buffer = ""

                        self.parsed_pos += name_end + 1
                        continue

                    value_start_idx = name_end + 1
                    rest_of_slice = current_slice[value_start_idx:]

                    # A parameter can end in multiple ways:
                    # 1. [Normal] Encounter </parameter>
                    # 2. [Abnormal] Encounter next <parameter=
                    # 3. [Abnormal] Encounter </function>
                    # So we need to find the smallest one as the parameter end position.
                    cand_end_param = rest_of_slice.find(self.parameter_end_token)
                    cand_next_param = rest_of_slice.find(self.parameter_prefix)
                    cand_end_func = rest_of_slice.find(self.function_end_token)

                    candidates = []
                    if cand_end_param != -1:
                        candidates.append(
                            (cand_end_param, len(self.parameter_end_token))
                        )
                    if cand_next_param != -1:
                        candidates.append((cand_next_param, 0))
                    if cand_end_func != -1:
                        candidates.append((cand_end_func, 0))

                    if candidates:
                        best_cand = min(candidates, key=lambda x: x[0])
                        end_pos = best_cand[0]
                        end_token_len = best_cand[1]

                        raw_value = rest_of_slice[:end_pos]

                        # Cleanup value
                        if raw_value.startswith("\n"):
                            raw_value = raw_value[1:]
                        if raw_value.endswith("\n"):
                            raw_value = raw_value[:-1]

                        converted_val = self._convert_param_value(
                            raw_value,
                            param_name,
                            self._current_args_config,
                            self.current_func_name,
                        )
                        self._current_tool_arguments[param_name] = converted_val

                        # Construct JSON fragment: "key": value
                        json_key_val = f"{json.dumps(param_name)}: {json.dumps(converted_val, ensure_ascii=False)}"

                        fragment = (
                            f", {json_key_val}"
                            if self.current_tool_param_count > 0
                            else json_key_val
                        )

                        self._emit_arg_fragment(calls, fragment)
                        self.current_tool_param_count += 1

                        # Advance cursor
                        total_len = (name_end + 1) + end_pos + end_token_len
                        self.parsed_pos += total_len
                        continue

                # Incomplete parameter tag or value
                break

            # -------------------------------------------------------
            # 4. Function End: </function>
            # -------------------------------------------------------
            if current_slice.startswith(self.function_end_token):
                if self._streaming_param_name is not None:
                    self._finish_streaming_param(calls)

                if not self.json_started:
                    self._emit_arg_fragment(calls, "{")
                    self.json_started = True

                self._emit_arg_fragment(calls, "}")
                self.parsed_pos += len(self.function_end_token)
                self._ensure_tool_tracking_capacity(self.current_tool_id)
                if self.current_tool_id >= 0 and self.current_tool_id < len(
                    self.prev_tool_call_arr
                ):
                    self.prev_tool_call_arr[self.current_tool_id][
                        "arguments"
                    ] = self._current_tool_arguments
                self.current_func_name = None
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </tool_call>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_end_token):
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False  # [FIX] Exit tool call region
                self._has_completed_tool_call = (
                    True  # Enable inter-tool whitespace buffering
                )
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            # If current position is not the start of a tag (i.e., doesn't start with <), it might be plain text,
            # or a newline between two tags.
            # But we need to be careful not to output truncated tags like "<fun" as text.

            next_open_angle = current_slice.find("<")

            if next_open_angle == -1:
                # This entire segment is plain text
                if not self.is_inside_tool_call:
                    # Only buffer whitespace if we've completed at least one tool call
                    # (to strip newlines between consecutive tool calls)
                    if self._has_completed_tool_call and current_slice.strip() == "":
                        # Buffer it - might be discarded if followed by <tool_call>
                        self._inter_tool_whitespace_buffer += current_slice
                    else:
                        # Real content - flush any buffered whitespace first
                        if self._inter_tool_whitespace_buffer:
                            normal_text_chunks.append(
                                self._inter_tool_whitespace_buffer
                            )
                            self._inter_tool_whitespace_buffer = ""
                        normal_text_chunks.append(current_slice)
                # [FIX] If inside tool call, discard this text (usually \n), don't append
                self.parsed_pos += len(current_slice)
                continue

            elif next_open_angle == 0:
                # Looks like a Tag, but doesn't match any known Tag above

                possible_tags = [
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ]

                is_potential_tag = False
                for tag in possible_tags:
                    if tag.startswith(current_slice):
                        is_potential_tag = True
                        break

                if is_potential_tag:
                    break  # Wait for more
                else:
                    # Just a plain '<' symbol - this is real content
                    if not self.is_inside_tool_call:
                        # Flush any buffered whitespace first
                        if self._inter_tool_whitespace_buffer:
                            normal_text_chunks.append(
                                self._inter_tool_whitespace_buffer
                            )
                            self._inter_tool_whitespace_buffer = ""
                        normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue

            else:
                # '<' is in the middle
                text_segment = current_slice[:next_open_angle]
                if not self.is_inside_tool_call:
                    # Only buffer whitespace if we've completed at least one tool call
                    if self._has_completed_tool_call and text_segment.strip() == "":
                        # Buffer it
                        self._inter_tool_whitespace_buffer += text_segment
                    else:
                        # Real content - flush any buffered whitespace first
                        if self._inter_tool_whitespace_buffer:
                            normal_text_chunks.append(
                                self._inter_tool_whitespace_buffer
                            )
                            self._inter_tool_whitespace_buffer = ""
                        normal_text_chunks.append(text_segment)
                # [FIX] If inside tool call, discard whitespace/text before Tag
                self.parsed_pos += next_open_angle
                continue

        # Memory Cleanup: Slice the buffer
        # Keep unparsed part, discard parsed part
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        # Note: We don't flush _inter_tool_whitespace_buffer here because we're
        # still waiting to see if <tool_call> follows. It will be flushed when
        # real content arrives, or discarded when <tool_call> arrives.

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
