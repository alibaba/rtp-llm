import json
import logging
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
from rtp_llm.openai.renderers.sglang_helpers.function_call.utils import (
    _find_common_prefix,
)

logger = logging.getLogger(__name__)


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
        self.function_calls_regex = (
            r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
        )
        self.invoke_regex = (
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)'
        )
        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
        self.current_tool_id = -1
        self.encoding_module = encoding_module
        self.thinking_mode = thinking_mode

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
            return StreamingParseResult(normal_text=text, calls=[])

        logger.info(
            f"[DeepSeekV4Detector] detect_and_parse called with thinking_mode={self.thinking_mode}, text_length={len(text)}"
        )

        # Try to use official parser if available
        if self.encoding_module and hasattr(
            self.encoding_module, "parse_message_from_completion_text"
        ):
            try:
                logger.info(
                    f"[DeepSeekV4Detector] Using official parse_message_from_completion_text"
                )
                # Use the official high-level parser
                parsed_msg = self.encoding_module.parse_message_from_completion_text(
                    text, self.thinking_mode
                )

                # Extract content and reasoning content. vLLM's current
                # encoding parser returns "reasoning"; the checkpoint parser
                # used by RTP-LLM returns "reasoning_content".
                content = parsed_msg.get("content", "")
                reasoning_content = parsed_msg.get(
                    "reasoning_content", parsed_msg.get("reasoning", "")
                )
                tool_calls_list = parsed_msg.get("tool_calls", [])

                logger.info(
                    f"[DeepSeekV4Detector] Official parser succeeded: reasoning={len(reasoning_content)}, content={len(content)}, tool_calls={len(tool_calls_list)}"
                )

                # Combine reasoning and content as normal_text
                normal_text_parts = []
                if reasoning_content:
                    normal_text_parts.append(reasoning_content)
                if content:
                    normal_text_parts.append(content)
                normal_text = "\n\n".join(normal_text_parts)

                calls = []
                for tc in tool_calls_list:
                    # tool_calls are already in OpenAI format with 'function' key
                    function = tc.get("function", {})
                    # arguments is a JSON string; convert to dict
                    arguments = function.get("arguments") or "{}"
                    arguments_dict = (
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments
                    )
                    tool_item = {
                        "name": function.get("name"),
                        "parameters": arguments_dict,  # Pass dict, not JSON string
                    }
                    calls.extend(
                        self.parse_base_json(tool_item, tools, start_index=len(calls))
                    )

                logger.info(
                    f"[DeepSeekV4Detector] Returning {len(calls)} parsed tool calls"
                )
                return StreamingParseResult(normal_text=normal_text, calls=calls)
            except Exception as e:
                logger.warning(
                    f"[DeepSeekV4Detector] Official parser failed, falling back to regex: {e}"
                )

        # Fallback to regex-based parsing
        logger.info(f"[DeepSeekV4Detector] Using regex fallback parser")
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
        parameters: Dict[str, Any] = {}
        for param_name, (param_value, string_attr) in raw_parameters.items():
            if string_attr == "true":
                parameters[param_name] = param_value
                continue

            param_type = "string"
            if param_name in param_config and isinstance(param_config[param_name], dict):
                param_type = param_config[param_name].get("type", "string")
            parameters[param_name] = self._convert_param_value(param_value, param_type)

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

            # Remove incomplete parameter_end_call prefix in case they are captured by param
            for token in reversed(self.prefix_parameter_end_call):
                remaining_content = remaining_content.rstrip(token)

            # Match start of a parameter tag + value (potentially incomplete)
            # Regex: <tag name="..." string="...">VALUE... (no end tag)
            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )

            if partial_match and (param_value := partial_match.group(3)):
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
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            invoke_matches = re.findall(
                self.invoke_regex, function_calls_content, re.DOTALL
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

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV4 format.
        Supports multiple consecutive invoke blocks and argument streaming.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if buffer contains any DSML markers or ends with potential tag prefix
        # This handles partial/streaming DSML content
        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)

        # Also check if text ends with start of a tag (to handle "<" arriving separately)
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(
            current_text.rstrip().endswith(prefix) for prefix in dsml_prefixes
        )

        if (
            not self.has_tool_call(current_text)
            and not potentially_dsml
            and not ends_with_prefix
        ):
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        all_calls: list[ToolCallItem] = []
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=self.invoke_regex,
                    string=current_text,
                    flags=re.DOTALL,
                )
                if not invoke_match:
                    break

                func_name = invoke_match.group(1).strip()
                invoke_content = invoke_match.group(2)
                # group(3) is either "</｜DSML｜invoke>" (complete) or "" (incomplete, matched with $)
                is_tool_end = bool(invoke_match.group(3))

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure arrays are large enough for current tool
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # 1. Send tool name if not sent yet
                if not self.current_tool_name_sent:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True

                # 2. Parse current parameters (partial or complete)
                current_params = self._parse_parameters_from_xml(
                    invoke_content,
                    allow_partial=not is_tool_end,
                    param_config=self._get_param_config(func_name, tools),
                )
                current_args_json = json.dumps(current_params, ensure_ascii=False)

                # 3. Calculate and send incremental arguments
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff = None

                if is_tool_end:
                    # If complete, send everything remaining
                    argument_diff = current_args_json[sent_len:]
                elif prev_params is not None:
                    # If partial, send stable prefix diff
                    prev_args_json = json.dumps(prev_params, ensure_ascii=False)
                    if current_args_json != prev_args_json:
                        prefix = _find_common_prefix(prev_args_json, current_args_json)
                        if len(prefix) > sent_len:
                            argument_diff = prefix[sent_len:]

                if argument_diff:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # Update the stored arguments
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer  # Update for next iteration

                    # Move to next tool call
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # No more invoke blocks found
            return StreamingParseResult(normal_text="", calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger=f"<｜DSML｜invoke",
        )
