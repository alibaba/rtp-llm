"""DSML Format Detector for DeepSeek V3.2

This detector parses DSML (DeepSeek Markup Language) formatted tool calls
from the model's output stream using the official encoding_dsv32.py parser.
"""

import json
import logging
import re
from typing import List

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
from rtp_llm.openai.renderers.sglang_helpers.function_call.ebnf_composer import (
    EBNFComposer,
)

logger = logging.getLogger(__name__)


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    def __init__(self, encoding_module=None, thinking_mode: str = "chat"):
        super().__init__()
        self.dsml_token = "｜DSML｜"
        self.bot_token = f"<{self.dsml_token}function_calls>"
        self.eot_token = f"</{self.dsml_token}function_calls>"
        self.function_call_regex = re.compile(
            rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}",
            re.DOTALL,
        )
        self.invoke_regex = re.compile(
            rf'<{self.dsml_token}invoke\s+name="([^"]+)"\s*>(.*?)(</{self.dsml_token}invoke>|$)',
            re.DOTALL,
        )
        self.invoke_end_token = f"</{self.dsml_token}invoke>"
        self.parameter_regex = re.compile(
            rf'<{self.dsml_token}parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</{self.dsml_token}parameter>',
            re.DOTALL,
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        self.encoding_module = encoding_module
        self.thinking_mode = thinking_mode  # "thinking" or "chat"

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

        logger.info(f"[DeepSeekV32Detector] detect_and_parse called with thinking_mode={self.thinking_mode}, text_length={len(text)}")

        # Try to use official parser if available
        if self.encoding_module and hasattr(self.encoding_module, "parse_message_from_completion_text"):
            try:
                logger.info(f"[DeepSeekV32Detector] Using official parse_message_from_completion_text")
                # Use the official high-level parser
                parsed_msg = self.encoding_module.parse_message_from_completion_text(text, self.thinking_mode)

                # Extract content and reasoning_content
                content = parsed_msg.get("content", "")
                reasoning_content = parsed_msg.get("reasoning_content", "")
                tool_calls_list = parsed_msg.get("tool_calls", [])

                logger.info(f"[DeepSeekV32Detector] Official parser succeeded: reasoning={len(reasoning_content)}, content={len(content)}, tool_calls={len(tool_calls_list)}")

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
                    arguments_dict = json.loads(function.get("arguments"))
                    tool_item = {
                        "name": function.get("name"),
                        "parameters": arguments_dict  # Pass dict, not JSON string
                    }
                    calls.extend(self.parse_base_json(tool_item, tools))

                logger.info(f"[DeepSeekV32Detector] Returning {len(calls)} parsed tool calls")
                return StreamingParseResult(normal_text=normal_text, calls=calls)
            except Exception as e:
                logger.warning(f"[DeepSeekV32Detector] Official parser failed, falling back to regex: {e}")

        # Fallback to regex-based parsing
        logger.info(f"[DeepSeekV32Detector] Using regex fallback parser")
        return self._parse_with_regex(text, tools)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text

    def _parse_with_regex(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Fallback regex-based parsing when official parser is not available."""
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            # Extract content between function_calls tags
            function_calls_match = self.function_call_regex.search(text)
            if not function_calls_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])
            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            invoke_matches = self.invoke_regex.findall(function_calls_content)
            for func_name, invoke_content in invoke_matches:
                # Parse parameters from XML format
                func_args = self._parse_parameters_from_xml(invoke_content)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
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
        parameters = {}
        param_matches = self.parameter_regex.findall(invoke_content)
        for param_name, param_type, param_value in param_matches:
            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()
        return parameters

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call or any DSML-related content
        # Key insight: DSML tags contain distinctive markers like "｜DSML｜"
        # If we see these markers anywhere, we should keep buffering
        has_tool_call = (
            self.bot_token in current_text or "<｜DSML｜invoke" in current_text
        )

        # Check if buffer contains any DSML markers or ends with potential tag prefix
        # This handles partial/streaming DSML content
        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)

        # Also check if text ends with start of a tag (to handle "<" arriving separately)
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(
            current_text.rstrip().endswith(prefix) for prefix in dsml_prefixes
        )

        if not has_tool_call and not potentially_dsml and not ends_with_prefix:
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        all_calls: list[ToolCallItem] = []
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                invoke_match = self.invoke_regex.search(current_text)
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

                # Don't pre-allocate arrays until we actually complete a tool call
                # This prevents _check_for_unstreamed_tool_args from sending incomplete calls

                # Parse current parameters from XML/JSON
                current_params = self._parse_parameters_from_xml(invoke_content)
                current_args_json = json.dumps(current_params, ensure_ascii=False)

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Only emit the tool call when it's complete (saw </｜DSML｜invoke>)
                    # This ensures each function returns at most once
                    calls_for_this_invoke: list[ToolCallItem] = []

                    # Check if invoke_content is empty or whitespace only
                    # If so, skip this tool call entirely (it's likely incomplete or malformed)
                    if not invoke_content.strip():
                        # Remove the incomplete tool call from buffer
                        self._buffer = current_text[invoke_match.end() :]
                        current_text = self._buffer
                        continue

                    # Send tool name
                    calls_for_this_invoke.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    # Send parameters as complete JSON
                    # Always send parameters, even if empty, to maintain consistency
                    calls_for_this_invoke.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=current_args_json,
                        )
                    )

                    # Ensure arrays are large enough for current tool
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    # Update the stored arguments
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": current_params,
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        current_args_json
                    )

                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer  # Update for next iteration

                    # Add calls for this invoke to all_calls
                    all_calls.extend(calls_for_this_invoke)

                    # Move to next tool call
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False

                    # Don't pre-allocate arrays for the next tool
                    # Only allocate when we actually complete a tool call
                    # This prevents _check_for_unstreamed_tool_args from sending incomplete calls

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
            begin=f'<{self.dsml_token}invoke name="' + name + '">',
            end=f"</{self.dsml_token}invoke>",
            trigger=f'<{self.dsml_token}invoke name="' + name + '">',
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token=self.bot_token,
            sequence_end_token=self.eot_token,
            tool_call_separator="",
            call_rule_fmt='"<｜DSML｜invoke name="{name}">{arguments_rule}</｜DSML｜invoke>"',
            function_format="json",
        )
