import ast
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
    _GetInfoFunc,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.ebnf_composer import (
    EBNFComposer,
)


def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    properties = (tool.function.parameters or {}).get("properties", {})
    if not isinstance(properties, dict) or arg_key not in properties:
        return None
    return properties[arg_key].get("type", None)


def _convert_to_number(value: str):
    """Convert string to int or float."""
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except (ValueError, AttributeError):
        return value


def parse_arguments(value, arg_type=None):
    """Parse argument value with multiple fallback strategies. Always try to parse as JSON first.

    Args:
        value: Raw string value to parse
        arg_type: Expected type hint ('string', 'number', 'object', etc.)

    Returns:
        Tuple of (parsed_value, is_valid_json)
    """
    # Strategy 1: Direct JSON parsing
    try:
        parsed_value = json.loads(value)
        # Type coercion for number type
        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        return parsed_value, True
    except Exception:
        logging.warn("parse_arguments strategy 1 failed for: %s", value)

    # Strategy 2: Unescape and parse
    try:
        wrapped = json.loads('{"tmp": "' + value + '"}')
        parsed_value = json.loads(wrapped["tmp"])
        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        return parsed_value, True
    except Exception:
        logging.warn("parse_arguments strategy 2 failed for: %s", value)

    # Strategy 3: ast.literal_eval
    try:
        parsed_value = ast.literal_eval(value)
        return parsed_value, True
    except Exception:
        logging.warn("parse_arguments strategy 3 failed for: %s", value)

    return value, False


class Glm4MoeDetector(BaseFormatDetector):
    """
    Detector for GLM-4.5, GLM-4.6, and GLM-4.7 models.

    Supported formats:

    1. GLM-4.5/4.6 with newline separator:
       <tool_call>get_weather
       <arg_key>city</arg_key>
       <arg_value>北京</arg_value>
       </tool_call>

    2. GLM-4.5/4.6 with literal \\n separator:
       <tool_call>get_weather\\n<arg_key>city</arg_key>\\n<arg_value>北京</arg_value>\\n</tool_call>

    3. GLM-4.7 with no separator:
       <tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>

    4. GLM-4.7 no-argument call:
       <tool_call>get_time</tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a glm-4.5 / glm-4.6 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = self.func_call_regex.findall(text)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = self.func_detail_regex.search(match_result)
                if func_detail is None:
                    continue
                func_name = (
                    func_detail.group(1).strip().rstrip("\\n")
                    if func_detail.group(1)
                    else ""
                )
                func_args = func_detail.group(2) if func_detail.group(2) else ""
                pairs = self.func_arg_regex.findall(func_args)
                arguments = {}
                for arg_key, arg_value in pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()
                    arg_type = get_argument_type(func_name, arg_key, tools)
                    parsed_value, is_good_json = parse_arguments(arg_value, arg_type)

                    if arg_type == "string":
                        # Ensure string type even if parsed as dict/list
                        if isinstance(parsed_value, str):
                            arguments[arg_key] = parsed_value
                        elif isinstance(parsed_value, (dict, list)):
                            arguments[arg_key] = json.dumps(
                                parsed_value, ensure_ascii=False
                            )
                        else:
                            arguments[arg_key] = str(parsed_value)
                    elif arg_type is None:
                        # If type is not defined, keep parsed value as-is
                        arguments[arg_key] = parsed_value if is_good_json else arg_value
                    else:
                        # For other types (number, object, array, etc.)
                        arguments[arg_key] = parsed_value if is_good_json else arg_value
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": arguments}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for GLM-4.5 and GLM-4.6 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        start = current_text.find(self.bot_token)
        if start == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                current_text = ""
            return StreamingParseResult(normal_text=current_text)
        # find ensures we find the first self.eot_token so there will be at most one tool_call in current_text[:end+len(self.eot_token)
        end = current_text.find(self.eot_token)
        if end != -1:
            # Initialize state if this is the first tool call
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
            # Ensure we have enough entries in our tracking arrays
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            result = self.detect_and_parse(
                current_text[: end + len(self.eot_token)], tools=tools
            )
            if result.calls:
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": result.calls[0].name,
                    "arguments": json.loads(result.calls[0].parameters),
                }
                self.streamed_args_for_tool[self.current_tool_id] = result.calls[
                    0
                ].parameters
                result.calls[0].tool_index = self.current_tool_id
                self.current_tool_id += 1
            self._buffer = current_text[end + len(self.eot_token) :]
            return result
        normal_text = current_text[:start]
        self._buffer = current_text[start:]
        return StreamingParseResult(normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.bot_token,
            individual_call_end_token=self.eot_token,
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"{name}" "\\n" ( {arguments_rule} "\\n" )?',
            key_value_rule_fmt='"<arg_key>{key}</arg_key>" "\\n" "<arg_value>" {valrule} "</arg_value>"',
            key_value_separator='"\\n"',
        )
