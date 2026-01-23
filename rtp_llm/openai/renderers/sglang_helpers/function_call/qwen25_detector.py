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
    _GetInfoFunc,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.ebnf_composer import (
    EBNFComposer,
)

logger = logging.getLogger(__name__)


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>\n"
        self.eot_token = "\n</tool_call>"
        self.tool_call_separator = "\n"
        self._normal_text_buffer = ""  # Buffer for handling partial end tokens

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx] if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <tool_call>\n...\n</tool_call> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            try:
                parsed_call = json.loads(match_result.strip())
                calls.extend(self.parse_base_json(parsed_call, tools))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}"
                )
                continue

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Qwen 2.5 tool calls.

        MTP-safe: First checks for complete <tool_call>...</tool_call> blocks in buffer.
        Falls back to base class incremental parsing for partial data.

        NOTE: This MTP-safe path is a workaround, not a true incremental parser.
        The root cause is that the base class assumes bot_token at buffer start (for not MTP, yes).
        When prefix content exists (e.g., "</think>\n\n<tool_call>..."), the base
        class fails with MalformedJSON because it tries to parse from index 0.

        Current behavior: Buffer accumulates failed chunks until both bot_token
        and eot_token are present, then parses everything at once (batch parsing).

        A cleaner fix would be to modify base class to:
        1. Find bot_token anywhere in buffer (not just at start)
        2. Emit text before bot_token as normal_text immediately
        3. Continue true incremental parsing from bot_token onward

        This workaround was added to avoid modifying base class behavior that
        affects all detectors. Future refactoring should consider the base class fix.
        Currently, this workaround only applies to Qwen25Detector, DeepSeekV31Detector and KimiK2Detector.
        """
        self._buffer += new_text

        collected_calls: list = []
        collected_normal_text = ""

        # MTP-safe path: Parse any complete tool call blocks first
        # This handles MTP scenarios where multiple tokens arrive in single chunk
        while self.bot_token in self._buffer and self.eot_token in self._buffer:
            bot_idx = self._buffer.find(self.bot_token)
            eot_idx = self._buffer.find(self.eot_token)

            # Only process if we have a complete block (eot comes after bot)
            if eot_idx <= bot_idx:
                break

            # Extract text before tool call as normal text
            if bot_idx > 0:
                collected_normal_text += self._buffer[:bot_idx]

            # Extract and parse the complete tool call block
            block_end = eot_idx + len(self.eot_token)
            complete_block = self._buffer[bot_idx:block_end]

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

            # Parse the complete block using detect_and_parse
            result = self.detect_and_parse(complete_block, tools)
            if result.calls:
                for call in result.calls:
                    call.tool_index = self.current_tool_id
                    # Store tool call info for serving layer
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": call.name,
                        "arguments": (
                            json.loads(call.parameters) if call.parameters else {}
                        ),
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        call.parameters or ""
                    )
                    collected_calls.append(call)
                    self.current_tool_id += 1
                    # Extend arrays for next potential tool call
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

            # Remove processed block from buffer
            self._buffer = self._buffer[block_end:]

        # If we parsed any complete blocks, return those results
        if collected_calls or collected_normal_text:
            # Reset buffer for base class if we're switching to incremental mode
            remaining = self._buffer
            self._buffer = ""
            # If there's remaining content that might be partial, handle with base class
            if remaining:
                self._buffer = remaining
            return StreamingParseResult(
                normal_text=collected_normal_text, calls=collected_calls
            )

        # Fall back to base class incremental parsing for partial data
        # Reset buffer since we're passing to base class which will re-accumulate
        remaining = self._buffer
        self._buffer = ""
        result = super().parse_streaming_increment(remaining, tools)

        # Handle partial end tokens that are streamed character by character
        if result.normal_text:
            self._normal_text_buffer += result.normal_text

            # Check if buffer contains complete end token (without leading newline)
            end_token_without_newline = self.eot_token[1:]  # "</tool_call>"
            if end_token_without_newline in self._normal_text_buffer:
                cleaned_text = self._normal_text_buffer.replace(
                    end_token_without_newline, ""
                )
                self._normal_text_buffer = ""
                result.normal_text = cleaned_text
            else:
                # Check if buffer might contain partial end token at the end
                partial_match_len = self._ends_with_partial_token(
                    self._normal_text_buffer, end_token_without_newline
                )

                if partial_match_len:
                    # Keep potential partial match in buffer, return the rest
                    result.normal_text = self._normal_text_buffer[:-partial_match_len]
                    self._normal_text_buffer = self._normal_text_buffer[
                        -partial_match_len:
                    ]
                else:
                    # No partial match, return all buffered text
                    result.normal_text = self._normal_text_buffer
                    self._normal_text_buffer = ""

        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n{"name":"' + name + '", "arguments":',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.bot_token.replace("\n", "\\n"),
            individual_call_end_token=self.eot_token.replace("\n", "\\n"),
            tool_call_separator="\\n",
            function_format="json",
        )
