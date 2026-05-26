import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import orjson
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import Tool
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.utils import (
    _find_common_prefix,
    _is_complete_json,
    _partial_json_loads,
)

logger = logging.getLogger(__name__)


def _forward_unknown_tools() -> bool:
    """Check if unknown tool calls should be forwarded instead of dropped."""
    return os.environ.get("RTP_LLM_FORWARD_UNKNOWN_TOOLS", "").lower() == "true"


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # When an unknown tool name is detected and forwarding is off,
        # enter discard mode to silently consume tokens until eot_token,
        # matching non-streaming behavior where the whole block is skipped.
        self._discarding_unknown_tool = False

        # Token configuration (override in subclasses)
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def parse_base_json(
        self, action: Any, tools: List[Tool], start_index: int = 0
    ) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for i, act in enumerate(action):
            name = act.get("name")
            if not (name and name in tool_indices):
                logger.warning(f"Model attempted to call undefined function: {name}")
                if not _forward_unknown_tools():
                    continue

            results.append(
                ToolCallItem(
                    tool_index=start_index + i,
                    name=name,
                    parameters=json.dumps(
                        act.get("parameters") or act.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                )
            )

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = orjson.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.

        This base implementation works best with formats where:
        1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "

        Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
        - Each tool call is wrapped in a separate block: See Qwen25Detector
        - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - Tool call is Pythonic style

        For incompatible formats, detectors should override this method with custom logic.
        """
        # Discard mode: consume tokens silently until end-of-tool token,
        # so the entire unknown tool call block is skipped (not emitted).
        if self._discarding_unknown_tool:
            self._buffer += new_text
            eot_pos = self._buffer.find(self.eot_token)
            if eot_pos != -1:
                remaining = self._buffer[eot_pos + len(self.eot_token) :]
                self._buffer = ""
                self._discarding_unknown_tool = False
                if remaining:
                    return self.parse_streaming_increment(remaining, tools)
                return StreamingParseResult()
            return StreamingParseResult()

        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # The current_text has tool_call if it is the start of a new tool call sequence
        # or it is the start of a new tool call after a tool call separator, when there is a previous tool call
        if not (
            self.has_tool_call(current_text)
            or (
                self.current_tool_id > 0
                and current_text.startswith(self.tool_call_separator)
            )
        ):
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial bot_token, keep buffering
                return StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                # Priority check: if we're processing a subsequent tool (current_tool_id > 0),
                # first check if text starts with the tool separator. This is critical for
                # parallel tool calls because the bot_token (e.g., '[') can also
                # appear inside array parameters of the current tool, and we must not
                # mistakenly identify that as the start of a new tool.
                if self.current_tool_id > 0 and current_text.startswith(
                    self.tool_call_separator
                ):
                    start_idx = len(self.tool_call_separator)
                else:
                    # Only search for bot_token if not processing subsequent tool
                    tool_call_pos = current_text.find(self.bot_token)
                    if tool_call_pos != -1:
                        start_idx = tool_call_pos + len(self.bot_token)
                    else:
                        start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                (obj, end_idx) = _partial_json_loads(current_text[start_idx:], flags)

                is_current_complete = _is_complete_json(
                    current_text[start_idx : start_idx + end_idx]
                )

                # Validate tool name if present
                if "name" in obj and obj["name"] not in self._tool_indices:
                    logger.warning(
                        f"Model attempted to call undefined function: {obj['name']}"
                    )
                    if not _forward_unknown_tools():
                        # Enter discard mode: keep buffering until eot_token
                        # so the entire tool call block is silently skipped,
                        # consistent with non-streaming parse_base_json.
                        self._discarding_unknown_tool = True
                        self.current_tool_id = -1
                        self.current_tool_name_sent = False
                        if self.streamed_args_for_tool:
                            self.streamed_args_for_tool.pop()
                        # Check if eot_token is already in the buffer
                        eot_pos = current_text.find(self.eot_token)
                        if eot_pos != -1:
                            remaining = current_text[eot_pos + len(self.eot_token) :]
                            self._buffer = ""
                            self._discarding_unknown_tool = False
                            if remaining:
                                return self.parse_streaming_increment(remaining, tools)
                            return StreamingParseResult()
                        return StreamingParseResult()

                # Handle parameters/arguments consistency
                # NOTE: we assume here that the obj is always partial of a single tool call
                if "parameters" in obj:
                    assert (
                        "arguments" not in obj
                    ), "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except MalformedJSON:
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            collected_calls: List[ToolCallItem] = []

            # Case 1: emit the tool name if we haven't yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and (
                    function_name in self._tool_indices or _forward_unknown_tools()
                ):
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    collected_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True

                    # Seed prev_tool_call_arr so subsequent diff has a baseline.
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

                    # Fall through to Case 2 only when the entire tool-call
                    # JSON arrived in this chunk; otherwise return now.
                    if not is_current_complete:
                        return StreamingParseResult(calls=collected_calls)
                else:
                    return StreamingParseResult()

            # Case 2: stream arguments.
            cur_arguments = current_tool_call.get("arguments")

            if cur_arguments is not None:
                # Calculate how much of the arguments we've already streamed
                sent = len(self.streamed_args_for_tool[self.current_tool_id])
                cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                prev_arguments = None
                if self.current_tool_id < len(self.prev_tool_call_arr):
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                argument_diff = None
                # Snapshot before the is_current_complete branch below may
                # increment self.current_tool_id.
                tool_index_to_use = self.current_tool_id

                # If the current tool's JSON is complete, send all remaining arguments
                if is_current_complete:
                    argument_diff = cur_args_json[sent:]
                    # Only remove the processed portion, keep unprocessed content
                    self._buffer = current_text[start_idx + end_idx :]

                # If the tool is still being parsed, send incremental changes
                elif prev_arguments is not None:
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    if cur_args_json != prev_args_json:
                        prefix = _find_common_prefix(prev_args_json, cur_args_json)
                        argument_diff = prefix[sent:]

                # Invariant: cur_args_json must start with what we've already
                # streamed. If it doesn't, we can't un-emit, and emitting any
                # suffix would produce malformed JSON on the client. Drop and
                # log; truncation is the lesser evil.
                streamed_so_far = self.streamed_args_for_tool[self.current_tool_id]
                if argument_diff is not None and not cur_args_json.startswith(
                    streamed_so_far
                ):
                    logger.error(
                        "tool_call stream diverges from current parse; "
                        "client will see truncated args. "
                        "streamed=%r cur=%r is_complete=%s",
                        streamed_so_far,
                        cur_args_json,
                        is_current_complete,
                    )
                    argument_diff = None

                # Update prev_tool_call_arr with current state
                if self.current_tool_id >= 0:
                    # Ensure prev_tool_call_arr is large enough
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

                # Advance to next tool if complete
                if is_current_complete:
                    self.current_tool_name_sent = False
                    self.current_tool_id += 1

                if argument_diff is not None:
                    collected_calls.append(
                        ToolCallItem(
                            tool_index=tool_index_to_use,
                            parameters=argument_diff,
                        )
                    )
                    self.streamed_args_for_tool[tool_index_to_use] += argument_diff

            return StreamingParseResult(calls=collected_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains function call markers specific to this format.
        """
        raise NotImplementedError()

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return True

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        """
        Return a function that creates StructureInfo for constrained generation.

        The returned function takes a tool name and returns a StructureInfo object
        containing the begin/end patterns and trigger tokens needed for constrained
        generation of function calls in this format.

        Returns:
            A function that takes a tool name (str) and returns StructureInfo
        """
        raise NotImplementedError()
