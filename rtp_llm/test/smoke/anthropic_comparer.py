import copy
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException
from smoke.utils import create_temporary_copy

from rtp_llm.anthropic.api_datatype import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicErrorResponse,
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicOutputTextBlock,
    AnthropicOutputThinkingBlock,
    AnthropicOutputToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicUsage,
)


def _normalize_anthropic_message(msg: AnthropicMessage) -> AnthropicMessage:
    normalized = msg.model_copy(deep=True)
    normalized.id = "msg_smoke"
    tool_idx = 0
    for block in normalized.content:
        if isinstance(block, AnthropicOutputToolUseBlock):
            block.id = f"toolu_smoke_{tool_idx}"
            tool_idx += 1
    return normalized


def _parse_anthropic_sse_lines(curl_response: Any) -> List[Tuple[str, Dict[str, Any]]]:
    if isinstance(curl_response, str):
        raw_lines = curl_response.splitlines()
    else:
        raw_lines = [
            line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
            for line in curl_response
        ]

    events: List[Tuple[str, Dict[str, Any]]] = []
    current_event: Optional[str] = None
    data_lines: List[str] = []

    def flush_event() -> None:
        nonlocal current_event, data_lines
        if current_event is None and not data_lines:
            return
        if not current_event or not data_lines:
            raise ValueError(
                f"Malformed Anthropic SSE event: event={current_event!r}, data_lines={data_lines!r}"
            )
        payload = json.loads("\n".join(data_lines))
        events.append((current_event, payload))
        current_event = None
        data_lines = []

    for line in raw_lines:
        stripped = line.rstrip("\r")
        if stripped == "":
            flush_event()
            continue
        if stripped.startswith(":"):
            continue
        if stripped.startswith("event:"):
            if current_event is not None:
                raise ValueError(
                    f"Repeated 'event:' line before blank separator: prev={current_event!r}, line={stripped!r}"
                )
            current_event = stripped[len("event:") :].strip()
        elif stripped.startswith("data:"):
            data_lines.append(stripped[len("data:") :].strip())
        else:
            raise ValueError(
                f"Unexpected non-SSE line in Anthropic stream: {stripped!r}"
            )
    flush_event()
    return events


def _require_delta_str(delta: Dict[str, Any], key: str, delta_type: str) -> str:
    if key not in delta or not isinstance(delta[key], str):
        raise ValueError(
            f"{delta_type} missing required string field {key!r}: {delta!r}"
        )
    return delta[key]


def reconstruct_anthropic_stream(
    events: List[Tuple[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    if not events:
        raise ValueError("Anthropic SSE stream returned no events")

    if events[0][0] != "message_start":
        raise ValueError(
            f"Expected first SSE event 'message_start', got {events[0][0]!r}"
        )

    start_data = events[0][1]
    if start_data.get("type") != "message_start" or not isinstance(
        start_data.get("message"), dict
    ):
        raise ValueError(f"Invalid message_start payload: {start_data}")

    msg_skeleton = copy.deepcopy(start_data["message"])
    if msg_skeleton.get("type") != "message" or msg_skeleton.get("role") != "assistant":
        raise ValueError(f"Invalid message_start skeleton: {msg_skeleton}")
    if msg_skeleton.get("content") != []:
        raise ValueError(
            f"Expected empty initial content in message_start, got {msg_skeleton.get('content')}"
        )
    start_usage = msg_skeleton.get("usage")
    if isinstance(start_usage, dict) and "output_tokens" in start_usage:
        start_out = start_usage["output_tokens"]
        if isinstance(start_out, bool) or start_out != 0:
            raise ValueError(
                f"Expected message_start output_tokens == 0, got {start_out!r}"
            )

    blocks: List[Dict[str, Any]] = []
    partial_json_buffers: Dict[int, List[str]] = {}
    open_blocks: set[int] = set()
    closed_blocks: set[int] = set()
    saw_message_delta = False
    saw_message_stop = False

    for event_name, payload in events[1:]:
        if event_name == "ping":
            continue
        if event_name == "error":
            raise SmokeException(
                QueryStatus.VISIT_FAILED,
                f"Anthropic SSE stream emitted error event: {payload}",
            )
        if saw_message_stop:
            raise ValueError(f"Unexpected event {event_name!r} after message_stop")

        if event_name == "content_block_start":
            idx = payload.get("index")
            block = copy.deepcopy(payload.get("content_block"))
            if not isinstance(idx, int) or isinstance(idx, bool) or idx != len(blocks):
                raise ValueError(
                    f"Non-monotonic content_block_start index {idx!r} (expected {len(blocks)})"
                )
            if not isinstance(block, dict) or "type" not in block:
                raise ValueError(f"Invalid content_block_start payload: {payload}")
            blocks.append(block)
            open_blocks.add(idx)
            if block["type"] == "tool_use":
                partial_json_buffers[idx] = []

        elif event_name == "content_block_delta":
            idx = payload.get("index")
            delta = payload.get("delta")
            if idx not in open_blocks:
                raise ValueError(f"content_block_delta for non-open index {idx!r}")
            if not isinstance(delta, dict):
                raise ValueError(f"Invalid content_block_delta.delta: {payload!r}")
            block = blocks[idx]
            block_type = block["type"]
            delta_type = delta.get("type")
            if block_type == "thinking":
                if delta_type == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + _require_delta_str(
                        delta, "thinking", "thinking_delta"
                    )
                elif delta_type == "signature_delta":
                    block["signature"] = block.get(
                        "signature", ""
                    ) + _require_delta_str(delta, "signature", "signature_delta")
                else:
                    raise ValueError(
                        f"Unexpected delta_type {delta_type!r} for thinking block"
                    )
            elif block_type == "text":
                if delta_type != "text_delta":
                    raise ValueError(
                        f"Unexpected delta_type {delta_type!r} for text block"
                    )
                block["text"] = block.get("text", "") + _require_delta_str(
                    delta, "text", "text_delta"
                )
            elif block_type == "tool_use":
                if delta_type != "input_json_delta":
                    raise ValueError(
                        f"Unexpected delta_type {delta_type!r} for tool_use block"
                    )
                partial_json_buffers[idx].append(
                    _require_delta_str(delta, "partial_json", "input_json_delta")
                )
            else:
                raise ValueError(f"Unsupported block type in stream: {block_type!r}")

        elif event_name == "content_block_stop":
            idx = payload.get("index")
            if idx not in open_blocks or idx in closed_blocks:
                raise ValueError(
                    f"content_block_stop for unopen/already-closed index {idx!r}"
                )
            open_blocks.remove(idx)
            closed_blocks.add(idx)
            if blocks[idx]["type"] == "tool_use":
                joined = "".join(partial_json_buffers.get(idx, []))
                blocks[idx]["input"] = json.loads(joined) if joined.strip() else {}

        elif event_name == "message_delta":
            if saw_message_delta:
                raise ValueError(
                    "Duplicate message_delta event in Anthropic SSE stream"
                )
            if open_blocks:
                raise ValueError(
                    f"message_delta received with unclosed blocks: {sorted(open_blocks)}"
                )
            saw_message_delta = True
            delta = payload.get("delta") or {}
            msg_skeleton["stop_reason"] = delta.get("stop_reason")
            msg_skeleton["stop_sequence"] = delta.get("stop_sequence")
            delta_usage = payload.get("usage")
            if not isinstance(delta_usage, dict) or "output_tokens" not in delta_usage:
                raise ValueError(
                    f"message_delta missing required usage.output_tokens: {payload!r}"
                )
            out_tokens = delta_usage["output_tokens"]
            if (
                not isinstance(out_tokens, int)
                or isinstance(out_tokens, bool)
                or out_tokens < 0
            ):
                raise ValueError(
                    f"message_delta.usage.output_tokens must be non-negative int, got {out_tokens!r}"
                )
            usage = msg_skeleton.setdefault("usage", {})
            usage["output_tokens"] = out_tokens

        elif event_name == "message_stop":
            if not saw_message_delta:
                raise ValueError("message_stop received before message_delta")
            saw_message_stop = True

        else:
            raise ValueError(f"Unknown Anthropic SSE event: {event_name!r}")

    if open_blocks:
        raise ValueError(
            f"Stream ended with unclosed content blocks: {sorted(open_blocks)}"
        )
    if not saw_message_delta or not saw_message_stop:
        raise ValueError(
            f"Incomplete Anthropic SSE stream (message_delta={saw_message_delta}, message_stop={saw_message_stop})"
        )

    msg_skeleton["content"] = blocks
    return msg_skeleton


class AnthropicComparer(BaseComparer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        endpoint_path = self.request_endpoint.split("?", 1)[0].rstrip("/")
        self.is_count_tokens = endpoint_path.endswith("/count_tokens")
        expected_status = int(self.qr_info.get("expected_status_code", 200))
        self.is_stream = (
            bool(self.qr_info["query"].get("stream", False))
            and not self.is_count_tokens
            and expected_status < 400
        )

    def _dump_result_for_save(self, actual_result: BaseModel) -> Dict[str, Any]:
        if isinstance(actual_result, AnthropicMessage):
            return actual_result.to_wire_dict()
        return actual_result.model_dump()

    def _rewrite_query_info(self, query_info: BaseModel) -> None:
        preserve_path = self.qr_info.get("compare_config", {}).get(
            "preserve_media_path", False
        )
        messages = getattr(query_info, "messages", None) or []
        for message in messages:
            if not isinstance(message.content, list):
                continue
            for block in message.content:
                if (
                    isinstance(block, AnthropicImageBlock)
                    and block.source.type == "url"
                    and block.source.url
                ):
                    block.source.url = create_temporary_copy(
                        block.source.url, preserve_path=preserve_path
                    )
                elif isinstance(block, AnthropicToolResultBlock) and isinstance(
                    block.content, list
                ):
                    for part in block.content:
                        if (
                            isinstance(part, AnthropicImageBlock)
                            and part.source.type == "url"
                            and part.source.url
                        ):
                            part.source.url = create_temporary_copy(
                                part.source.url, preserve_path=preserve_path
                            )

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        if self.is_count_tokens:
            query_info = AnthropicCountTokensRequest(**query_json)
        else:
            query_info = AnthropicMessagesRequest(**query_json)
        self._rewrite_query_info(query_info)
        return query_info

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        if self.is_count_tokens:
            return AnthropicCountTokensResponse(**result_json)
        expected_status = int(self.qr_info.get("expected_status_code", 200))
        if expected_status >= 400 or result_json.get("type") == "error":
            return AnthropicErrorResponse(**result_json)
        compare_config = self.qr_info.get("compare_config", {})
        skip_content = bool(
            compare_config.get("skip_content", False)
            or compare_config.get("skip_choices", False)
        )
        if not result_json and skip_content:
            return AnthropicMessage(
                id="msg_smoke",
                model="",
                content=[],
                stop_reason=None,
                usage=AnthropicUsage(),
            )
        return AnthropicMessage(**result_json)

    def curl_response_to_json(
        self, query_info: BaseModel, curl_response: Any
    ) -> Dict[str, Any]:
        if self.is_count_tokens:
            return json.loads(curl_response)
        if self.is_stream:
            events = _parse_anthropic_sse_lines(curl_response)
            reconstructed = reconstruct_anthropic_stream(events)
            validated = AnthropicMessage(**reconstructed)
            return _normalize_anthropic_message(validated).model_dump()
        parsed = json.loads(curl_response)
        if parsed.get("type") == "error":
            return parsed
        validated = AnthropicMessage(**parsed)
        return _normalize_anthropic_message(validated).model_dump()

    def _dump_value(self, value: Any) -> str:
        if isinstance(value, BaseModel):
            value = value.model_dump()
        return json.dumps(value, ensure_ascii=False, indent=2)

    def _format_diff(self, title: str, expect: Any, actual: Any) -> str:
        lines = [f"{title}:", "", "  expect:"]
        lines.extend("    " + line for line in self._dump_value(expect).split("\n"))
        lines.append("")
        lines.append("  actual:")
        lines.extend("    " + line for line in self._dump_value(actual).split("\n"))
        return "\n".join(lines)

    def _extract_combined_text(self, msg: AnthropicMessage) -> str:
        parts: List[str] = []
        for block in msg.content:
            if isinstance(block, AnthropicOutputTextBlock):
                parts.append(block.text)
            elif isinstance(block, AnthropicOutputThinkingBlock):
                parts.append(block.thinking)
        return "\n".join(parts)

    @staticmethod
    def _has_non_empty_content(msg: AnthropicMessage) -> bool:
        for block in msg.content:
            if isinstance(block, AnthropicOutputTextBlock) and block.text.strip():
                return True
            if (
                isinstance(block, AnthropicOutputThinkingBlock)
                and block.thinking.strip()
            ):
                return True
            if isinstance(block, AnthropicOutputToolUseBlock) and block.name:
                return True
        return False

    def compare_result(self, expect_result: Any, actual_result: Any) -> None:
        compare_config = self.qr_info.get("compare_config", {})
        diffs: List[str] = []

        if type(expect_result) != type(actual_result):
            diffs.append(
                self._format_diff(
                    "type mismatch",
                    type(expect_result).__name__,
                    type(actual_result).__name__,
                )
            )
            raise SmokeException(QueryStatus.COMPARE_FAILED, "\n".join(diffs))

        if isinstance(
            actual_result, (AnthropicCountTokensResponse, AnthropicErrorResponse)
        ):
            if expect_result != actual_result:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    self._format_diff(
                        "response mismatch", expect_result, actual_result
                    ),
                )
            return

        assert isinstance(expect_result, AnthropicMessage)
        assert isinstance(actual_result, AnthropicMessage)
        expect_norm = _normalize_anthropic_message(expect_result)
        actual_norm = _normalize_anthropic_message(actual_result)

        skip_content = bool(
            compare_config.get("skip_content", False)
            or compare_config.get("skip_choices", False)
        )
        skip_usage = bool(compare_config.get("skip_usage", False))

        if expect_norm.model and actual_norm.model != expect_norm.model:
            diffs.append(
                self._format_diff(
                    "model mismatch", expect_norm.model, actual_norm.model
                )
            )

        if not skip_usage and expect_norm.usage != actual_norm.usage:
            diffs.append(
                self._format_diff(
                    "usage mismatch", expect_norm.usage, actual_norm.usage
                )
            )

        expected_stop_reason = compare_config.get("expected_stop_reason")
        if (
            expected_stop_reason is not None
            and actual_norm.stop_reason != expected_stop_reason
        ):
            diffs.append(
                self._format_diff(
                    "stop_reason mismatch",
                    expected_stop_reason,
                    actual_norm.stop_reason,
                )
            )
        elif not skip_content and expect_norm.stop_reason != actual_norm.stop_reason:
            diffs.append(
                self._format_diff(
                    "stop_reason mismatch",
                    expect_norm.stop_reason,
                    actual_norm.stop_reason,
                )
            )

        if "expected_stop_sequence" in compare_config:
            expected_stop_sequence = compare_config["expected_stop_sequence"]
            if actual_norm.stop_sequence != expected_stop_sequence:
                diffs.append(
                    self._format_diff(
                        "stop_sequence mismatch",
                        expected_stop_sequence,
                        actual_norm.stop_sequence,
                    )
                )
        elif (
            not skip_content and expect_norm.stop_sequence != actual_norm.stop_sequence
        ):
            diffs.append(
                self._format_diff(
                    "stop_sequence mismatch",
                    expect_norm.stop_sequence,
                    actual_norm.stop_sequence,
                )
            )

        if not skip_content:
            expect_dump = [b.model_dump() for b in expect_norm.content]
            actual_dump = [b.model_dump() for b in actual_norm.content]
            if expect_dump != actual_dump:
                diffs.append(
                    self._format_diff(
                        "content blocks mismatch", expect_dump, actual_dump
                    )
                )
        elif not compare_config.get("allow_empty_content", False):
            if not self._has_non_empty_content(actual_norm):
                diffs.append(
                    self._format_diff(
                        "empty content when skip_content enabled",
                        "non-empty content block",
                        [b.model_dump() for b in actual_norm.content],
                    )
                )

        combined_text = self._extract_combined_text(actual_norm)
        for required in compare_config.get("required_substrings", []):
            if required not in combined_text:
                diffs.append(
                    self._format_diff(
                        "missing required substring", required, combined_text
                    )
                )
        for forbidden in compare_config.get("forbidden_substrings", []):
            if forbidden and forbidden in combined_text:
                diffs.append(
                    self._format_diff("present forbidden substring", "none", forbidden)
                )

        if (
            compare_config.get("require_json_object_content")
            or "expected_json" in compare_config
        ):
            text_only = "".join(
                b.text
                for b in actual_norm.content
                if isinstance(b, AnthropicOutputTextBlock)
            ).strip()
            try:
                parsed = json.loads(text_only)
            except Exception as exc:
                diffs.append(
                    self._format_diff("invalid JSON content", str(exc), text_only)
                )
            else:
                if not isinstance(parsed, dict):
                    diffs.append(
                        self._format_diff(
                            "expected JSON object", "dict", type(parsed).__name__
                        )
                    )
                elif (
                    "expected_json" in compare_config
                    and parsed != compare_config["expected_json"]
                ):
                    diffs.append(
                        self._format_diff(
                            "expected_json mismatch",
                            compare_config["expected_json"],
                            parsed,
                        )
                    )

        if diffs:
            raise SmokeException(QueryStatus.COMPARE_FAILED, "\n\n".join(diffs))
