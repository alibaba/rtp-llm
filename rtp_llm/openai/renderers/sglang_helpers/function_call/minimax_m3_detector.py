import json
import logging
import re
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Tuple

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

# Every tag in MiniMax-M3's tool call syntax is prefixed by this namespace token
# (a single token in the tokenizer's added vocab). Keying tag boundaries on
# NS_TOKEN + "<" instead of a bare "<" keeps literal angle brackets inside
# parameter values from being mistaken for markup.
NS_TOKEN = "]<]minimax[>["
TAG_OPEN = NS_TOKEN + "<"

TOOL_CALL_BEGIN = NS_TOKEN + "<tool_call>"
TOOL_CALL_END = NS_TOKEN + "</tool_call>"
INVOKE_OPEN = TAG_OPEN + "invoke"
INVOKE_CLOSE = TAG_OPEN + "/invoke>"

_INVOKE_NAME_RE = re.compile(r"""name\s*=\s*(?:"([^"]*)"|'([^']*)')""")

_STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})
_INT_PREFIXES = ("int", "uint", "long", "short", "unsigned")
_FLOAT_PREFIXES = ("num", "float", "double", "decimal")
_BOOL_TYPES = frozenset({"boolean", "bool", "binary"})


class _Node:
    """A parsed XML element: either a leaf carrying text, or a container with children."""

    __slots__ = ("text", "children")

    def __init__(self) -> None:
        self.text: str = ""
        self.children: List[Tuple[str, "_Node"]] = []


def _resolve_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """Collapse anyOf/oneOf/allOf wrappers down to the first informative subschema."""
    if not isinstance(schema, dict):
        return None
    if any(k in schema for k in ("type", "properties", "items", "enum")):
        return schema
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key) or []:
            resolved = _resolve_schema(sub)
            if resolved is not None:
                return resolved
    return None


_PY_TYPE_NAMES = {bool: "boolean", int: "integer", float: "number", str: "string"}


def _schema_type(schema: Optional[Dict[str, Any]]) -> Optional[str]:
    if schema is None:
        return None
    raw = schema.get("type")
    if isinstance(raw, list):
        raw = next((t for t in raw if t != "null"), None)
    if isinstance(raw, str):
        return raw.strip().lower()
    # A bare `enum` carries no `type`, so infer it from the allowed values.
    values = [v for v in schema.get("enum") or [] if v is not None]
    names = {_PY_TYPE_NAMES.get(type(v)) for v in values}
    if len(names) == 1:
        return names.pop()
    return None


def _coerce_scalar(text: str, schema: Optional[Dict[str, Any]], path: str) -> Any:
    if text.strip().lower() == "null":
        return None

    schema_type = _schema_type(schema)
    if schema_type is None or schema_type in _STRING_TYPES:
        return text

    stripped = text.strip()

    if schema_type.startswith(_INT_PREFIXES):
        try:
            return int(stripped)
        except ValueError:
            logger.warning(
                f"MiniMax-M3 tool call: value {text!r} at {path} is not an integer, keeping string"
            )
            return text

    if schema_type.startswith(_FLOAT_PREFIXES):
        try:
            value = float(stripped)
        except ValueError:
            logger.warning(
                f"MiniMax-M3 tool call: value {text!r} at {path} is not a number, keeping string"
            )
            return text
        looks_integral = "." not in stripped and "e" not in stripped.lower()
        return int(value) if looks_integral and value.is_integer() else value

    if schema_type in _BOOL_TYPES:
        lowered = stripped.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        logger.warning(
            f"MiniMax-M3 tool call: value {text!r} at {path} is not a boolean, coercing to false"
        )
        return False

    if schema_type in ("object", "array"):
        # The model is expected to expand these into nested XML, but it sometimes
        # emits raw JSON text instead.
        try:
            return json.loads(stripped)
        except ValueError:
            logger.warning(
                f"MiniMax-M3 tool call: value {text!r} at {path} is not valid JSON "
                f"for declared type {schema_type}, keeping string"
            )
            return text

    return text


def _strip_one_newline(text: str) -> str:
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text


def _node_to_value(node: _Node, schema: Any, path: str) -> Any:
    resolved = _resolve_schema(schema)
    schema_type = _schema_type(resolved)

    if node.children:
        tags = [tag for tag, _ in node.children]
        # `<item>` wrappers mark array elements. Trust the schema first, since an
        # object may legitimately own a property literally named "item".
        is_array = schema_type == "array" or (
            schema_type is None and all(tag == "item" for tag in tags)
        )
        if is_array:
            item_schema = (resolved or {}).get("items")
            return [
                _node_to_value(child, item_schema, f"{path}[{i}]")
                for i, (_, child) in enumerate(node.children)
            ]

        properties = (resolved or {}).get("properties") or {}
        return {
            tag: _node_to_value(child, properties.get(tag), f"{path}.{tag}")
            for tag, child in node.children
        }

    text = _strip_one_newline(node.text)
    if not text:
        if schema_type == "array":
            return []
        if schema_type == "object":
            return {}
    return _coerce_scalar(text, resolved, path)


def _tokenize(segment: str) -> List[Tuple[str, str]]:
    """Split a tag body into ("open"|"close"|"text", value) events."""
    events: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(segment):
        tag_start = segment.find(TAG_OPEN, pos)
        if tag_start == -1:
            events.append(("text", segment[pos:]))
            break
        if tag_start > pos:
            events.append(("text", segment[pos:tag_start]))
        body_start = tag_start + len(TAG_OPEN)
        tag_end = segment.find(">", body_start)
        if tag_end == -1:
            break
        body = segment[body_start:tag_end].strip()
        if body.startswith("/"):
            events.append(("close", body[1:].strip()))
        else:
            events.append(("open", body))
        pos = tag_end + 1
    return events


def _build_tree(events: List[Tuple[str, str]]) -> _Node:
    root = _Node()
    stack = [root]
    names: List[Optional[str]] = [None]
    for kind, value in events:
        if kind == "text":
            stack[-1].text += value
        elif kind == "open":
            child = _Node()
            stack[-1].children.append((value, child))
            stack.append(child)
            names.append(value)
        else:
            # Unwind to the matching open tag. A close tag that matches nothing on
            # the stack is dropped rather than popping an unrelated level.
            for depth in range(len(stack) - 1, 0, -1):
                if names[depth] == value:
                    del stack[depth:]
                    del names[depth:]
                    break
    return root


def _repair_misnesting(
    node: _Node, schema: Any, ancestor_properties: FrozenSet[str] = frozenset()
) -> List[Tuple[str, _Node]]:
    """
    Recover from a dropped closing tag, which otherwise nests every following
    sibling inside the unclosed element.

    Returns the children that cannot legally live under `node`, for the caller to
    re-adopt. Two signals identify a mis-nested child:
      - an array holds anything other than `<item>`;
      - an object holds a key absent from its own schema but declared by an
        enclosing object.
    Both are checked against the schema, so an object with undeclared properties
    keeps its unknown keys instead of having them hoisted away.
    """
    resolved = _resolve_schema(schema)

    if _schema_type(resolved) == "array":
        item_schema = (resolved or {}).get("items")
        kept: List[Tuple[str, _Node]] = []
        tail: List[Tuple[str, _Node]] = []
        for tag, child in node.children:
            if tail or tag != "item":
                tail.append((tag, child))
            else:
                kept.append((tag, child))
        node.children = kept
        spilled: List[Tuple[str, _Node]] = []
        for _, child in kept:
            spilled.extend(_repair_misnesting(child, item_schema, ancestor_properties))
        spilled.extend(tail)
        return spilled

    properties = (resolved or {}).get("properties") or {}
    known = frozenset(properties)
    inherited = ancestor_properties | known
    orphans: List[Tuple[str, _Node]] = []
    index = 0
    while index < len(node.children):
        tag, child = node.children[index]
        if properties and tag not in known and tag in ancestor_properties:
            orphans.append(node.children.pop(index))
            continue
        spilled = _repair_misnesting(child, properties.get(tag), inherited)
        # Adopt what this schema declares; let the rest keep bubbling up. Adopted
        # children are spliced in document order and revisited by the loop, so
        # each is repaired against its own schema.
        adopted = [entry for entry in spilled if entry[0] in known]
        orphans.extend(entry for entry in spilled if entry[0] not in known)
        if adopted:
            node.children[index + 1 : index + 1] = adopted
        index += 1
    return orphans


def _parse_invoke_name(header: str) -> Optional[str]:
    match = _INVOKE_NAME_RE.search(header)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip() or None


def _parameters_schema(
    name: str, tools: Optional[List[Tool]]
) -> Optional[Dict[str, Any]]:
    for tool in tools or []:
        function = getattr(tool, "function", None)
        if function is not None and getattr(function, "name", None) == name:
            parameters = getattr(function, "parameters", None)
            return parameters if isinstance(parameters, dict) else None
    return None


def _iter_invokes(text: str) -> Iterator[Tuple[Optional[str], str]]:
    pos = 0
    while True:
        start = text.find(INVOKE_OPEN, pos)
        if start == -1:
            return
        header_end = text.find(">", start + len(TAG_OPEN))
        if header_end == -1:
            return
        body_end = text.find(INVOKE_CLOSE, header_end)
        if body_end == -1:
            return
        yield (
            _parse_invoke_name(text[start + len(TAG_OPEN) : header_end]),
            text[header_end + 1 : body_end],
        )
        pos = body_end + len(INVOKE_CLOSE)


class MiniMaxM3Detector(BaseFormatDetector):
    """
    Detector for MiniMax-M3's XML tool call format.

    Format Structure:
    ```
    NS<tool_call>
    NS<invoke name="func">NS<param>value NS</param>NS</invoke>
    NS</tool_call>
    ```
    where NS is the literal namespace token `]<]minimax[>[`. Values are untyped
    text, so the tool's JSON Schema drives type recovery; nested objects and
    arrays are recursively expanded (arrays via `NS<item>` wrappers).
    """

    def __init__(self) -> None:
        super().__init__()
        self.bot_token = TOOL_CALL_BEGIN
        self.eot_token = TOOL_CALL_END
        self._in_tool_call = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _held_back_prefix_len(self, buffer: str) -> int:
        """
        Length of the longest buffer suffix that could still grow into bot_token.

        The base class helper returns the *shortest* such suffix, which corrupts
        this format: bot_token is `]<]minimax[>[<tool_call>`, whose leading `]`
        recurs at index 2, so a buffer ending in `]<]` matches at length 1 and the
        `]<` would be flushed as normal text, leaving the token unrecognizable.
        """
        for length in range(min(len(buffer), len(self.bot_token) - 1), 0, -1):
            if self.bot_token.startswith(buffer[-length:]):
                return length
        return 0

    def _decode_arguments(
        self, name: str, body: str, tools: Optional[List[Tool]]
    ) -> Dict[str, Any]:
        schema = _parameters_schema(name, tools)
        root = _build_tree(_tokenize(body))
        root.children.extend(_repair_misnesting(root, schema))
        value = _node_to_value(root, schema, name)
        if isinstance(value, dict):
            return value
        logger.warning(
            f"MiniMax-M3 tool call: arguments for {name!r} decoded to "
            f"{type(value).__name__}, expected object"
        )
        return {}

    def _emit_call(
        self,
        name: Optional[str],
        body: str,
        tools: Optional[List[Tool]],
        start_index: int,
    ) -> List[ToolCallItem]:
        if not name:
            logger.warning("MiniMax-M3 tool call: <invoke> without a name, skipping")
            return []
        action = {"name": name, "parameters": self._decode_arguments(name, body, tools)}
        # parse_base_json handles unknown-tool filtering and JSON serialization.
        return self.parse_base_json(action, tools or [], start_index=start_index)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text)

        calls: List[ToolCallItem] = []
        for name, body in _iter_invokes(text):
            calls.extend(self._emit_call(name, body, tools, len(calls)))

        return StreamingParseResult(
            normal_text=text[: text.find(self.bot_token)], calls=calls
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Emits one ToolCallItem per complete `<invoke>` block, carrying the whole
        arguments JSON at once. Incremental JSON diffing of a partially received
        nested tree buys nothing here, and downstream
        `streaming_parse_result_to_tool_calls` concatenates arguments per
        tool_index anyway.
        """
        self._buffer += new_text
        calls: List[ToolCallItem] = []
        normal_chunks: List[str] = []

        while True:
            if not self._in_tool_call:
                begin = self._buffer.find(self.bot_token)
                if begin == -1:
                    keep = self._held_back_prefix_len(self._buffer)
                    if len(self._buffer) > keep:
                        flush_to = len(self._buffer) - keep
                        normal_chunks.append(self._buffer[:flush_to])
                        self._buffer = self._buffer[flush_to:]
                    break
                if begin > 0:
                    normal_chunks.append(self._buffer[:begin])
                self._buffer = self._buffer[begin + len(self.bot_token) :]
                self._in_tool_call = True
                continue

            if self._consume_one_invoke(tools, calls):
                continue

            end = self._buffer.find(self.eot_token)
            if end != -1 and INVOKE_OPEN not in self._buffer[:end]:
                self._buffer = self._buffer[end + len(self.eot_token) :]
                self._in_tool_call = False
                continue

            break

        return StreamingParseResult(
            normal_text="".join(normal_chunks), calls=calls
        )

    def _consume_one_invoke(
        self, tools: List[Tool], calls: List[ToolCallItem]
    ) -> bool:
        start = self._buffer.find(INVOKE_OPEN)
        if start == -1:
            return False
        header_end = self._buffer.find(">", start + len(TAG_OPEN))
        if header_end == -1:
            return False
        body_end = self._buffer.find(INVOKE_CLOSE, header_end)
        if body_end == -1:
            return False

        name = _parse_invoke_name(self._buffer[start + len(TAG_OPEN) : header_end])
        body = self._buffer[header_end + 1 : body_end]
        self._buffer = self._buffer[body_end + len(INVOKE_CLOSE) :]

        emitted = self._emit_call(name, body, tools, self.current_tool_id + 1)
        if emitted:
            self.current_tool_id += len(emitted)
            calls.extend(emitted)
        return True

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
