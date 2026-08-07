"""Tool-call detector for MiniMax-M2.

Format (from `MiniMax-M2.7/docs/tool_calling_guide.md`):
```
<minimax:tool_call>
<invoke name="search_web">
<parameter name="query_tag">["news"]</parameter>
<parameter name="query_list">["latest"]</parameter>
</invoke>
<invoke name="another_func">
...
</invoke>
</minimax:tool_call>
```

Multiple `<invoke>` blocks may appear inside a single `<minimax:tool_call>`.
Each `<parameter>` value is a string; if the tool's JSON schema declares a
non-string type (object/array/number/boolean/integer), we attempt to parse
it as JSON and fall back to the raw string on failure.

Streaming model (per OpenAI delta protocol — see kimik2 / glm47_moe for
peers): tool-call deltas are emitted in true incremental fashion.

  1. As soon as `<invoke name="X">` is seen, emit a `ToolCallItem(name=X,
     parameters="")` so the client can render "calling X…" before any
     argument has arrived.
  2. Each closed `<parameter name="K">V</parameter>` emits a delta whose
     `parameters` is the next JSON-string fragment of the running args
     object (`'{"K": <coerced(V)>'` for the first param, `', "K": ...'`
     for each subsequent).
  3. `</invoke>` emits the closing `'}'`.

Concatenating all `parameters` deltas for a given `tool_index` yields the
full args JSON object. Empty `<invoke name="X"></invoke>` emits `'{}'`.
Granularity is per-parameter, not per-character — string params are not
streamed mid-value because we need the full value to apply schema-driven
type coercion.
"""

from __future__ import annotations

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
    ToolCallItem,
    _GetInfoFunc,
)

_TOOL_BLOCK_RE = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
# Match either single- or double-quoted name="..." / name='...' in one regex
# so that `re.finditer` walks invokes/parameters in their original textual
# order (mixed-quote inputs would otherwise be split into two passes and
# concatenated, scrambling order).
_INVOKE_RE = re.compile(
    r"""<invoke\s+name\s*=\s*(?P<q>["'])(?P<name>[^"']+)(?P=q)\s*>(?P<body>.*?)</invoke>""",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"""<parameter\s+name\s*=\s*(?P<q>["'])(?P<name>[^"']+)(?P=q)\s*>(?P<value>.*?)</parameter>""",
    re.DOTALL,
)
# Open-tag-only variants used by the streaming state machine to detect
# block boundaries before the corresponding close-tag arrives.
_INVOKE_OPEN_RE = re.compile(
    r"""<invoke\s+name\s*=\s*(?P<q>["'])(?P<name>[^"']+)(?P=q)\s*>""",
    re.DOTALL,
)
_PARAM_OPEN_RE = re.compile(
    r"""<parameter\s+name\s*=\s*(?P<q>["'])(?P<name>[^"']+)(?P=q)\s*>""",
    re.DOTALL,
)
_INVOKE_CLOSE_TOKEN = "</invoke>"
_PARAM_CLOSE_TOKEN = "</parameter>"

# Streaming-state enum — a tiny constants-on-the-class pattern keeps the
# state-machine tractable without a real Enum dependency.
_OUT, _IN_TC, _IN_INVOKE, _IN_PARAM = 0, 1, 2, 3


def _coerce_param(raw: str, param_type: str | None) -> Any:
    """Convert a raw string parameter value to its declared JSON-schema type.

    Falls back to the raw string if conversion fails or the schema is unknown.
    Mirrors the reference parser in MiniMax's official tool_calling_guide.
    """
    if param_type in (None, "string"):
        return raw
    try:
        if param_type in ("object", "array"):
            return json.loads(raw)
        if param_type == "boolean":
            val = raw.strip().lower()
            if val in ("true", "1", "yes"):
                return True
            if val in ("false", "0", "no"):
                return False
            return raw
        if param_type == "integer":
            return int(raw.strip())
        if param_type in ("number", "float"):
            return float(raw.strip())
        # Unknown declared type — best-effort JSON, fall back to string.
        return json.loads(raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        return raw


def _params_for_tool(func_name: str, tools: List[Tool]) -> Dict[str, str]:
    """Return {param_name: declared JSON-schema type} for the named tool."""
    for tool in tools or []:
        fn = tool.function
        if not fn or fn.name != func_name:
            continue
        params = fn.parameters or {}
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        if not isinstance(props, dict):
            return {}
        out: Dict[str, str] = {}
        for k, v in props.items():
            if isinstance(v, dict) and isinstance(v.get("type"), str):
                out[k] = v["type"]
        return out
    return {}


def _suffix_prefix_len(buf: str, token: str) -> int:
    """Return the length of the longest suffix of `buf` that is a *strict*
    prefix of `token`. Used by streaming to decide how many chars to keep
    in the buffer when bot_token hasn't fully arrived yet.

    Example: buf='abc<minimax:tool_', token='<minimax:tool_call>' -> 14
    (the '<minimax:tool_' suffix matches the same prefix of token).
    """
    max_n = min(len(buf), len(token) - 1)
    for n in range(max_n, 0, -1):
        if token.startswith(buf[-n:]):
            return n
    return 0


def _strip_one_newline(s: str) -> str:
    """The model emits parameter values surrounded by literal newlines for
    readability. Strip exactly one leading/trailing newline (HF reference
    parser does the same — preserves intentional internal whitespace)."""
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


def _parse_invokes(block: str) -> List[Dict[str, Any]]:
    """Extract `<invoke name="...">..</invoke>` items from a single tool_call
    block. Returns a list of {name, raw_params: [(key, raw_value), ...]}."""
    items: List[Dict[str, Any]] = []
    for invoke_m in _INVOKE_RE.finditer(block):
        fn_name = invoke_m.group("name")
        body = invoke_m.group("body")
        pairs: List[tuple[str, str]] = [
            (param_m.group("name"), _strip_one_newline(param_m.group("value")))
            for param_m in _PARAM_RE.finditer(body)
        ]
        items.append({"name": fn_name, "raw_params": pairs})
    return items


class MinimaxM2Detector(BaseFormatDetector):
    """One-shot + true-streaming tool-call parser for MiniMax-M2."""

    def __init__(self):
        super().__init__()
        self.bot_token = "<minimax:tool_call>"
        self.eot_token = "</minimax:tool_call>"
        # Streaming state-machine bookkeeping. See the per-state handler in
        # `parse_streaming_increment` for the transition diagram.
        self._stream_state: int = _OUT
        self._stream_invoke_name: str = ""
        # Whether we've already emitted the leading '{' for the current invoke's
        # arguments JSON; controls whether the next param fragment starts with
        # '{' or ', '.
        self._stream_invoke_open_emitted: bool = False
        self._stream_param_name: str = ""
        self._tool_indices: Dict[str, int] = {}

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    # ---- one-shot ----
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx] if idx != -1 else text
        if idx == -1:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            for block in _TOOL_BLOCK_RE.findall(text):
                for invoke in _parse_invokes(block):
                    fn_name = invoke["name"]
                    type_map = _params_for_tool(fn_name, tools)
                    args: Dict[str, Any] = {}
                    for k, raw_v in invoke["raw_params"]:
                        args[k] = _coerce_param(raw_v, type_map.get(k))
                    match_result = {"name": fn_name, "parameters": args}
                    calls.extend(
                        self.parse_base_json(
                            match_result, tools, start_index=len(calls)
                        )
                    )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logging.error("[MinimaxM2Detector] parse failure: %s", e)
            # Return only the pre-bot_token segment; never leak the raw
            # <minimax:tool_call> XML to the user on a parse error.
            return StreamingParseResult(normal_text=normal_text, calls=[])

    # ---- streaming (true incremental, state-machine) ----
    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Drive the streaming state machine forward by one chunk.

        States:
          _OUT      : outside any <minimax:tool_call> block.
          _IN_TC    : inside a tool_call block, between/before invokes.
          _IN_INVOKE: inside an <invoke>...</invoke>, between/before params.
          _IN_PARAM : inside a <parameter>...</parameter>, awaiting close.

        Each iteration of the drain loop either advances `self._buffer` past
        a fully-resolved token boundary or breaks to wait for the next chunk.
        Calls accumulate per `tool_index`; concatenating their `parameters`
        deltas yields the full arguments JSON object for that tool.
        """
        self._buffer += new_text
        if not self._tool_indices:
            self._tool_indices = self._get_tool_indices(tools)

        calls: List[ToolCallItem] = []
        normal_chunks: List[str] = []
        tool_call_started = self.current_tool_id != -1

        while True:
            if self._stream_state == _OUT:
                if not self._advance_outside(normal_chunks):
                    break
                tool_call_started = True
                continue

            if self._stream_state == _IN_TC:
                if not self._advance_in_tool_call(calls, tools):
                    break
                continue

            if self._stream_state == _IN_INVOKE:
                if not self._advance_in_invoke(calls):
                    break
                continue

            if self._stream_state == _IN_PARAM:
                if not self._advance_in_param(calls, tools):
                    break
                continue

            break  # unreachable

        # Suppress free-form normal_text once any tool_call has started — the
        # downstream framing assumes post-tool_call text is part of the
        # model's chain-of-tools, not user-visible content.
        normal_text = "" if tool_call_started else "".join(normal_chunks)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    # ---- streaming state handlers ----
    def _advance_outside(self, normal_chunks: List[str]) -> bool:
        """`_OUT` → consume safe normal_text or transition to `_IN_TC`.

        Returns True if the state machine progressed and the outer loop
        should continue, False to wait for more input.
        """
        i = self._buffer.find(self.bot_token)
        if i == -1:
            # Don't drop a partial bot_token suffix (e.g. "<minimax:tool_") —
            # could complete on the next chunk.
            keep_n = _suffix_prefix_len(self._buffer, self.bot_token)
            if keep_n:
                head = self._buffer[:-keep_n]
                if head:
                    normal_chunks.append(head)
                self._buffer = self._buffer[-keep_n:]
            else:
                if self._buffer:
                    normal_chunks.append(self._buffer)
                self._buffer = ""
            return False
        if i > 0:
            normal_chunks.append(self._buffer[:i])
        self._buffer = self._buffer[i + len(self.bot_token) :]
        self._stream_state = _IN_TC
        return True

    def _advance_in_tool_call(
        self, calls: List[ToolCallItem], tools: List[Tool]
    ) -> bool:
        """`_IN_TC` → either start a new invoke or close the tool_call block."""
        invoke_m = _INVOKE_OPEN_RE.search(self._buffer)
        eot_idx = self._buffer.find(self.eot_token)
        invoke_pos = invoke_m.start() if invoke_m else -1

        # Wait for more if neither is observable yet.
        if invoke_pos == -1 and eot_idx == -1:
            return False

        # Tool_call closes before any further invoke — drain it and exit.
        if eot_idx != -1 and (invoke_pos == -1 or eot_idx < invoke_pos):
            self._buffer = self._buffer[eot_idx + len(self.eot_token) :]
            self._stream_state = _OUT
            return True

        assert invoke_m is not None
        fname = invoke_m.group("name")
        if fname not in self._tool_indices:
            # Unknown tool: skip past the matching </invoke> if present so we
            # can resume parsing siblings; otherwise wait for it.
            close_idx = self._buffer.find(_INVOKE_CLOSE_TOKEN, invoke_m.end())
            if close_idx == -1:
                return False
            logging.warning("Unknown M2 tool name %r — dropping invoke", fname)
            self._buffer = self._buffer[close_idx + len(_INVOKE_CLOSE_TOKEN) :]
            return True

        # Allocate the next tool slot and emit name + empty args delta.
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": fname,
            "arguments": {},
        }
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, name=fname, parameters="")
        )
        self.current_tool_name_sent = True

        self._stream_invoke_name = fname
        self._stream_invoke_open_emitted = False
        self._buffer = self._buffer[invoke_m.end() :]
        self._stream_state = _IN_INVOKE
        return True

    def _advance_in_invoke(self, calls: List[ToolCallItem]) -> bool:
        """`_IN_INVOKE` → start a parameter or close the current invoke."""
        param_m = _PARAM_OPEN_RE.search(self._buffer)
        close_idx = self._buffer.find(_INVOKE_CLOSE_TOKEN)
        param_pos = param_m.start() if param_m else -1

        if param_pos == -1 and close_idx == -1:
            return False

        if close_idx != -1 and (param_pos == -1 or close_idx < param_pos):
            # Close brace — emit '{}' for empty invokes, '}' otherwise.
            fragment = ("{" if not self._stream_invoke_open_emitted else "") + "}"
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=fragment,
                )
            )
            self.streamed_args_for_tool[self.current_tool_id] += fragment
            self._buffer = self._buffer[close_idx + len(_INVOKE_CLOSE_TOKEN) :]
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self._stream_invoke_name = ""
            self._stream_invoke_open_emitted = False
            self._stream_state = _IN_TC
            return True

        assert param_m is not None
        self._stream_param_name = param_m.group("name")
        self._buffer = self._buffer[param_m.end() :]
        self._stream_state = _IN_PARAM
        return True

    def _advance_in_param(self, calls: List[ToolCallItem], tools: List[Tool]) -> bool:
        """`_IN_PARAM` → wait for `</parameter>`, then emit a JSON fragment."""
        close_idx = self._buffer.find(_PARAM_CLOSE_TOKEN)
        if close_idx == -1:
            return False

        raw_value = self._buffer[:close_idx]
        stripped = _strip_one_newline(raw_value)
        type_map = _params_for_tool(self._stream_invoke_name, tools)
        coerced = _coerce_param(stripped, type_map.get(self._stream_param_name))

        json_key = json.dumps(self._stream_param_name, ensure_ascii=False)
        json_value = json.dumps(coerced, ensure_ascii=False)
        sep = "{" if not self._stream_invoke_open_emitted else ", "
        fragment = f"{sep}{json_key}: {json_value}"

        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id, name=None, parameters=fragment
            )
        )
        self.streamed_args_for_tool[self.current_tool_id] += fragment
        # Maintain the running args dict for downstream serving consumers.
        args = self.prev_tool_call_arr[self.current_tool_id].setdefault("arguments", {})
        args[self._stream_param_name] = coerced

        self._stream_invoke_open_emitted = True
        self._stream_param_name = ""
        self._buffer = self._buffer[close_idx + len(_PARAM_CLOSE_TOKEN) :]
        self._stream_state = _IN_INVOKE
        return True

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:  # pragma: no cover
        raise NotImplementedError(
            "MiniMax-M2 detector does not yet expose a StructureInfo for "
            "constrained decoding."
        )
