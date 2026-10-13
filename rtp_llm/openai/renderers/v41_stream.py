"""Incremental V4.1 recipe protocol parser, implemented entirely in Python.

The state/action mapping follows deepseek-recipe 8cadfede, stream/state_machine.rs
and processor.rs. The accompanying v41_recipe.LICENSE contains the MIT notice.
Input is decoded Unicode; the renderer retains RTP's token/UTF-8 normalizer.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolDelta:
    kind: str
    text: str = ""
    index: int = -1


class V41StreamParser:
    def __init__(
        self,
        *,
        thinking=False,
        parse_tools=True,
        force_tools=False,
        json_output=False,
        stop=(),
    ):
        self.parse_tools = parse_tools
        self.json_output = json_output
        self.stop = tuple(value for value in stop if value)
        self.stage = "tools" if force_tools else "reasoning" if thinking else "content"
        self.leading = True
        self.pending = ""
        self.tool_name = ""
        self.tool_index = -1
        self.stopped = False
        self.closed = False
        self.last_space = False
        self._configure()

    def _configure(self):
        self.branches = []
        self.action = "skip"

        def add(pattern, stage, action="skip", newlines=False):
            self.branches.append((pattern, stage, action, newlines))

        def stops():
            for value in self.stop:
                add(value, "finished", "stop")

        def tools():
            if self.parse_tools:
                add("<\uff5cDSML\uff5c", "tools", newlines=True)

        if self.stage == "content":
            if self.json_output:
                add("```json\n", "json")
                add("```\n", "json")
                add("{", "json", "label:{")
                add("[", "json", "label:[")
                self.action = "space"
            else:
                stops()
                self.action = "content"
            tools()
            add("</think>", "content_nonleading")
        elif self.stage == "json":
            stops()
            add("\n```", "json_done")
            self.action = "content"
        elif self.stage == "json_done":
            tools()
            self.action = "space"
        elif self.stage == "reasoning":
            tools()
            add("</think>", "content", newlines=True)
            self.action = "reasoning"
        elif self.stage == "tools":
            add('invoke name="', "name", "begin")
            add("</\uff5cDSML\uff5c calls>", "finished")
            add("</\uff5cDSML\uff5ctool_calls>", "finished")
        elif self.stage == "name":
            add('"', "arguments_first", "name_end")
            self.action = "name"
        elif self.stage in ("arguments_first", "arguments"):
            first = self.stage == "arguments_first"
            add("parameter name=", "parameter_name", "arg:{" if first else "arg:, ")
            add("</\uff5cDSML\uff5c", "tools", "arg:{}" if first else "arg:}")
        elif self.stage == "parameter_name":
            add(" ", "parameter_type", "arg:: ")
            self.action = "argument"
        elif self.stage == "parameter_type":
            add('true">', "string_value", 'arg:"')
            add('false">', "value")
        elif self.stage in ("string_value", "value"):
            string = self.stage == "string_value"
            add("</\uff5cDSML\uff5c", "arguments", 'arg:"' if string else "skip")
            self.action = "string" if string else "argument"

    def _transition(self, stage):
        self.leading = stage in ("content", "reasoning")
        self.stage = "content" if stage == "content_nonleading" else stage
        self._configure()

    def _emit(self, action, text, result):
        if not text:
            return
        if action == "stop":
            self.stopped = True
        elif action == "begin":
            self.tool_name = ""
        elif action == "name":
            self.tool_name += text
        elif action == "name_end":
            self.tool_index += 1
            result.append(ProtocolDelta("tool", self.tool_name, self.tool_index))
        elif action.startswith("arg:"):
            result.append(ProtocolDelta("arguments", action[4:], self.tool_index))
        elif action in ("argument", "string"):
            if action == "string":
                text = json.dumps(text, ensure_ascii=False)[1:-1]
            result.append(ProtocolDelta("arguments", text, self.tool_index))
        elif action.startswith("label:"):
            result.append(ProtocolDelta("content", action[6:]))
        elif action == "space":
            # JSON-mode surrounding prose is whitespace, independent of source
            # chunking. It never participates in user stop matching.
            if not self.last_space:
                result.append(ProtocolDelta("content", " "))
        elif action in ("content", "reasoning"):
            result.append(ProtocolDelta(action, text))
        if action != "skip":
            self.last_space = action == "space"

    def _retained_length(self):
        keep = 0
        for pattern, _, _, newlines in self.branches:
            length = min(len(pattern) - 1, len(self.pending))
            while length and not self.pending.endswith(pattern[:length]):
                length -= 1
            if newlines:
                end = len(self.pending) - length
                start = end
                while start and self.pending[start - 1] == "\n":
                    start -= 1
                length += end - start
            keep = max(keep, length)
        return keep

    def feed(self, text: str) -> list[ProtocolDelta]:
        if self.closed:
            raise ValueError("cannot feed a finished V4.1 parser")
        result = []
        for char in text:
            if self.stage == "finished":
                break
            if self.leading and self.stage in ("content", "reasoning"):
                if char == "\n":
                    continue
                self.leading = False
            self.pending += char
            for pattern, stage, action, newlines in self.branches:
                if not self.pending.endswith(pattern):
                    continue
                start = len(self.pending) - len(pattern)
                if newlines:
                    while start and self.pending[start - 1] == "\n":
                        start -= 1
                self._emit(self.action, self.pending[:start], result)
                self._emit(action, self.pending[start:], result)
                self.pending = ""
                self._transition(stage)
                break
            else:
                keep = self._retained_length()
                emit = len(self.pending) - keep
                self._emit(self.action, self.pending[:emit], result)
                self.pending = self.pending[emit:]
        return self._merge(result)

    def finish(self) -> list[ProtocolDelta]:
        if self.closed:
            return []
        self.closed = True
        result = []
        self._emit(self.action, self.pending, result)
        self.pending = ""
        return self._merge(result)

    def finish_reason(self, backend_reason=None):
        if self.stopped:
            return "stop"
        if backend_reason == "stop" and self.tool_index >= 0:
            return "tool_calls"
        return backend_reason or "stop"

    @staticmethod
    def _merge(events):
        merged = []
        for event in events:
            if (
                merged
                and event.kind != "tool"
                and (merged[-1].kind, merged[-1].index) == (event.kind, event.index)
            ):
                previous = merged.pop()
                event = ProtocolDelta(
                    event.kind, previous.text + event.text, event.index
                )
            merged.append(event)
        return merged
