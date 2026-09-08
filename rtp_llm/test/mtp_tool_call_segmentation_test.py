"""
Unit tests for MTP (multi-token-per-step) tool-call argument segmentation.

These tests run on CPU only (no model service, no GPU) and reproduce a bug where
tool-call arguments (e.g. a "todo" list) get wrongly segmented when a multi-byte
UTF-8 character spans a chunk boundary.

Background
----------
With speculative decoding (MTP), multiple tokens arrive in a single engine step.
``reasoning_tool_base_renderer._update_single_status`` decomposes them into
single-token text deltas via ``TokenNormalizer`` and feeds each delta to the
detector's streaming parser.

The bug has two parts:

1. ``TokenNormalizer._calculate_yielded_length`` only handled a *trailing*
   ``\uFFFD``, not a *leading* one.  When a multi-byte character (e.g. the
   space-merged Qwen encoding of ``" 薯"`` = 3 tokens) spans a chunk boundary,
   the character's *tail* bytes appear at the *start* of ``prev_token_id``
   without their *head* bytes, so it returned 0 and the normalizer re-processed
   already-emitted text.

2. The renderer kept only ``last_token_length = len(new_token_ids)`` tokens of
   context.  When a character's head bytes live in the *previous* step, that
   window is too short, leaving ``prev_token_id`` as pure ``\uFFFD`` that even a
   leading-\uFFFD-aware ``_calculate_yielded_length`` cannot resolve.

Both are in the shared normalizer/renderer path, so they affect every detector
(DeepSeek V4, Qwen25, DeepSeek V3.1, GLM-4, Kimi-K2, ...), not just DeepSeek V4.

Fix: ``_calculate_yielded_length`` now only treats a *trailing* ``\uFFFD`` as
incomplete, and the renderer keeps ``min(len(output_ids), _MAX_UTF8_WINDOW)``
tokens of context so a multi-byte character's head bytes are always available.
"""

import json
import unittest

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv4_detector import (
    DeepSeekV4Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.token_normalizer import (
    _MAX_UTF8_WINDOW,
    TokenNormalizer,
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF


class SpaceMergeByteTokenizer:
    """Byte-level tokenizer mimicking Qwen's tiktoken space-merge behavior.

    - ASCII characters (including space, quotes, brackets) are single tokens.
    - The DSML delimiter ``\uff5c`` (U+FF5C) is a single token (it is in the
      Qwen vocabulary), so DSML tags are never split.
    - A CJK character (e.g. Chinese) is split into its UTF-8 bytes.
    - A space immediately before a CJK character is merged into the first
      byte-token of that character (``b" " + b"\xe8"`` for ``" 薯"``).
    - Individual byte-tokens decode to ``\uFFFD`` (incomplete UTF-8).
    """

    def __init__(self):
        self._next_id = 1
        self._id_to_bytes = {}
        self._bytes_to_id = {}

    def _get_id(self, raw: bytes) -> int:
        if raw not in self._bytes_to_id:
            self._bytes_to_id[raw] = self._next_id
            self._id_to_bytes[self._next_id] = raw
            self._next_id += 1
        return self._bytes_to_id[raw]

    def encode(self, text: str):
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == "\uff5c":
                # DSML delimiter: keep as a single token so tags stay intact.
                tokens.append(self._get_id(ch.encode("utf-8")))
                i += 1
            elif ord(ch) < 128:
                if ch == " " and i + 1 < n and _is_cjk(text[i + 1]):
                    nxt = text[i + 1].encode("utf-8")
                    tokens.append(self._get_id(b" " + nxt[:1]))
                    for b in nxt[1:]:
                        tokens.append(self._get_id(bytes([b])))
                    i += 2
                else:
                    tokens.append(self._get_id(ch.encode("utf-8")))
                    i += 1
            elif _is_cjk(ch):
                for b in ch.encode("utf-8"):
                    tokens.append(self._get_id(bytes([b])))
                i += 1
            else:
                tokens.append(self._get_id(ch.encode("utf-8")))
                i += 1
        return tokens

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        data = b"".join(self._id_to_bytes.get(t, b"") for t in token_ids)
        return data.decode("utf-8", errors="replace")


DSML = "\uff5cDSML\uff5c"


def _todo_tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="todo_write",
            description="write todos",
            parameters={
                "type": "object",
                "properties": {"todos": {"type": "string"}},
                "required": ["todos"],
            },
        ),
    )


def _dsml_tool_call_text(todos) -> str:
    return (
        f"<{DSML}tool_calls>\n"
        f'<{DSML}invoke name="todo_write">\n'
        f'<{DSML}parameter name="todos" string="true">'
        + json.dumps(todos, ensure_ascii=False)
        + f"</{DSML}parameter>\n"
        f"</{DSML}invoke>\n"
        f"</{DSML}tool_calls>"
    )


def _collect_args(calls) -> dict:
    args = {}
    for c in calls:
        if c.parameters:
            args[c.tool_index] = args.get(c.tool_index, "") + c.parameters
    return args


def _ground_truth(detector, text, tools) -> dict:
    return _collect_args(detector.detect_and_parse(text, tools).calls)


def _simulate_renderer(detector, tokenizer, tokens, tools, chunk_size) -> dict:
    """Simulate reasoning_tool_base_renderer._update_single_status exactly.

    Mirrors the real state machine: ``output_ids`` is the full accumulated
    output, ``last_output_ids`` only advances when the normalizer yielded text,
    and ``new_token_ids = output_ids[len(last_output_ids):]`` therefore
    accumulates any tokens that were skipped (incomplete UTF-8) in the previous
    step.
    """
    output_ids = []
    last_output_ids = []
    last_token_length = 0
    calls = []

    i = 0
    while i < len(tokens):
        delta = tokens[i : i + chunk_size]
        i += chunk_size
        output_ids = output_ids + delta

        new_token_ids = output_ids[len(last_output_ids) :]

        normalizer = TokenNormalizer(tokenizer)
        prev_token_id = (
            last_output_ids[-last_token_length:] if last_token_length else []
        )

        yielded = False
        for delta in normalizer.normalize_tokens(prev_token_id, new_token_ids):
            yielded = True
            result = detector.parse_streaming_increment(delta, tools)
            calls.extend(result.calls)

        if yielded and new_token_ids:
            last_token_length = min(len(output_ids), _MAX_UTF8_WINDOW)
            last_output_ids = output_ids

    return _collect_args(calls)


class TestMtpToolCallSegmentation(unittest.TestCase):
    """Reproduce the MTP tool-call argument segmentation bug (CPU-only)."""

    def setUp(self):
        self.tokenizer = SpaceMergeByteTokenizer()
        self.tools = [_todo_tool()]
        # A todo whose content contains space-separated Chinese words, which the
        # Qwen-style tokenizer splits across multiple tokens with a merged space.
        self.todos = [
            {"content": "检查环境 可乐 薯片 饼干", "status": "进行中"},
            {"content": "启动服务", "status": "待处理"},
        ]
        self.text = _dsml_tool_call_text(self.todos)
        self.tokens = self.tokenizer.encode(self.text)

    def _run(self, detector_cls, chunk_size):
        detector = detector_cls()
        return _simulate_renderer(
            detector, self.tokenizer, self.tokens, self.tools, chunk_size
        )

    def test_deepseek_v4_todo_roundtrip_ground_truth(self):
        """The full-text (non-streaming) parse is correct; this is our baseline."""
        detector = DeepSeekV4Detector()
        args = _ground_truth(detector, self.text, self.tools)
        self.assertEqual(
            json.loads(args[0])["todos"],
            json.dumps(self.todos, ensure_ascii=False),
        )

    def test_deepseek_v4_todo_single_token_streaming_preserves_content(self):
        """Single-token streaming must reconstruct the todo exactly."""
        args = self._run(DeepSeekV4Detector, 1)
        self.assertEqual(
            json.loads(args[0])["todos"],
            json.dumps(self.todos, ensure_ascii=False),
            "single-token streaming corrupted the todo: %r" % args,
        )

    def test_deepseek_v4_todo_mtp_streaming_preserves_content(self):
        """MTP (multi-token per step) must reconstruct the todo exactly."""
        for chunk_size in (2, 3, 4, 5, 6, 8, 16):
            with self.subTest(chunk_size=chunk_size):
                args = self._run(DeepSeekV4Detector, chunk_size)
                self.assertEqual(
                    json.loads(args[0])["todos"],
                    json.dumps(self.todos, ensure_ascii=False),
                    "MTP chunk_size=%d corrupted the todo: %r" % (chunk_size, args),
                )


class TestNormalizerSharedPath(unittest.TestCase):
    """The corruption happens in the shared TokenNormalizer + renderer state
    machine, *before* any detector runs.

    ``reasoning_tool_base_renderer._update_single_status`` is inherited by every
    reasoning/tool renderer (deepseek_v4, deepseek_v31, deepseek_v32, kimik2,
    chatglm45, qwen3_code, qwen_reasoning_tool), so a single fix there repairs
    all detectors at once.  This test reproduces the corruption at the
    normalizer level with no detector involved.
    """

    def test_normalizer_roundtrip_corrupted_regardless_of_detector(self):
        tokenizer = SpaceMergeByteTokenizer()
        todos = [
            {"content": "检查环境 可乐 薯片 饼干", "status": "进行中"},
        ]
        text = _dsml_tool_call_text(todos)
        tokens = tokenizer.encode(text)

        output_ids = []
        last_output_ids = []
        last_token_length = 0
        deltas = []
        i = 0
        while i < len(tokens):
            new = tokens[i : i + 4]  # MTP chunk of 4
            i += 4
            output_ids = output_ids + new
            new_token_ids = output_ids[len(last_output_ids) :]
            normalizer = TokenNormalizer(tokenizer)
            prev = last_output_ids[-last_token_length:] if last_token_length else []
            yielded = False
            for d in normalizer.normalize_tokens(prev, new_token_ids):
                yielded = True
                deltas.append(d)
            if yielded and new_token_ids:
                last_token_length = min(len(output_ids), _MAX_UTF8_WINDOW)
                last_output_ids = output_ids

        self.assertEqual(
            "".join(deltas),
            text,
            "shared normalizer path corrupted the text: %r" % "".join(deltas),
        )


class TestCalculateYieldedLengthLeadingReplacement(unittest.TestCase):
    """The root cause: _calculate_yielded_length mishandles a leading \uFFFD.

    When the previous chunk ends with the tail byte of a multi-byte character
    that was already resolved (yielded) together with earlier bytes, the next
    ``prev_token_id`` starts with that tail byte.  Its decode has a *leading*
    ``\uFFFD``.  The fixed implementation must treat that leading ``\uFFFD`` as
    already-yielded text (it is the placeholder for the tail byte) and return
    the full decoded length, rather than 0.
    """

    def test_leading_replacement_is_counted_as_yielded(self):
        tokenizer = SpaceMergeByteTokenizer()
        normalizer = TokenNormalizer(tokenizer)

        # " 薯" (space + 薯) is encoded as [space+0xE8, 0x96, 0xAF]; "乐" as 3 bytes.
        shu = tokenizer.encode(" 薯")
        le = tokenizer.encode("乐")
        self.assertEqual(tokenizer.decode(shu), " 薯")
        self.assertEqual(tokenizer.decode(le), "乐")

        # prev_token_id = [tail byte of 薯, ...乐 bytes] — the tail byte of 薯 was
        # already yielded as part of " 薯" in the previous chunk.  Decoding this
        # window yields "\uFFFD乐" (the leading \uFFFD is the orphaned tail byte).
        prev = [shu[-1]] + le
        self.assertEqual(tokenizer.decode(prev), "\uFFFD乐")

        yielded = normalizer._calculate_yielded_length(prev)
        # The whole "\uFFFD乐" (2 chars) is already yielded; the leading \uFFFD
        # must not force the result back to 0.
        self.assertEqual(
            yielded,
            2,
            "leading \\uFFFD made _calculate_yielded_length return %d "
            "instead of 2" % yielded,
        )


if __name__ == "__main__":
    unittest.main()
