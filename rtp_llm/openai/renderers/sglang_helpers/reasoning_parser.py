import logging
import re
from typing import Dict, Optional, Tuple, Type


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning

        self._buffer = ""
        self.stripped_think_start = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return StreamingParseResult(
            normal_text=normal_text, reasoning_text=reasoning_text
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        logging.debug(
            f"[REASONING_DEBUG] parse_streaming_increment: buffer={repr(self._buffer)}, new_text={repr(new_text)}"
        )
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        logging.debug(
            f"[REASONING_DEBUG] parse_streaming_increment: current_text={repr(current_text)}, in_reasoning={self._in_reasoning}"
        )
        if any(
            token.startswith(current_text) and token != current_text
            for token in [self.think_start_token, self.think_end_token]
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Supported models:
      - DeepSeek-R1: Always generates thinking content without <think> start tag
      - DeepSeek-R1-0528: Generates thinking content with <think> start tag

    Format patterns:
      - DeepSeek-R1: "I need to think about this...</think>The answer is 42."
      - DeepSeek-R1-0528: "<think>I need to think about this...</think>The answer is 42."

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 models (e.g., Qwen/Qwen3-235B-A22B).
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Qwen3 models released before 07/2025 supports switching between thinking mode and normal
    mode using `enable_thinking` parameter in the request parameter.
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )


MM_THINK_START_TOKEN = "<mm:think>"
MM_THINK_END_TOKEN = "</mm:think>"

# Besides the `<mm:think>` / `</mm:think>` added tokens, M3 also emits the markers
# as plain tokens with zero-width characters spliced in (`</\u200bmm:think>`). That
# is a different token sequence, so a literal comparison never matches it and the
# marker leaks into the content. Worse, the chat template keys the "did this turn
# think?" check off the same literal, so a leaked marker fed back through the
# history makes the template prepend yet another marker every turn. Folding the
# escaped variants back to the literal form breaks that feedback loop.
_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"
_ZW = f"[{_ZERO_WIDTH_CHARS}]*"
_MM_THINK_TAG_RE = re.compile(
    "<" + _ZW + "(/?)" + _ZW + _ZW.join("mm:think") + _ZW + ">"
)
_MM_THINK_TOKENS = (MM_THINK_START_TOKEN, MM_THINK_END_TOKEN)


def normalize_mm_think_tags(text: str) -> str:
    """Fold zero-width-escaped M3 think markers back to their literal form."""
    if not text:
        return text
    normalized = _MM_THINK_TAG_RE.sub(lambda m: f"<{m.group(1)}mm:think>", text)
    return _unescape_trailing_partial_tag(normalized)


def _unescape_trailing_partial_tag(text: str) -> str:
    """Drop zero-width chars from a partial think marker at the end of the text.

    Streaming delivers the marker one token at a time, so the buffer routinely ends
    mid-marker (`</`, `</\u200b`, `</\u200bmm`, ...). The base detector only keeps
    buffering while the text is a literal prefix of a think token, so the zero-width
    chars have to go before it can recognise the partial marker.
    """
    idx = text.rfind("<")
    if idx == -1 or not any(c in text[idx:] for c in _ZERO_WIDTH_CHARS):
        return text
    head, tail = text[:idx], text[idx:]
    stripped = "".join(c for c in tail if c not in _ZERO_WIDTH_CHARS)
    if any(token.startswith(stripped) for token in _MM_THINK_TOKENS):
        return head + stripped
    return text


def strip_mm_think_tags(text: str) -> str:
    """Remove any complete think marker left in text destined for `content`."""
    if not text:
        return text
    for token in _MM_THINK_TOKENS:
        text = text.replace(token, "")
    return text


def _trailing_marker_prefix_len(text: str) -> int:
    """Length of the trailing run that could still grow into a think marker.

    A lone `<` counts: chunk boundaries fall wherever the token stream puts them, so
    a marker really can start at the very end of a chunk. This mirrors what the base
    class already does with its own buffer, and carries the same caveat — a response
    whose final character is `<` has it held back.
    """
    longest = max(len(token) for token in _MM_THINK_TOKENS) - 1
    for n in range(min(len(text), longest), 0, -1):
        if any(token.startswith(text[-n:]) for token in _MM_THINK_TOKENS):
            return n
    return 0


class MiniMaxM3ReasoningDetector(BaseReasoningFormatDetector):
    """
    Detector for MiniMax-M3 models.
    Assumes reasoning format:
      (<mm:think>)*(.*)</mm:think>

    M3 always closes with `</mm:think>`, even when it skips thinking: its system
    prompt instructs it to "begin your response directly after the </mm:think>
    prefix". Reasoning is therefore forced, as for DeepSeek-R1 — otherwise a
    non-thinking reply, which opens with a bare `</mm:think>`, leaks that tag
    into the content.

    M3 also repeats the closing marker, so the base class's single-shot split is
    not enough: once reasoning has ended every further marker would be treated as
    ordinary text. Markers are control tokens and never belong in `content`, so
    the leftovers are dropped instead.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        super().__init__(
            MM_THINK_START_TOKEN,
            MM_THINK_END_TOKEN,
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )
        self._marker_carry = ""

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        result = super().detect_and_parse(normalize_mm_think_tags(text))
        result.normal_text = strip_mm_think_tags(result.normal_text)
        return result

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        # Normalize the retained buffer together with the new chunk: a marker can be
        # split across chunks, so the escape only becomes visible once the pieces are
        # joined. Folding is idempotent, so re-running it on the buffer is safe.
        combined = normalize_mm_think_tags(self._buffer + new_text)
        self._buffer = ""
        result = super().parse_streaming_increment(combined)
        result.normal_text = self._drop_markers(result.normal_text)
        return result

    def _drop_markers(self, text: str) -> str:
        """Strip markers, carrying a trailing partial marker to the next chunk.

        The base class only keeps buffering while the *whole* buffer is a prefix of a
        marker, so a chunk ending in `answer</` fails that test and the marker gets
        emitted piecemeal — no single chunk ever contains a complete marker to strip.
        Multi-token steps (MTP) hit this routinely because one delta carries both text
        and the marker's opening.
        """
        if not text and not self._marker_carry:
            return text
        combined = strip_mm_think_tags(
            normalize_mm_think_tags(self._marker_carry + text)
        )
        keep = _trailing_marker_prefix_len(combined)
        self._marker_carry = combined[len(combined) - keep :] if keep else ""
        return combined[: len(combined) - keep]

    def flush_markers(self) -> str:
        """Release a held-back partial marker once no more tokens are coming.

        Without this, a reply whose last characters look like the start of a marker
        (`...<`, `...</`) would have them silently dropped.
        """
        carry, self._marker_carry = self._marker_carry, ""
        return carry


class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    DetectorMap: Dict[str, Type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "deepseek-v3": Qwen3Detector,
        "glm45": Qwen3Detector,
        "kimi": KimiDetector,
        "kimi_k2": Qwen3Detector,
        "minimax_m3": MiniMaxM3ReasoningDetector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "step3": DeepSeekR1Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type.lower() == "qwen3-thinking":
            force_reasoning = True

        self.detector = detector_class(
            stream_reasoning=stream_reasoning, force_reasoning=force_reasoning
        )

    def parse_non_stream(self, full_text: str) -> Tuple[str, str]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, str]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text

    def flush_markers(self) -> str:
        """Release any text a detector held back pending more tokens.

        Only detectors that buffer beyond the base class implement this.
        """
        flush = getattr(self.detector, "flush_markers", None)
        return flush() if flush else ""
