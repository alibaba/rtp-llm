"""DashSc thinking-mode planning helpers.

The dash-sc gRPC path receives pre-tokenized ``input_ids`` and bypasses the
OpenAI renderer, so request-time thinking decisions need to happen before the
backend ``GenerateInput`` is built.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

MAX_THINK_ANCHOR_SCAN_TOKENS = 8192
MAX_THINK_ANCHOR_DECODE_TOKENS = 4 * MAX_THINK_ANCHOR_SCAN_TOKENS


def match_think_anchor(
    input_ids: list[int],
    *,
    bos_tokens: Sequence[int],
    bare_bos_tokens: Sequence[int],
    anchor: str,
    decode: Optional[Callable],
    max_token_id: Optional[int] = None,
) -> tuple[bool, list[int]]:
    """Find a complete trailing tag, plus an independently removable token suffix.

    A tag merged with preceding prompt text still opens the thinking gate, but
    returns no suffix: echoing/removing that token would change the user prompt.
    Only trailing LF is ignored; spaces, partial tags and closed blocks do not match.
    Reject an indeterminate tail beyond the scan budget instead of silently
    changing the request's thinking mode or scanning an entire client payload.
    All decoder calls share a token-work budget, including suffix probes and
    positional retries, so invisible tokens cannot amplify a bounded tail scan.
    """
    for tokens in (bos_tokens, bare_bos_tokens):
        if tokens and input_ids[-len(tokens) :] == list(tokens):
            return True, list(tokens)
    anchor = anchor.rstrip("\n")
    if decode is None or not anchor:
        return False, []

    decoded_tokens = 0

    def bounded_decode(ids: list[int], **kwargs) -> str:
        nonlocal decoded_tokens
        decoded_tokens += len(ids)
        if decoded_tokens > MAX_THINK_ANCHOR_DECODE_TOKENS:
            raise ValueError(
                "thinking boundary cannot be determined within "
                f"{MAX_THINK_ANCHOR_DECODE_TOKENS} decoded input tokens"
            )
        return decode(ids, **kwargs)

    def decode_tail(ids: list[int]) -> str:
        if any(i < 0 or (max_token_id is not None and i > max_token_id) for i in ids):
            raise ValueError(
                "input_ids contain a token outside the tokenizer vocabulary"
            )
        try:
            text = bounded_decode(ids, skip_special_tokens=False)
        except TypeError as decode_error:
            # Some custom tokenizers expose decode(ids) only. Do not mistake a
            # TypeError raised inside a compatible decoder for a signature mismatch.
            try:
                signature = inspect.signature(decode)
            except (ValueError, TypeError):
                # Extension callables need not expose a Python signature.
                try:
                    text = bounded_decode(ids)
                except Exception:
                    raise decode_error
            else:
                try:
                    signature.bind(ids, skip_special_tokens=False)
                except TypeError:
                    text = bounded_decode(ids)
                else:
                    raise decode_error
        return text.rstrip("\n")

    window = max(64, len(bos_tokens) + 16)
    tail_end = len(input_ids)
    scan_start = max(0, tail_end - MAX_THINK_ANCHOR_SCAN_TOKENS)
    tail_start = tail_end
    tail_text = ""
    while tail_start > scan_start:
        tail_start = max(scan_start, tail_start - window)
        tail_text = decode_tail(input_ids[tail_start:tail_end])
        if len(tail_text) >= len(anchor):
            break
        if not tail_text:
            # Skip newline-only chunks without repeatedly decoding a growing tail.
            tail_end = tail_start
    if scan_start and tail_start == scan_start and len(tail_text) < len(anchor):
        raise ValueError(
            "thinking boundary cannot be determined within "
            f"{MAX_THINK_ANCHOR_SCAN_TOKENS} trailing input tokens"
        )
    if not tail_text.endswith(anchor):
        return False, []
    for size in range(1, tail_end - tail_start + 1):
        suffix_text = decode_tail(input_ids[tail_end - size : tail_end])
        if suffix_text == anchor:
            return True, input_ids[tail_end - size :]
        if suffix_text.endswith(anchor) and "\ufffd" not in suffix_text:
            # We already reached preceding prompt text. Longer suffixes cannot
            # be safely echoed/removed; avoid scanning the remaining tail window.
            break
    return True, []


def build_empty_think_phase2_input_ids(
    input_ids: list[int],
    matched_bos_ids: list[int],
    empty_think_tokens: list[int],
    *,
    prompt_has_think_anchor: bool = False,
    empty_think_suffix_tokens: Sequence[int] = (),
) -> list[int]:
    """Replace a removable anchor, or close a context-merged one in place."""
    base = list(input_ids)
    if matched_bos_ids and base[-len(matched_bos_ids) :] == matched_bos_ids:
        return base[: -len(matched_bos_ids)] + list(empty_think_tokens)
    if prompt_has_think_anchor:
        return base + list(empty_think_suffix_tokens)
    return base + list(empty_think_tokens)


THINK_MODE_AUTO = "auto"
THINK_MODE_FORCE = "force"
DEFAULT_MAX_THINKING_TOKENS = 32000
INT32_MAX = 2_147_483_647


def normalize_think_mode(mode: object) -> str:
    """Normalize startup think mode.

    RTP historically treats ``THINK_MODE=1`` as "enabled". For dash-sc's
    pre-tokenized path, map that enabled value to Auto so requests that already
    carry ``<think>`` keep their exact prefix while requests without it only get
    the single BOS token appended. String values are accepted for targeted tests
    and future config plumbing.
    """

    if isinstance(mode, str):
        mode_lower = mode.strip().lower()
        if mode_lower in {"force", "forced", "2"}:
            return THINK_MODE_FORCE
        if mode_lower in {"auto", "1", "true", "yes", "on"}:
            return THINK_MODE_AUTO
        return THINK_MODE_AUTO
    if isinstance(mode, bool):
        return THINK_MODE_AUTO if mode else THINK_MODE_AUTO
    if isinstance(mode, int):
        return THINK_MODE_FORCE if mode == 2 else THINK_MODE_AUTO
    return THINK_MODE_AUTO


@dataclass(frozen=True)
class DashScThinkConfig:
    enabled: bool = False
    mode: str = THINK_MODE_AUTO
    bos_tokens: tuple[int, ...] = ()
    end_think_token_ids: tuple[int, ...] = ()
    eos_tokens: tuple[int, ...] = ()
    empty_tokens: tuple[int, ...] = ()
    is_deepseek_v4: bool = False
    dsv4_abort_token_id: int = 1

    @staticmethod
    def disabled() -> "DashScThinkConfig":
        return DashScThinkConfig(enabled=False)

    @property
    def usable(self) -> bool:
        return (
            self.enabled
            and bool(self.bos_tokens)
            and bool(self.end_think_token_ids)
            and bool(self.eos_tokens)
            and bool(self.empty_tokens)
            and self.bos_token_id is not None
            and self.eos_token_id is not None
        )

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.bos_tokens[0] if self.bos_tokens else None

    @property
    def eos_token_id(self) -> Optional[int]:
        if len(self.eos_tokens) >= 2:
            return self.eos_tokens[1]
        if self.eos_tokens:
            return self.eos_tokens[0]
        if self.end_think_token_ids:
            return self.end_think_token_ids[0]
        return None


@dataclass(frozen=True)
class DashScThinkPlan:
    thinking: bool
    input_ids: list[int]
    echo_prefix_ids: list[int] = field(default_factory=list)
    max_thinking_tokens: int = DEFAULT_MAX_THINKING_TOKENS
    prompt_append_len: int = 0
    think_bos_tokens_len: int = 0
    prompt_metric_excluded_len: int = 0
    reason: str = ""

    @staticmethod
    def disabled(input_ids: list[int], reason: str) -> "DashScThinkPlan":
        return DashScThinkPlan(thinking=False, input_ids=list(input_ids), reason=reason)


def plan_dash_sc_thinking(
    input_ids: list[int],
    *,
    think_config: Optional[DashScThinkConfig],
    enable_thinking: Optional[bool],
    max_new_think_tokens: Optional[int],
    default_max_thinking_tokens: int = DEFAULT_MAX_THINKING_TOKENS,
) -> DashScThinkPlan:
    ids = list(input_ids)
    if think_config is None or not think_config.usable:
        return DashScThinkPlan.disabled(ids, "think_config_disabled")
    if enable_thinking is False:
        return DashScThinkPlan.disabled(ids, "enable_thinking_false")
    if isinstance(max_new_think_tokens, int) and max_new_think_tokens == 0:
        return DashScThinkPlan.disabled(ids, "max_new_think_tokens_non_positive")

    bos_tokens = list(think_config.bos_tokens)
    empty_tokens = list(think_config.empty_tokens)
    # DashLLM spec: only when ``empty_tokens`` is the input *suffix* should the
    # request opt out of phase-1 reasoning. ``contains_sublist`` would also fire
    # on an empty-think marker buried in multi-turn chat history (e.g. an
    # earlier assistant turn that produced no thinking), wrongly downgrading a
    # new turn whose tail is ``<think>\n``.
    if (
        empty_tokens
        and len(ids) >= len(empty_tokens)
        and ids[-len(empty_tokens) :] == empty_tokens
    ):
        return DashScThinkPlan.disabled(ids, "empty_think_present_tail")

    if isinstance(max_new_think_tokens, int):
        max_thinking_tokens = (
            INT32_MAX if max_new_think_tokens < 0 else int(max_new_think_tokens)
        )
    else:
        max_thinking_tokens = int(default_max_thinking_tokens)
    bos_token_id = think_config.bos_token_id
    assert bos_token_id is not None

    if len(ids) >= len(bos_tokens) and ids[-len(bos_tokens) :] == bos_tokens:
        prompt_excluded = len(bos_tokens)
        return DashScThinkPlan(
            thinking=True,
            input_ids=ids,
            echo_prefix_ids=bos_tokens,
            max_thinking_tokens=max_thinking_tokens,
            think_bos_tokens_len=len(bos_tokens),
            prompt_metric_excluded_len=prompt_excluded,
            reason="tail_full_bos",
        )

    if ids and ids[-1] == bos_token_id:
        return DashScThinkPlan(
            thinking=True,
            input_ids=ids,
            echo_prefix_ids=[bos_token_id],
            max_thinking_tokens=max_thinking_tokens,
            think_bos_tokens_len=1,
            prompt_metric_excluded_len=1,
            reason="tail_single_bos",
        )

    # No special "bos appears within last 4096 tokens" branch: relying on the
    # model to spontaneously emit ``<think>\n`` again when prompt history
    # contains it elsewhere (and the tail does not) is unreliable. DashLLM
    # ``_think.py:407`` keeps such a branch (``len=0`` no echo / no append),
    # but in RTP the C++ ``ThinkModeLogitsProcessor`` *forces* phase-1
    # thinking via ``in_think_mode=True``, so the assumption that the model
    # will mirror the historic ``<think>`` is decoupled from the actual
    # logits dispatch and breaks for several MRCR cases (qid 40 / 233 / 300
    # / 555 / 643): model proceeds straight to answer tokens, dashscope-side
    # parser never sees a ``<think>`` anchor, ``content`` ends up empty.
    # Falling through to Auto/Force append below restores the anchor by
    # appending bos to the prompt and echoing it on the first chunk.
    if think_config.mode == THINK_MODE_FORCE:
        appended = bos_tokens
        reason = "force_append_full_bos"
    else:
        appended = [bos_token_id]
        reason = "auto_append_single_bos"

    return DashScThinkPlan(
        thinking=True,
        input_ids=ids + appended,
        echo_prefix_ids=list(appended),
        max_thinking_tokens=max_thinking_tokens,
        prompt_append_len=len(appended),
        think_bos_tokens_len=len(appended),
        reason=reason,
    )
