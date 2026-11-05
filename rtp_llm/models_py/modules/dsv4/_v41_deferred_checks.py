"""Deferred device-assert grouping for V4.1 per-layer validation (G3).

Under ``DSV41_GROUPED_ASSERTS=1`` the device-side checks registered through
:func:`add_check` accumulate and are emitted as one stacked ``all()`` plus a
single ``torch._assert_async`` per flush group (normally one transformer
layer, flushed from ``V41Block.finish``; an overflow flush bounds the group
where a structural flush point is not on the call path). Every check is
still computed and still gates the run; only the assert launches are
batched. ``torch._assert_async`` rejects long device messages, so the merged
assert message stays short and the full per-check registry is published on
stderr once per unique composition. Default off: call sites keep their
individual asserts.

Graph capture never defers — host-side accumulation cannot be replayed — so
callers must fall back to the immediate assert while a stream is capturing.
"""

import os
import sys

import torch

_GROUPED_ASSERTS_FLAG = "DSV41_GROUPED_ASSERTS"
_MAX_PENDING = 32
# torch._assert_async enforces a device-message length limit (observed:
# "RuntimeError: Message length" past ~1k). Stay far below any plausible cap.
_MAX_MESSAGE = 200

_pending: list = []
_registry_seen: set = set()


def grouped_asserts_enabled() -> bool:
    return os.environ.get(_GROUPED_ASSERTS_FLAG, "0") == "1"


def add_check(valid: torch.Tensor, message: str) -> None:
    """Queue one device-side scalar boolean check for the next grouped flush."""
    _pending.append((valid, message))
    if len(_pending) >= _MAX_PENDING:
        flush_checks("overflow")


def _publish_registry(group: str, messages) -> None:
    signature = tuple(messages)
    if signature in _registry_seen:
        return
    _registry_seen.add(signature)
    print(
        "V4.1 deferred %s check registry (%d checks):" % (group, len(messages)),
        file=sys.stderr,
    )
    for i, message in enumerate(messages):
        print("  [%d] %s" % (i, message), file=sys.stderr)


def flush_checks(group: str) -> None:
    """Emit one grouped ``torch._assert_async`` covering all pending checks."""
    if not _pending:
        return
    items = _pending[:]
    _pending.clear()
    if len(items) == 1:
        torch._assert_async(items[0][0], items[0][1][:_MAX_MESSAGE])
        return
    stacked = torch.stack([valid.reshape(()) for valid, _ in items])
    _publish_registry(group, [message for _, message in items])
    merged = "V4.1 deferred %s checks failed (%d grouped; registry on stderr)" % (
        group,
        len(items),
    )
    torch._assert_async(stacked.all(), merged[:_MAX_MESSAGE])


def pending_count() -> int:
    return len(_pending)
