"""Context helpers for forward access telemetry."""

from __future__ import annotations

import contextvars
from typing import Optional

from rtp_llm.dash_sc.proxy.access_record import ForwardAccessRecord

_CONTEXT_ATTR = "_dash_sc_forward_access_record"
_CURRENT_RECORD: contextvars.ContextVar[
    Optional[tuple[int, ForwardAccessRecord]]
] = (
    contextvars.ContextVar("dash_sc_forward_access_record", default=None)
)


def attach_forward_access_record(context, record: ForwardAccessRecord) -> None:
    _CURRENT_RECORD.set((id(context), record))
    try:
        setattr(context, _CONTEXT_ATTR, record)
    except Exception:
        pass


def get_forward_access_record(context) -> Optional[ForwardAccessRecord]:
    try:
        record = getattr(context, _CONTEXT_ATTR, None)
    except Exception:
        record = None
    if isinstance(record, ForwardAccessRecord):
        return record
    current = _CURRENT_RECORD.get()
    if current is not None and current[0] == id(context):
        return current[1]
    return None
