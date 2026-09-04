"""Child-local core-dump policy for xgrammar sandbox workers."""

from __future__ import annotations

import ctypes
import os

_PR_SET_DUMPABLE = 4
_XGRAMMAR_SANDBOX_CORE_DUMP_ENV = "RTP_LLM_XGRAMMAR_SANDBOX_CORE_DUMP"


def _disable_core_dumps_for_current_process() -> None:
    """Disable core dumps for this process without changing its parent."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _xgrammar_sandbox_core_dump_enabled() -> bool:
    value = os.getenv(_XGRAMMAR_SANDBOX_CORE_DUMP_ENV, "0").strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(
        f"{_XGRAMMAR_SANDBOX_CORE_DUMP_ENV} must be one of "
        "0, 1, false, true, no, yes"
    )


def _configure_xgrammar_sandbox_core_dump_for_current_process() -> None:
    """Apply the xgrammar worker policy to this process only."""
    if not _xgrammar_sandbox_core_dump_enabled():
        _disable_core_dumps_for_current_process()
