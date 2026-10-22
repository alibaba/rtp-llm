"""Process-wide signal disposition guards for the sCR checkpoint lifecycle."""

from __future__ import annotations

import logging
import signal

LOGGER = logging.getLogger(__name__)

_sigpipe_warning_logged = False


def ignore_sigpipe(reason: str = "") -> None:
    """Keep SIGPIPE ignored so a closed pipe yields EPIPE instead of a kill.

    A broken pipe is expected around checkpointing: AION writes to a helper's
    stdin, RPC peers close sockets, and SCR may tear down a helper before every
    writer observes it. The default SIGPIPE action terminates the process, which
    leaves a dead rank inside the checkpoint image and makes the restore resume
    a broken process group. Ignoring the signal keeps the rank alive and turns
    the failure into an ``EPIPE``/``BrokenPipeError`` on the write path.

    Python ignores SIGPIPE at startup, but native libraries loaded later (AION,
    NCCL, RDMA) can install their own handler, so this is re-asserted at startup
    and again right before a template checkpoint barrier.
    """

    global _sigpipe_warning_logged
    try:
        previous = signal.getsignal(signal.SIGPIPE)
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        # Non-POSIX platforms and non-main threads cannot change dispositions.
        return
    if previous is signal.SIG_IGN:
        return
    if _sigpipe_warning_logged:
        LOGGER.debug("Re-ignored SIGPIPE%s", f" {reason}" if reason else "")
        return
    _sigpipe_warning_logged = True
    LOGGER.warning(
        "SIGPIPE disposition was %s%s; ignoring it so a closed pipe cannot "
        "terminate the rank",
        previous,
        f" {reason}" if reason else "",
    )


__all__ = ["ignore_sigpipe"]
