"""Startup timing shared by Python orchestration and the engine log."""

import logging
import os
import sys
import time
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)
_restore_origin = None


def startup_event(stage, event, *, elapsed_ms=None, **fields):
    values = {"stage": stage, "event": event, "pid": os.getpid()}
    if elapsed_ms is not None:
        values["elapsed_ms"] = f"{elapsed_ms:.3f}"
    if _restore_origin is not None and _restore_origin[0] == os.getpid():
        values["since_resume_ms"] = (
            f"{(time.monotonic() - _restore_origin[1]) * 1000:.3f}"
        )
        values["generation"] = _restore_origin[2]
    values.update(fields)
    message = "[RTPLLM_STARTUP] " + " ".join(f"{k}={v}" for k, v in values.items())
    LOGGER.info(message)
    # Never import the GPU runtime just for logging (e.g. in a launcher).
    native_log = getattr(
        sys.modules.get("libth_transformer"), "log_startup_event", None
    )
    if callable(native_log):
        try:
            native_log(message)
        except Exception:
            LOGGER.debug("native startup logging unavailable", exc_info=True)


def mark_restore_resumed(generation, worker_id):
    global _restore_origin
    # Reset after Epsilon returns: a clock captured in the checkpoint includes
    # checkpoint downtime and is not a valid measure of post-restore startup.
    _restore_origin = (os.getpid(), time.monotonic(), generation)
    startup_event("restore.barrier_returned", "ready", worker_id=worker_id)


@contextmanager
def startup_stage(stage, **fields):
    started = time.monotonic()
    startup_event(stage, "begin", **fields)
    try:
        yield
    except BaseException as error:
        startup_event(
            stage,
            "failed",
            elapsed_ms=(time.monotonic() - started) * 1000,
            error_type=type(error).__name__,
            **fields,
        )
        raise
    else:
        startup_event(
            stage, "end", elapsed_ms=(time.monotonic() - started) * 1000, **fields
        )
