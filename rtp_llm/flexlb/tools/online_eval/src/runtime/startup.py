"""Startup waits share the caller's budget and report the failing phase."""

import time
from runtime.deadline import Deadline


def startup_deadline(deadline, timeout_s):
    return deadline if deadline is not None else Deadline(time.monotonic() + timeout_s)


def wait_ready(deadline, phase, probe, *, process=None, interval_s=0.5, stable_s=0):
    since = None
    while True:
        try:
            deadline.check()
            if process is not None and not process.alive():
                raise RuntimeError(f"{phase}: process exited\n{process.tail_log()}")
            ready = probe()
            deadline.check()
            if process is not None and not process.alive():
                raise RuntimeError(f"{phase}: process exited\n{process.tail_log()}")
            now = deadline.clock()
            since = (now if since is None else since) if ready else None
            if since is not None and now - since >= stable_s:
                return ready
            deadline.sleep(interval_s)
        except TimeoutError as exc:
            raise TimeoutError(f"{phase}: startup budget expired") from exc
