"""Bound a hung test in-process: dump all stacks, then exit with code 70.

device_resource.py (the gpu_lock wrapper) kills the whole test process group
from the outside and asks for a SIGUSR1 stack dump first. This module is the
in-test counterpart, for the cases the wrapper cannot cover: a direct bazel
run without ``--run_under``, a hang that must be attributed to one worker
process, or a test that wants a larger/smaller budget than the wrapper's.

Usage::

    from rtp_llm.test.utils import hang_watchdog

    timer = hang_watchdog.install()  # honours RTP_TEST_WATCHDOG_SECONDS

A daemon thread fires after ``seconds`` (default ``RTP_TEST_WATCHDOG_SECONDS``,
1800; ``<=0`` disables), prints every thread's traceback via ``faulthandler``,
and calls ``os._exit(70)``. 70 is deliberately not an ordinary test-failure
code, so a watchdog kill stays distinguishable from an assertion failure.

``install()`` also registers SIGUSR1 as a stack-dump signal, so the wrapper's
SIGUSR1 (sent just before SIGKILL) prints stacks instead of terminating the
test outright. The default 1800s budget is meant for fast unit tests; call
``install(seconds=...)`` explicitly for a known long-running test.
"""

import faulthandler
import os
import signal
import sys
import threading
import time
from typing import Optional

DEFAULT_SECONDS = 1800
EXIT_CODE = 70


def seconds_from_env() -> int:
    """Watchdog budget from RTP_TEST_WATCHDOG_SECONDS, defaulting to 1800."""
    raw = os.environ.get("RTP_TEST_WATCHDOG_SECONDS", str(DEFAULT_SECONDS))
    try:
        return int(raw)
    except ValueError:
        print(
            f"hang_watchdog: invalid RTP_TEST_WATCHDOG_SECONDS={raw!r}; "
            f"using {DEFAULT_SECONDS}",
            file=sys.stderr,
        )
        return DEFAULT_SECONDS


def mark(phase: str) -> None:
    """Print a flushed phase marker so a hang can be located without a stack.

    Cheap enough to call before every step that could block (device init,
    process-group setup, JIT/kernel warmup, CUDA graph capture); the marker
    ends up in the bazel test log together with the watchdog output.
    """
    print(
        f"hang_watchdog: phase={phase} pid={os.getpid()} t={time.monotonic():.1f}",
        file=sys.stderr,
        flush=True,
    )


def _register_stack_dump_signal() -> None:
    """Make the wrapper's SIGUSR1 print stacks instead of killing us outright."""
    if not hasattr(signal, "SIGUSR1"):
        return
    try:
        # chain=False: never re-raise into a previous handler, which could be
        # the default disposition and terminate the process before the
        # watchdog thread gets to report a reason.
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (AttributeError, ValueError, OSError) as e:
        # Most likely a non-main thread, where signal handlers cannot be
        # installed. The watchdog thread itself still works.
        print(
            f"hang_watchdog: cannot register SIGUSR1 stack dump: {e}", file=sys.stderr
        )


def install(seconds: Optional[int] = None) -> Optional[threading.Timer]:
    """Arm the watchdog; returns the cancelable timer, or None if disabled.

    The SIGUSR1 stack-dump hook is registered even when the timer is disabled
    (``seconds <= 0``): the gpu_lock wrapper uses SIGUSR1 to ask for a stack
    dump before it SIGKILLs the process group, and that must keep working.
    """
    if seconds is None:
        seconds = seconds_from_env()
    _register_stack_dump_signal()
    if seconds <= 0:
        return None

    def _fire() -> None:
        faulthandler.dump_traceback()
        print(
            f"hang_watchdog: still running after {seconds}s; exiting {EXIT_CODE}",
            file=sys.stderr,
            flush=True,
        )
        os._exit(EXIT_CODE)

    timer = threading.Timer(seconds, _fire)
    timer.daemon = True
    timer.name = "hang-watchdog"
    timer.start()
    return timer
