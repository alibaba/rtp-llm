import ctypes
import logging
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1


def arm_parent_death_sigkill(expected_parent_pid: int) -> bool:
    """Kill this process if its direct Linux parent disappears.

    The caller must capture expected_parent_pid before spawning this process.
    Comparing it immediately after prctl closes the parent-before-prctl race.
    """
    if not sys.platform.startswith("linux"):
        logging.warning(
            "Parent-death protection is unavailable on platform %s", sys.platform
        )
        return False
    # PID 1 is a valid direct parent when start_server is the container init.
    if expected_parent_pid <= 0:
        raise ValueError(f"invalid expected parent pid: {expected_parent_pid}")

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))

    actual_parent_pid = os.getppid()
    if actual_parent_pid != expected_parent_pid:
        # Do not log before killing: a blocked logging handler would reopen the
        # parent-before-prctl orphan window this branch is meant to close.
        os.kill(os.getpid(), signal.SIGKILL)
        raise RuntimeError("unreachable")
    return True
