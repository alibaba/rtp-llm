import ctypes
import os
import signal
import sys


def run_with_parent_death_signal(parent_pid: int, target, *args):
    """Bind a worker to its spawning parent before starting model/server work."""
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        prctl.restype = ctypes.c_int
        # PR_SET_PDEATHSIG is per child and must be installed again at each
        # process-tree level. SIGKILL also works during native calls or shutdown
        # handlers that defer/ignore SIGTERM. Spawn from a long-lived thread.
        if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, "failed to install parent-death signal")
        # The parent can die while spawn imports/unpickles the child, before
        # prctl is installed. Compare with the PID captured BEFORE spawning.
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)
    return target(*args)
