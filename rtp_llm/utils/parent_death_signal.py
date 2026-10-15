import ctypes
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1


def install_parent_death_signal(
    signum: int = signal.SIGKILL, expected_parent_pid: int | None = None
) -> bool:
    """Ask Linux to terminate this process when its current parent exits.

    GPU workers must not outlive the process that supervises them.  Container
    runtimes do not always reap descendants when only the container's main PID
    is replaced, which can otherwise leave an invisible CUDA context holding
    most of a device's memory.

    ``expected_parent_pid`` should be captured by the supervisor before it
    starts this process.  It closes the earlier race where the supervisor exits
    before this function begins and the child has already been reparented.

    Returns False on platforms without Linux prctl support.  On Linux, setup
    failures are surfaced because continuing without the lifecycle guarantee
    can leak GPU processes.  The second getppid() closes the race where the
    parent exits between observing it and installing PR_SET_PDEATHSIG.
    """
    if sys.platform != "linux":
        return False

    parent_pid = os.getppid()
    if expected_parent_pid is not None and parent_pid != expected_parent_pid:
        os.kill(os.getpid(), signum)
        return True
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

    if prctl(_PR_SET_PDEATHSIG, signum, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))

    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signum)

    return True
