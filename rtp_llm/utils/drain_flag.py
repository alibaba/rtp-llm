"""A one-way cross-process flag for "the service is draining".

``multiprocessing.Event`` is not safe here: ``set()`` notifies waiters through a Condition that
then waits for every waiter to acknowledge, and a waiter killed inside ``wait()`` (a rank hit by
SIGKILL) never does, so ``set()`` blocks forever. The process that called it -- the backend manager
reacting to that very rank's death -- then never exits.

This flag is one byte of shared memory: ``set()`` writes it and never waits for anyone, and
``wait()`` polls it. It keeps the three ``Event`` methods its users call.
"""

import ctypes
import multiprocessing
import time
from typing import Optional

_POLL_S = 0.05


class DrainFlag:
    def __init__(self):
        # No lock: a holder killed mid-update must not wedge the others, and a byte write is atomic.
        self._flag = multiprocessing.RawValue(ctypes.c_bool, False)

    def set(self) -> None:
        self._flag.value = True

    def is_set(self) -> bool:
        return bool(self._flag.value)

    def wait(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._flag.value:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(_POLL_S, remaining))
            else:
                time.sleep(_POLL_S)
        return True
