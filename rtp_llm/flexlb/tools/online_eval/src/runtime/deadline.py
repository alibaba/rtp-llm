"""Cooperative deadlines and the synchronous main-thread POSIX guard."""

import signal
import threading
import time
from contextlib import contextmanager


class StageTimeout(TimeoutError):
    pass


@contextmanager
def interruptible(deadline, enabled=False):
    """Main-thread POSIX guard for synchronous backend calls, never a future timeout."""
    if not enabled:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("signal deadline enforcement requires the main thread")
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    entered = time.monotonic()

    def expire(signum, frame):
        raise StageTimeout("stage wall-clock deadline expired")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline.remaining()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_timer[0] - (time.monotonic() - entered)),
                previous_timer[1],
            )


class Deadline:
    def __init__(
        self, expires_at, clock=time.monotonic, sleeper=time.sleep, cancelled=None
    ):
        self.expires_at = expires_at
        self.clock = clock
        self.sleeper = sleeper
        self.cancelled = cancelled or threading.Event()

    def remaining(self):
        remaining = self.expires_at - self.clock()
        if self.cancelled.is_set() or remaining <= 0:
            raise StageTimeout("deadline expired or operation cancelled")
        return remaining

    def check(self):
        if self.cancelled.is_set() or self.clock() >= self.expires_at:
            raise StageTimeout("deadline expired or operation cancelled")

    def sleep(self, seconds):
        target = self.clock() + seconds
        while True:
            now = self.clock()
            if self.cancelled.is_set() or now >= self.expires_at:
                raise StageTimeout("deadline expired or operation cancelled")
            if now >= target:
                break
            self.sleeper(min(0.1, target - now, self.expires_at - now))
        self.check()
