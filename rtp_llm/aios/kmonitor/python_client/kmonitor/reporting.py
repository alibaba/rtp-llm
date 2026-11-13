"""Sleep reporting state shared by the local serving processes.

Even epochs report; odd epochs discard. Only senders and transitions take the
send lock; metric producers never wait for network I/O.
No shared state is installed when sleep mode is disabled.
"""

import time
from contextlib import contextmanager, nullcontext
from multiprocessing import get_context
from threading import RLock


class ReportingState:
    def __init__(self, rank_count: int = 1, context=None, *, frontend_only=False):
        if rank_count < 1:
            raise ValueError("reporting rank_count must be positive")
        context = context or get_context()
        self.lock = context.RLock()
        self.send_lock = context.RLock()
        self.epoch = context.Value("q", 0, lock=False)
        self.resume_time = context.Value("d", 0, lock=False)
        self.rank_enabled = context.Array("b", [1] * rank_count, lock=False)
        self.frontend_only = frontend_only

    def set_rank_enabled(self, rank: int, enabled: bool) -> None:
        with self.send_lock, self.lock:
            self.rank_enabled[rank] = int(enabled)
            # Keep the previous state while local ranks converge. One restored
            # rank must not resume every frontend while its peers still sleep.
            if all(self.rank_enabled):
                self._set_enabled(True)
            elif not any(self.rank_enabled):
                self._set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        with self.send_lock, self.lock:
            for rank in range(len(self.rank_enabled)):
                self.rank_enabled[rank] = int(enabled)
            self._set_enabled(enabled)

    def _set_enabled(self, enabled: bool) -> None:
        if (self.epoch.value % 2 == 0) != enabled:
            if enabled:
                self.resume_time.value = time.time()
            self.epoch.value += 1


_state = None
_unconfigured_send_lock = RLock()
_state_install_lock = RLock()


def configure(state: ReportingState) -> None:
    global _state
    # Finish pre-install sends and metric operations before publishing the
    # shared state. Never hold the producer lock while waiting for network I/O.
    with _unconfigured_send_lock, _state_install_lock:
        _state = state


@contextmanager
def reporting_lock():
    # Keep the installed state stable through the metric lock and nested epoch
    # reads. Otherwise a producer entering with no state can later acquire the
    # new state lock after its metric lock, reversing the collector's order.
    with _state_install_lock:
        with _state.lock if _state is not None else nullcontext():
            yield


@contextmanager
def reporting_send_lock():
    # Read the installed state after acquiring the startup fence, so a sender
    # queued before configure() cannot bypass the newly installed shared lock.
    with _unconfigured_send_lock:
        state = _state
        if state is None:
            yield
            return
    with state.send_lock:
        yield


def reporting_epoch() -> int:
    with reporting_lock():
        return _state.epoch.value if _state is not None else 0


def reporting_resume_time() -> float:
    with reporting_lock():
        return _state.resume_time.value if _state is not None else 0


def set_backend_reporting(enabled: bool, local_rank: int) -> None:
    global _state
    if _state is None:
        # A directly started backend has no shared frontend controller. Fence
        # any send or metric operation that began before installation as well.
        with _unconfigured_send_lock, _state_install_lock:
            if _state is None:
                _state = ReportingState()
    if len(_state.rank_enabled) == 1:
        local_rank = 0
    _state.set_rank_enabled(local_rank, enabled)


def set_instance_reporting(enabled: bool) -> None:
    # Backend callbacks own their rank entries. Only a standalone frontend has
    # no local backend callback and needs the coordinator's completion notice.
    if _state is not None and _state.frontend_only:
        _state.set_enabled(enabled)
