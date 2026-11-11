"""Bounded physical-step windows; independent of transport batch completion."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from .VLLMExecutionSample import PrefillExecutionSample


@dataclass(frozen=True)
class PrefillExecutionWindow:
    steps: tuple[PrefillExecutionSample, ...]

    def __post_init__(self):
        if not self.steps or len({s.step_id for s in self.steps}) != len(self.steps):
            raise ValueError("window requires distinct physical steps")
        end = self.steps[0].completed_ns
        previous = self.steps[0].started_ns
        for step in self.steps[1:]:
            if step.started_ns < previous or step.started_ns > end:
                raise ValueError("window contains reordered or idle execution")
            end = max(end, step.completed_ns)
            previous = step.started_ns

    @property
    def elapsed_ms(self):
        return (
            max(s.completed_ns for s in self.steps) - self.steps[0].started_ns
        ) / 1e6

    @property
    def global_padded_tokens(self):
        return sum(s.global_padded_tokens for s in self.steps)

    @property
    def execution_step_count(self):
        return len(self.steps)

    def wire_requests(self):
        # A request may continue across several Core steps. Preserve its first
        # observed context and all actual compute, without repeating prefix I/O.
        requests = {}
        for step in self.steps:
            for request in step.requests:
                previous = requests.get(request.request_id)
                if previous is None:
                    requests[request.request_id] = [
                        request.context_tokens,
                        request.compute_tokens,
                        request.external_read_tokens,
                    ]
                else:
                    if previous[0] + previous[1] != request.context_tokens:
                        raise ValueError("request recomputed or skipped observed work")
                    previous[1] += request.compute_tokens
                    previous[2] += request.external_read_tokens
        return tuple(
            (identity, context + compute, context, external)
            for identity, (context, compute, external) in requests.items()
        )


class PrefillExecutionWindowAssembler:
    """Register at submission, finish in any order, emit each step at most once.

    A failed/incomplete step invalidates its whole window. Fixed window sizes
    prevent continuous dynamic batching from extending a cohort indefinitely.
    This only assembles facts; it does not fit a model or change engine admission.
    """

    def __init__(self, window_sizes=(4, 8, 16), max_pending_steps=128):
        if (
            not window_sizes
            or any(type(n) is not int or n <= 0 for n in window_sizes)
            or max_pending_steps < max(window_sizes)
        ):
            raise ValueError("invalid window bounds")
        self._sizes = tuple(window_sizes)
        self._limit = max_pending_steps
        self._sequence = 0
        self._windows = []
        self._pending = {}
        self._last_id = 0
        self._lock = RLock()
        self.rejected_windows = 0

    def register(self, step_id):
        with self._lock:
            if type(step_id) is not int or step_id <= self._last_id:
                raise ValueError("physical step ids must follow submission order")
            self._last_id = step_id
            if len(self._pending) >= self._limit:
                self.rejected_windows += len(self._windows)
                self._pending.clear()
                self._windows.clear()
            if not self._windows or len(self._windows[-1][1]) == self._windows[-1][0]:
                size = self._sizes[self._sequence % len(self._sizes)]
                self._sequence += 1
                self._windows.append((size, []))
            self._windows[-1][1].append(step_id)
            self._pending[step_id] = None

    def complete(self, step_id, sample):
        with self._lock:
            if step_id not in self._pending:
                return ()
            if sample is not None and (
                not isinstance(sample, PrefillExecutionSample)
                or sample.step_id != step_id
            ):
                raise ValueError("completion does not match submitted step")
            if self._pending[step_id] is not None:
                return ()
            self._pending[step_id] = sample if sample is not None else False
            emitted = []
            while self._windows:
                size, ids = self._windows[0]
                if any(self._pending[i] is None for i in ids):
                    break
                # Drain a short tail when every submitted execution completed.
                # At full load there are later in-flight steps, so the fixed
                # window remains open until its configured bound is reached.
                if len(ids) < size and any(v is None for v in self._pending.values()):
                    break
                self._windows.pop(0)
                samples = tuple(self._pending.pop(i) for i in ids)
                if any(s is False for s in samples):
                    self.rejected_windows += 1
                    continue
                try:
                    # Keep idle time out of latency targets without discarding
                    # complete execution spans on either side of the gap.
                    windows = []
                    first = 0
                    end = samples[0].completed_ns
                    previous_start = samples[0].started_ns
                    for index, step in enumerate(samples[1:], 1):
                        if step.started_ns < previous_start:
                            raise ValueError("physical steps are reordered")
                        if step.started_ns > end:
                            windows.append(PrefillExecutionWindow(samples[first:index]))
                            first = index
                        end = max(end, step.completed_ns)
                        previous_start = step.started_ns
                    windows.append(PrefillExecutionWindow(samples[first:]))
                    emitted.extend(windows)
                except ValueError:
                    self.rejected_windows += 1
            return tuple(emitted)
