import logging
import math
import os
import time

import numpy as np
import torch

from rtp_llm.models_py.sampling.adaptive_controller import AdaptiveDraftController


class AdaptiveVerificationSession:
    def __init__(self, capacity: int, steps: int, device: str) -> None:
        self.controller = AdaptiveDraftController(capacity, steps, torch.device(device))
        self.steps = steps
        self.fixed_cap = int(os.environ.get("RTP_LLM_ADAPTIVE_VERIFY_CAP", "-1"))
        if not -1 <= self.fixed_cap <= steps:
            raise ValueError(
                "RTP_LLM_ADAPTIVE_VERIFY_CAP must be -1 or between 0 and draft steps"
            )
        self.ids: list[int] = []
        self.registered: set[int] = set()
        self.round_costs: dict[int, dict[int, float]] = {}
        self.events: list[torch.cuda.Event] | None = None
        self.pending_costs: list[tuple[int, int, list[torch.cuda.Event], float]] = []
        self.event_pool: list[list[torch.cuda.Event]] = []
        self.total = 0
        self.round = 0
        self._prepared = None
        self._feedback_stream = torch.cuda.Stream(device=device)
        self._pending_predictions = []
        self._prediction_buffers = []
        self._host_predictions = {}
        self._request_epochs = {}
        self._next_epoch = 0

    def _collect_predictions(self) -> None:
        pending = []
        for ids, epochs, generation, host, ready in self._pending_predictions:
            if not ready.query():
                pending.append((ids, epochs, generation, host, ready))
                continue
            values = host.numpy()
            for index, request_id in enumerate(ids):
                if self._request_epochs.get(request_id) == epochs[index]:
                    row = values[index]
                    if np.all(np.isfinite(row)) and np.all((row >= 0) & (row <= 1)):
                        self._host_predictions[request_id] = (generation, row.copy())
                    else:
                        # Prediction failure must not strand TP peers waiting
                        # for a budget broadcast. Sampling still uses real logits.
                        self._host_predictions.pop(request_id, None)
            self._prediction_buffers.append(host)
        self._pending_predictions = pending

    def _queue_predictions(self) -> None:
        if not self.ids or len(self._pending_predictions) >= 16:
            return
        slots = self.controller._slots(self.ids)
        source_stream = torch.cuda.current_stream(self.controller.device)
        values = self.controller.confidence_probs.index_select(
            0, self.controller._device_indices(slots)
        )
        host = next(
            (
                buffer
                for buffer in self._prediction_buffers
                if buffer.shape == values.shape
            ),
            None,
        )
        if host is None:
            host = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
        else:
            self._prediction_buffers = [
                b for b in self._prediction_buffers if b is not host
            ]
        # Bound reusable pinned storage when the online batch size varies.
        self._prediction_buffers = self._prediction_buffers[-16:]
        produced = torch.cuda.Event()
        produced.record(source_stream)
        ready = torch.cuda.Event()
        with torch.cuda.stream(self._feedback_stream):
            self._feedback_stream.wait_event(produced)
            host.copy_(values, non_blocking=True)
            values.record_stream(self._feedback_stream)
            ready.record()
        self._pending_predictions.append(
            (
                tuple(self.ids),
                tuple(self._request_epochs[i] for i in self.ids),
                self.round,
                host,
                ready,
            )
        )

    def _collect_costs(self) -> None:
        pending = []
        for batch, total, events, cpu_ms in self.pending_costs:
            if not events[1].query():
                pending.append((batch, total, events, cpu_ms))
                continue
            # CPU and GPU work overlap; adding them double counts execution.
            # Include the whole round's GPU queue and the host submission cost,
            # rather than pricing only target forward and ignoring its tail.
            measured = max(events[0].elapsed_time(events[1]), cpu_ms, 1e-6)
            costs = self.round_costs.setdefault(batch, {})
            costs[total] = (
                0.8 * costs[total] + 0.2 * measured if total in costs else measured
            )
            self.event_pool.append(events)
        self.pending_costs = pending

    def begin(self, ids: list[int]) -> None:
        self._cpu_round_start = time.perf_counter()
        if self.controller.abort_pending():
            logging.warning(
                "[adaptive-verify] discarding an unfinished verification round"
            )
        for stale in self.registered - set(ids):
            self.controller.release(stale)
            self._host_predictions.pop(stale, None)
            self._request_epochs.pop(stale, None)
        self.registered.intersection_update(ids)
        for request_id in ids:
            if request_id not in self.registered:
                self._next_epoch += 1
                self._request_epochs[request_id] = self._next_epoch
            self.controller.register(request_id)
            self.registered.add(request_id)
        self.ids = list(ids)
        self._collect_costs()
        self._collect_predictions()
        self.events = None
        if self.fixed_cap < 0 and len(self.pending_costs) < 16:
            self.events = (
                self.event_pool.pop()
                if self.event_pool
                else [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            )
            self.events[0].record()
        self._prepare_plan()

    def predict(self, ids: list[int], step: int, logits: torch.Tensor) -> None:
        if not ids or self.fixed_cap >= 0:
            return
        # The draft sampler is top-1 without request temperature or penalties.
        # Estimate the distribution that produced those proposals, not Target's.
        if any(request_id not in self.registered for request_id in ids):
            return
        self.controller.predict(ids, step, logits)

    def _prepare_plan(self) -> None:
        plan_start = time.perf_counter()
        batch = len(self.ids)
        costs = self.round_costs.setdefault(batch, {})
        cap = self.fixed_cap
        if cap < 0:
            in_flight = {
                total for size, total, _, _ in self.pending_costs if size == batch
            }
            cap = next(
                (
                    n
                    for n in range(self.steps, -1, -1)
                    if batch * (n + 1) not in costs and batch * (n + 1) not in in_flight
                ),
                -1,
            )
            if cap < 0 and any(
                batch * (n + 1) not in costs for n in range(self.steps + 1)
            ):
                cap = self.steps
        draft = np.zeros(batch + 1)
        verify = np.ones(batch * (self.steps + 1) + 1)
        if cap < 0:
            points = sorted(costs)
            # Captured graphs execute discrete padded row counts. Interpolating
            # invents savings while replay can still use the larger graph.
            # Unknown sizes pay the higher adjacent measured cost. Preserve exact
            # measurements: real kernels need not have monotonic latency, and
            # inflating the full-cap baseline can falsely favor a smaller cap.
            measured = np.asarray([costs[n] for n in points])
            sizes = np.arange(batch, len(verify))
            buckets = np.minimum(
                np.searchsorted(points, sizes, side="left"), len(points) - 1
            )
            exact = np.asarray(points)[buckets] == sizes
            ceiling = np.maximum(
                measured[buckets], measured[np.maximum(buckets - 1, 0)]
            )
            verify[batch:] = np.where(exact, measured[buckets], ceiling)
        confidence = None
        snapshots = [self._host_predictions.get(request_id) for request_id in self.ids]
        if snapshots and all(
            snapshot is not None and self.round - snapshot[0] <= 16
            for snapshot in snapshots
        ):
            confidence = np.stack([snapshot[1] for snapshot in snapshots])
        self._prepared = self.controller.prepare_from_host(
            self.ids,
            confidence=confidence,
            draft_cost_ms=draft,
            verify_cost_ms=verify,
            fixed_cap=cap if cap >= 0 else None,
        )
        self.total = self._prepared[2]
        self._calibration = self.fixed_cap < 0 and cap >= 0
        mode = (
            0
            if self.fixed_cap >= 0
            else (1 if self._calibration else (3 if confidence is None else 2))
        )
        zero, full = self.controller.host_cap_counts
        self._metrics = (mode, zero, full, (time.perf_counter() - plan_start) * 1e6)

    def metrics_snapshot(self) -> tuple[int, int, int, float]:
        """Host-only decision counters; no GPU reductions or synchronization."""
        return self._metrics

    def prepared_plan(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._prepared is None:
            raise RuntimeError("begin must precede adaptive planning")
        return self._prepared

    def plan(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        caps, rows, total = self.prepared_plan()
        self.controller.commit_prepared(self.ids, caps)
        logging.debug(
            "[adaptive-verify] round=%d batch=%d draft_steps=%d target_rows=%d fixed_cap=%d calibration=%s",
            self.round,
            len(self.ids),
            self.steps,
            self.total,
            self.fixed_cap,
            self._calibration,
        )
        return caps, rows, total

    def finish(self, host_elapsed_ms: float | None = None) -> None:
        """Record after draft-prefill, sampling and dispatch, without waiting.

        This is a round service-cost estimate, not a client latency measurement.
        The caller may include host gather/dispatch time excluded by begin().
        """
        if self.events is None:
            return
        elapsed = (time.perf_counter() - self._cpu_round_start) * 1000
        if (
            host_elapsed_ms is not None
            and math.isfinite(host_elapsed_ms)
            and host_elapsed_ms >= 0
        ):
            elapsed = host_elapsed_ms
        # A wall-clock adjustment in C++ telemetry must not fail a valid forward
        # on rank0 while its TP peers continue. Fall back to our monotonic clock.
        self.events[1].record()
        self.pending_costs.append((len(self.ids), self.total, self.events, elapsed))
        self.events = None

    def observe(
        self,
        accepted: torch.Tensor,
        forced: list[bool],
        effective_capacities: torch.Tensor | None = None,
    ) -> None:
        self.controller.observe(
            self.ids,
            accepted,
            forced,
            asynchronous=True,
            effective_capacities=effective_capacities,
        )
        if self.fixed_cap < 0:
            self._queue_predictions()
        self.round += 1
        self._prepared = None


def create_session(
    capacity: int, steps: int, device: str
) -> AdaptiveVerificationSession:
    return AdaptiveVerificationSession(capacity, steps, device)
