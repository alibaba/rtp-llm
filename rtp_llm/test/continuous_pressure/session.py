"""Bounded supervisor for preparation, formal supply, capture and drain."""

from __future__ import annotations

import hashlib
import math
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from .config import ExecutionLimits, PressureConfig
from .diagnostics import (
    PressureCoverage,
    PressureTarget,
    assess_activity,
    group_valid,
    select_targets,
    validate_inventory,
)
from .phases import PressurePhases
from .ports import EventSink, PressurePoll
from .seed_plan import SeedPlan
from .sender import BoundedSender, CancellableRequestHandle
from .sequence import AdaptiveBacklogController, ArrivalClock


class DispatchClosed(Exception):
    """Input preparation crossed the final supply cutoff; no RPC was sent."""


class PressureSession:
    """Bounded supervisor. Callbacks do not retain an experiment's event history.

    ``make_payload(kind, job_id, formal_sequence)`` runs in accepted workers.
    ``request(payload, register, before)`` calls ``before(payload)`` immediately
    before sending, uses the returned request ID and registers the transport
    call for cancellation. DispatchClosed from ``before`` closes permission to send.

    ``poll(timeout)`` returns PressurePoll. Remote evidence also uses
    ``poll(timeout, drain_remote_completions=True)`` after ingress drains.
    Optional ``diagnostics_control(action, phase, timeout)`` acknowledges enable,
    phase and disable calls. The optional capture controller has nonblocking
    start(now), pending(now) and stop(reason, now) methods. It can implement
    consume(event) and finish(coverage) for capture evidence.
    ``sink(event)`` persists or aggregates events incrementally. The dispatch
    event records the attempted time before the final permission check.
    Completion and cancellation evidence carry the confirmed send time.
    """

    def __init__(
        self,
        config: PressureConfig,
        *,
        session_id,
        limits: ExecutionLimits,
        concurrency,
        max_input_tokens,
        seed_plan: SeedPlan,
        startup_targets: tuple[PressureTarget, ...],
        make_payload,
        request,
        poll,
        sink: EventSink,
        controller=None,
        diagnostics_control=None,
        rate=0,
        poisson=False,
        random_seed=0,
        clock=time.monotonic,
        sleep=time.sleep,
        executor_factory=ThreadPoolExecutor,
    ):
        self.config, self.limits = config, limits
        self.seed_plan = seed_plan
        self.startup_targets = startup_targets
        seed_count = seed_plan.seed_count
        seed_input_tokens = seed_plan.seed_input_tokens
        self.arrivals = ArrivalClock(rate, poisson, random_seed)
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("positive concurrency required")
        limits.require_request(max_input_tokens)
        limits.require_request(seed_input_tokens)
        self.fill_batch = min(
            concurrency, limits.concurrency_for(config.fill_input_tokens)
        )
        if type(seed_count) is not int or seed_count < 0:
            raise ValueError("invalid seed count")
        self.session_id = session_id
        self.seed_count = seed_count
        self.adaptive = None
        if config.rate_control == "adaptive_backlog":
            if poisson:
                raise ValueError("adaptive backlog requires constant arrivals")
            if rate != 0:
                raise ValueError(
                    "adaptive backlog uses request_rate_max instead of fixed rate"
                )
            self.adaptive = AdaptiveBacklogController(
                rate_min=config.request_rate_min,
                rate_max=config.request_rate_max,
                backlog_low=config.backlog_low,
                backlog_high=config.backlog_high,
            )
        elif config.family_schedule == "tiered_rolling" and rate != 0:
            raise ValueError("tiered rolling requires an unlimited arrival rate")
        self.max_input_tokens, self.seed_input_tokens = (
            max_input_tokens,
            seed_input_tokens,
        )
        self.make_payload, self.request, self.poll, self.sink = (
            make_payload,
            request,
            poll,
            sink,
        )
        self.controller = controller
        self.diagnostics_control = diagnostics_control
        self.control_jobs = []
        self.clock, self.sleep, self.executor_factory = clock, sleep, executor_factory
        self.phase_concurrency = concurrency
        sender_limit = config.max_outstanding if self.adaptive else concurrency
        self.sender = BoundedSender(sender_limit)
        self.lock = threading.RLock()
        self.dispatch_turn = threading.Condition(self.lock)
        # Insertion order is the accepted formal arrival order. Registration
        # retires the head as soon as its transport exists.
        self.formal_turns = {}

        self.wakeup = threading.Event()
        self.jobs = {}
        self.job_id = self.sequence = self.formal_submitted = 0
        self.measurement_dispatches = 0
        self.ingress_termination_uncertain = False
        self.remote_request_reuse = dict(
            completed_requests=0, reused_requests=0, reused_tokens=0
        )
        self.job_kinds = {}
        self.cancel_requested = set()
        self.seed_submitted = 0
        self.seed_completed = 0
        self.seed_failed = 0
        self.seed_cancelled = 0
        self.max_seed_pending = 0
        self.seed_started = self.seed_finished = None
        self.seed_deadline = float("inf")
        self.seed_stop_requested = False
        self.seed_stop_snapshot = ()
        self.fill_sent = 0
        self.fill_headroom_reached = False
        self.fill_headroom_snapshot = ()
        rewarm_cohort_size = config.cohort_size or concurrency
        if (
            config.evidence_policy == "tree_pressure"
            and config.family_schedule == "tiered_rolling"
            and rewarm_cohort_size % 2
        ):
            raise ValueError("tiered real pressure requires an even effective cohort")
        self.rewarm_planned = (
            rewarm_cohort_size // 2
            if config.evidence_policy == "tree_pressure"
            and config.family_schedule == "tiered_rolling"
            else 0
        )
        self.rewarm_submitted = 0
        self.rewarm_completed = 0
        self.rewarm_failed = 0
        self.closed = False
        self.supply_deadline = float("inf")
        self.profile_started = False
        self.profile_pending = False
        self.profile_stopped = False
        self.support_started = None
        self.last_poll_s = float("-inf")
        self.poll_future = None
        self.next_poll = float("inf")
        self.last_progress = self.last_dispatch = clock()
        self.last_execution_progress = clock()
        self.progress = None
        self.ready = False
        self.supply_stats = dict(
            max_outstanding=0,
            rate_count=0,
            rate_sum=0.0,
            rate_min=None,
            rate_max=None,
            backlog_regions=dict(low=0, interpolated=0, high=0),
            saturated=False,
        )

    def emit(self, event, **values):
        with self.lock:
            record = dict(event=event, session_id=self.session_id, **values)
            self.sink(record)
            if event == "phase":
                self.last_emitted_phase = values["phase"]
            if self.controller and hasattr(self.controller, "consume"):
                self.controller.consume(record)
            if event == "phase" and self.diagnostics_control:
                if len(self.control_jobs) >= 16:
                    raise RuntimeError("diagnostic control queue full")
                self.control_jobs.append(
                    (
                        "phase",
                        values["phase"],
                        self.control_executor.submit(
                            self.diagnostics_control,
                            "phase",
                            values["phase"],
                            self.config.control_timeout_s,
                        ),
                    )
                )

    def _control_result(self, action, phase, future, timeout=0):
        try:
            result = future.result(timeout=timeout)
            self.emit(
                "control",
                action=action,
                phase=phase,
                result=result,
                received_s=self.clock(),
            )
            if result is not None and not result.get("success", False):
                raise RuntimeError("diagnostic rank acknowledgement failed")
        except Exception as exc:
            self.emit(
                "error", action=action, phase=phase, error=repr(exc), at_s=self.clock()
            )
            self.fail("diagnostics_control_failed")
            return False
        return True

    def _collect_controls(self):
        with self.lock:
            pending = []
            for action, phase, future in self.control_jobs:
                if future.done():
                    self._control_result(action, phase, future)
                else:
                    pending.append((action, phase, future))
            self.control_jobs = pending

    def _disable_diagnostics(self):
        # Pending phase markers cannot delay cleanup behind a growing work queue.
        for action, phase, future in self.control_jobs:
            if future.cancel():
                self.emit("control_cancelled", action=action, phase=phase)
            else:
                self._control_result(
                    action,
                    phase,
                    future,
                    timeout=self.config.control_timeout_s + 0.1,
                )
        self.control_jobs.clear()
        future = self.control_executor.submit(
            self.diagnostics_control,
            "disable",
            "",
            self.config.control_timeout_s,
        )
        self._control_result(
            "disable",
            "",
            future,
            timeout=self.config.control_timeout_s + 0.1,
        )

    def _final_steps(self):
        deadline = self.clock() + 2.0
        try:
            if self.poll_future is not None:
                evidence = self.poll_future.result(timeout=1.0)
                self.poll_future = None
                if not self._record_pressure(evidence):
                    raise RuntimeError(evidence.raw["diagnostics_error"])
            # One extra empty page handles an exactly full final page.
            for _ in range(17):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    break
                future = self.poll_executor.submit(self.poll, min(1.0, remaining))
                evidence = future.result(timeout=min(1.1, remaining))
                if not self._record_pressure(evidence):
                    raise RuntimeError(evidence.raw["diagnostics_error"])
                if evidence.step_history_complete and not evidence.step_page_full:
                    return
            raise RuntimeError("final step drain limit reached")
        except Exception as exc:
            self.emit("error", action="final_steps", error=repr(exc), at_s=self.clock())
            self.fail("diagnostics_drain_failed")

    def fail(self, reason):
        with self.lock:
            if self.state.phase != "drain":
                self.supply_deadline = min(
                    self.supply_deadline, self.clock() + self.config.drain_timeout_s
                )
            self.state.fail(reason)
            self.dispatch_turn.notify_all()

    def _advance(self):
        with self.lock:
            now = self.clock()
            previous = self.state.phase
            if self.state.phase == "seed" and now >= self.seed_deadline:
                self.fail("seed_timeout")
            if (
                self.profile_started
                and self.profile_pending
                and now >= self.state.measurement_start + self.config.capture_timeout_s
            ):
                self.state.fail("profile_timeout")
            covered = 0.0
            if self.state.phase == "steady" and now > self.state.entered:
                intervals = self.coverage.intervals(self.state.entered, now)
                if (
                    intervals
                    and self.ready
                    and now - intervals[-1][1] <= self.config.max_sample_age_s
                ):
                    covered = intervals[-1][1] - intervals[-1][0]
            self.state.advance(
                now, steady_covered_s=covered, profile_pending=self.profile_pending
            )
            if self.state.phase == "support" and self.support_started is None:
                self.support_started = self.state.measurement_end
            if self.state.phase != previous:
                if self.state.phase == "drain":
                    self.supply_deadline = min(
                        self.supply_deadline, now + self.config.drain_timeout_s
                    )
                self.emit("phase", phase=self.state.phase, at_s=now)
                self.dispatch_turn.notify_all()
            return now

    def _dispatch_phase(self, kind):
        # Publishing a phase transition calls the sink. Sample again after
        # that callback so the timestamp and phase reflect the same clock edge.
        while True:
            previous = self.state.phase
            now = self._advance()
            current = self.state.phase
            if current == previous:
                break
        if (
            self.closed
            or now >= self.supply_deadline
            or current == "drain"
            or (kind == "seed" and current != "seed")
            or (kind == "rewarm" and current != "fill")
            or (
                kind == "formal" and current not in ("steady", "measurement", "support")
            )
        ):
            raise DispatchClosed()
        return (current if kind == "formal" else kind), now

    def _before(self, job_id, kind, scheduled, payload, dispatch_context):
        ids = payload["input_ids"]
        expected = (
            self.config.fill_input_tokens
            if kind == "fill"
            else self.seed_input_tokens if kind == "seed" else self.max_input_tokens
        )
        if len(ids) > expected or (kind in ("seed", "fill") and len(ids) != expected):
            raise ValueError("generated input violates frozen token bound")
        self.limits.concurrency_for(len(ids))
        input_sha256 = hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
        metadata = {k: v for k, v in payload.items() if k != "input_ids"}
        with self.dispatch_turn:
            if kind == "formal":
                self.dispatch_turn.wait_for(
                    lambda: next(iter(self.formal_turns), None) == job_id
                    or job_id not in self.formal_turns
                    or self.closed
                    or self.state.phase == "drain"
                )
                if job_id not in self.formal_turns:
                    raise DispatchClosed()
            phase, now = self._dispatch_phase(kind)
            if self.config.evidence_policy == "tree_then_mock_memory":
                payload["mock_memory_cache_request"] = metadata[
                    "mock_memory_cache_request"
                ] = (phase == "measurement")
            elif self.config.evidence_policy in {
                "tree_then_mock_disk",
                "tree_then_mock_remote",
            }:
                payload["mock_cache_request"] = metadata["mock_cache_request"] = (
                    phase == "measurement"
                )
            sequence = self.sequence
            self.sequence += 1
            identity = f"{self.session_id}:{phase}:{sequence}"
            record = dict(
                job_id=job_id,
                kind=kind,
                sequence=sequence,
                phase=phase,
                request_id=identity,
                attempted_dispatch_s=now,
                scheduled_offset_s=scheduled,
                input_len=len(ids),
                input_sha256=input_sha256,
                metadata=metadata,
            )
            record.update(dispatch_context)
            self.jobs[job_id] = record
            self.emit("dispatch", **record)
            # The tentative dispatch is durable before transport is permitted.
            # Collection reconciles it as not_dispatched if the sink crossed
            # a cutoff or changed the phase encoded in the request identity.
            final_phase, confirmed_s = self._dispatch_phase(kind)
            if final_phase != phase:
                raise DispatchClosed()
            record["dispatched_s"] = confirmed_s
            if scheduled is not None:
                record["schedule_lag_s"] = max(
                    0, confirmed_s - self.formal_origin - scheduled
                )
            self.last_dispatch = confirmed_s
            if kind == "formal":
                self.formal_submitted += 1
                if phase == "measurement":
                    self.measurement_dispatches += 1
            return identity

    def _submit(self, kind, scheduled=None, formal_sequence=None, **dispatch_context):
        job_id = self.job_id
        if formal_sequence is not None:
            dispatch_context = dict(dispatch_context, formal_sequence=formal_sequence)

        def retire_turn():
            if kind == "formal":
                with self.dispatch_turn:
                    if job_id in self.formal_turns:
                        del self.formal_turns[job_id]
                        self.dispatch_turn.notify_all()

        def work(register):
            def transport_created(call):
                retire_turn()
                register(call)

            try:
                payload = self.make_payload(kind, job_id, formal_sequence)
                result = self.request(
                    payload,
                    transport_created,
                    lambda p: self._before(
                        job_id, kind, scheduled, p, dispatch_context
                    ),
                )
                return dict(result, _pressure_completed_s=self.clock())
            except DispatchClosed:
                return {"not_dispatched": True}
            finally:
                retire_turn()

        def start():
            if kind == "formal":
                with self.dispatch_turn:
                    self.formal_turns[job_id] = None

            def done():
                # Queued futures can be cancelled without entering work().
                retire_turn()
                self.wakeup.set()

            try:
                return CancellableRequestHandle(self.executor, work, on_done=done)
            except BaseException:
                retire_turn()
                raise

        accepted = self.sender.submit(job_id, start)
        if accepted:
            self.job_kinds[job_id] = kind
            self.job_id += 1
        return accepted

    def _collect(self):
        for job_id, result in self.sender.collect():
            if result.get("ingress_termination_uncertain"):
                self.ingress_termination_uncertain = True
            kind = self.job_kinds.pop(job_id, None)
            cancelled = job_id in self.cancel_requested
            self.cancel_requested.discard(job_id)
            with self.lock:
                record = self.jobs.pop(job_id, None)
            if record is None:
                if kind == "seed" and not cancelled:
                    self.seed_failed += 1
                    self.fail("request_failed")
                elif kind == "rewarm" and not cancelled:
                    self.rewarm_failed += 1
                    self.fail("request_failed")
                if not result.get("not_dispatched"):
                    self.fail("request_failed")
                    self.emit(
                        "completion",
                        job_id=job_id,
                        completed_s=self.clock(),
                        result=result,
                    )
                continue
            result = dict(result)
            now = result.pop("_pressure_completed_s", self.clock())
            self.emit(
                "completion",
                **record,
                completed_s=now,
                result=result,
                crossing=not result.get("not_dispatched")
                and record["phase"] == "measurement"
                and now >= self.state.measurement_end,
            )
            if result.get("success"):
                self.last_execution_progress = max(self.last_execution_progress, now)
            success = (
                result.get("success")
                and result.get("input_len") == record["input_len"]
                and result.get("output_len") == 1
            )
            if (
                self.config.evidence_policy.startswith("tree_then_mock_")
                and record["phase"] == "measurement"
                and success
                and not cancelled
            ):
                self.remote_request_reuse["completed_requests"] += 1
                reused = result.get("reuse_len")
                if type(reused) is int and reused > 0:
                    self.remote_request_reuse["reused_requests"] += 1
                    self.remote_request_reuse["reused_tokens"] += reused
            if kind == "seed":
                if success:
                    self.seed_completed += 1
                elif not cancelled:
                    self.seed_failed += 1
            elif kind == "rewarm":
                if success:
                    self.rewarm_completed += 1
                elif not cancelled:
                    self.rewarm_failed += 1
            if not success and not cancelled:
                self.fail("request_failed")

    def _consume_poll(self, evidence):
        with self.lock:
            self._apply_poll(evidence)

    def _record_pressure(self, evidence):
        self.emit("pressure", at_s=self.clock(), raw=evidence.raw)
        error = evidence.raw.get("diagnostics_error")
        if error:
            self.emit(
                "error", action="diagnostics_poll", error=error, at_s=self.clock()
            )
        return not error

    def _apply_poll(self, evidence):
        now = self.clock()
        self.last_poll_s = min(
            (s.client_receive_s for s in evidence.samples), default=now
        )
        if not self._record_pressure(evidence):
            self.fail("diagnostics_poll_failed")
            return
        self.coverage.add(evidence.samples, now, complete=evidence.sample_complete)
        if self.config.family_schedule == "tiered_rolling":
            occupancy = self._host_pool_occupancy(evidence.samples)
            at_limit = any(
                item["ratio"] >= self.config.fill_host_max_ratio for item in occupancy
            )
            if self.state.phase == "seed" and not self.seed_stop_requested and at_limit:
                self.seed_stop_requested = True
                self.seed_stop_snapshot = occupancy
                self.emit(
                    "seed_limit",
                    at_s=now,
                    reason="host_max_ratio",
                    configured_limit=self.seed_count,
                    submitted=self.seed_submitted,
                    completed=self.seed_completed,
                    host_pools=list(occupancy),
                )
            elif (
                self.state.phase == "fill"
                and not self.fill_headroom_reached
                and at_limit
            ):
                self.fill_headroom_reached = True
                self.fill_headroom_snapshot = occupancy
                self.emit(
                    "fill_limit",
                    at_s=now,
                    reason="host_max_ratio",
                    sent=self.fill_sent,
                    host_pools=list(occupancy),
                )
        self.ready = False
        if not self.coverage.epoch_valid:
            self.fail("pressure_epoch_invalid")
            return
        if evidence.sample_complete and len(evidence.samples) == len(
            self.coverage.ranks
        ):
            try:
                validate_inventory(
                    select_targets(evidence.samples, self.coverage.ranks),
                    self.coverage.targets,
                )
            except ValueError:
                self.coverage.epoch_valid = False
                self.fail("pressure_epoch_invalid")
                return
            try:
                self.ready = all(
                    self.coverage.target_valid(t, evidence.samples)
                    for t in self.coverage.targets
                )
                self.ready = self.ready and all(
                    0
                    <= now
                    - (
                        s.server_s
                        + (s.offset_bounds or (s.client_send_s - s.server_s,))[0]
                    )
                    <= self.config.max_sample_age_s
                    for s in evidence.samples
                )
            except (ValueError, KeyError, StopIteration):
                self.ready = False
        if not self.ready and self.state.phase in ("measurement", "support"):
            self.emit("pressure_lost", phase=self.state.phase, at_s=now)
        progress = tuple(
            (
                s.rank,
                tuple((g.group_id, g.triggers, g.gpu_host_published) for g in s.groups),
                tuple(
                    (p.pool_id, p.used - p.pending_release, p.host_resident)
                    for p in s.pools
                ),
                tuple(asdict(s.mock_memory).values()) if s.mock_memory else (),
            )
            for s in evidence.samples
        )
        if progress and progress != self.progress:
            self.last_progress, self.progress = now, progress
        if evidence.activity:
            for activity in evidence.activity:
                activity = type(activity)(
                    self.last_dispatch,
                    max(
                        self.last_progress,
                        self.last_execution_progress,
                        activity.last_progress_s,
                    ),
                    activity.active_step_started_s,
                    activity.running,
                    activity.waiting,
                )
                assessment = assess_activity(
                    activity, now, self.config.activity_timeout_s
                )
                if not assessment.valid:
                    self.fail(assessment.reasons[0])
        elif now - self.last_progress >= self.config.activity_timeout_s:
            self.fail("unknown_activity")

    @staticmethod
    def _host_pool_occupancy(samples):
        occupancy = []
        for sample in samples:
            pools = {pool.pool_id: pool for pool in sample.pools}
            for group in sample.groups:
                pool = pools.get(group.host_pool_id)
                if pool is None or pool.capacity <= 0:
                    continue
                resident = max(0, pool.used - pool.pending_release)
                occupancy.append(
                    dict(
                        rank=sample.rank,
                        group_id=group.group_id,
                        pool_id=pool.pool_id,
                        used=resident,
                        capacity=pool.capacity,
                        ratio=resident / pool.capacity,
                    )
                )
        return tuple(occupancy)

    def _tick(self):
        self._collect_controls()
        self._collect()
        now = self.clock()
        if self.poll_future is not None and self.poll_future.done():
            try:
                evidence = self.poll_future.result()
            except Exception as exc:
                evidence = PressurePoll((), {"error": repr(exc)})
            self.poll_future = None
            self._consume_poll(evidence)
        if self.poll_future is None and now >= self.next_poll:
            self.next_poll = now + self.config.sample_period_s
            self.poll_future = self.poll_executor.submit(
                self.poll, self.config.sample_period_s
            )
        if self.controller and self.profile_started and not self.profile_stopped:
            self.profile_pending = self.controller.pending(now)
        self._advance()
        if (
            self.state.phase == "fill"
            and now - self.last_progress >= self.config.fill_no_progress_s
        ):
            self.fail("pressure_not_reached")
        if (
            self.state.phase == "measurement"
            and self.controller
            and not self.profile_started
        ):
            self.controller.start(self.state.measurement_start)
            self.profile_started = True
            self.profile_pending = self.controller.pending(now)

    def _pause(self, until=None):
        delay = min(0.01, self.config.sample_period_s / 10)
        if until is not None:
            delay = min(delay, max(0.0, until - self.clock()))
        if self.sleep is time.sleep:
            self.wakeup.wait(delay)
            self.wakeup.clear()
        else:
            self.sleep(delay)

    def _batch(self, kind, count):
        if self.state.phase == "drain":
            return
        for _ in range(count):
            self._submit(kind)
        while self.sender.pending and self.state.phase != "drain":
            self._tick()
            self._pause()

    def _seed_result(self):
        planned = (
            self.seed_submitted
            if self.config.family_schedule == "tiered_rolling"
            and self.seed_stop_requested
            else self.seed_count
        )
        result = dict(
            planned=planned,
            submitted=self.seed_submitted,
            completed=self.seed_completed,
            failed=self.seed_failed,
            cancelled=self.seed_cancelled,
            pending=sum(kind == "seed" for kind in self.job_kinds.values()),
            max_inflight=self.max_seed_pending,
            elapsed_s=(
                (self.seed_finished - self.seed_started)
                if self.seed_started is not None and self.seed_finished is not None
                else 0.0
            ),
        )
        if self.config.family_schedule == "tiered_rolling":
            result.update(
                configured_limit=self.seed_count,
                effective_seed_count=self.seed_completed,
                stop_reason=(
                    "host_max_ratio" if self.seed_stop_requested else "configured_limit"
                ),
                stop_host_pools=list(self.seed_stop_snapshot),
            )
        return result

    def _rewarm_result(self):
        return dict(
            planned=self.rewarm_planned,
            submitted=self.rewarm_submitted,
            completed=self.rewarm_completed,
            failed=self.rewarm_failed,
            pending=sum(kind == "rewarm" for kind in self.job_kinds.values()),
        )

    def _run_rewarm_phase(self):
        if not self.rewarm_planned:
            return
        self.emit("rewarm_start", planned=self.rewarm_planned, at_s=self.clock())
        while (
            self.state.phase == "fill"
            and not self.fill_headroom_reached
            and self.rewarm_submitted < self.rewarm_planned
        ):
            while (
                self.rewarm_submitted < self.rewarm_planned
                and len(self.sender.pending) < self.sender.limit
                and not self.fill_headroom_reached
            ):
                if not self._submit("rewarm"):
                    break
                self.rewarm_submitted += 1
            while (
                any(kind == "rewarm" for kind in self.job_kinds.values())
                and self.state.phase == "fill"
            ):
                self._tick()
                if self.fill_headroom_reached:
                    self.fail("fill_headroom_exhausted")
                    break
                if self.state.phase != "drain":
                    self._pause()
            if self.state.phase != "fill":
                break
            if self.poll_future is not None:
                evidence = self.poll_future.result(timeout=self.config.sample_period_s)
                self.poll_future = None
                self._consume_poll(evidence)
            if self.fill_headroom_reached:
                self.fail("fill_headroom_exhausted")
            elif self.state.phase == "fill":
                self._consume_poll(self.poll(self.config.sample_period_s))
            if self.fill_headroom_reached:
                self.fail("fill_headroom_exhausted")
        result = self._rewarm_result()
        self.emit("rewarm_summary", **result)
        if (
            result["submitted"] != result["planned"]
            or result["completed"] != result["planned"]
            or result["failed"]
            or result["pending"]
        ):
            self.fail("rewarm_incomplete")

    def _run_seed_phase(self) -> dict:
        self.seed_started = self.clock()
        self.seed_deadline = self.seed_started + self.config.seed_timeout_s
        self.state.phase, self.state.entered = "seed", self.seed_started
        if self.seed_count:
            self.emit("phase", phase="seed", at_s=self.seed_started)
        while self.seed_completed + self.seed_failed < self.seed_count:
            self._advance()
            while (
                self.state.phase == "seed"
                and not self.seed_stop_requested
                and self.seed_submitted < self.seed_count
                and sum(kind == "seed" for kind in self.job_kinds.values())
                < self.phase_concurrency
            ):
                self._advance()
                if self.state.phase != "seed" or not self._submit("seed"):
                    break
                self.seed_submitted += 1
                self.max_seed_pending = max(
                    self.max_seed_pending, len(self.sender.pending)
                )
            self._tick()
            if self.state.phase == "drain":
                break
            if self.seed_stop_requested and not self.sender.pending:
                break
            if self.seed_completed == self.seed_count:
                break
            self._pause()
        self.seed_finished = self.clock()
        return self._seed_result()

    def _start_fill(self):
        # Rebase after the last seed completion. A pre-existing asynchronous poll
        # may have sampled seed transfers; consume it before taking this snapshot.
        if self.seed_count:
            if self.poll_future is not None:
                evidence = self.poll_future.result(timeout=self.config.sample_period_s)
                self.poll_future = None
                self._consume_poll(evidence)
            baseline = self.poll(self.config.sample_period_s)
            self._consume_poll(baseline)
            if self.state.phase == "drain":
                return
            if not baseline.sample_complete:
                raise RuntimeError("post-seed pressure inventory is incomplete")
            occupancy = self._host_pool_occupancy(baseline.samples)
            if self.config.family_schedule == "tiered_rolling" and any(
                item["ratio"] >= self.config.fill_host_max_ratio for item in occupancy
            ):
                self.fill_headroom_reached = True
                self.fill_headroom_snapshot = occupancy
                self.emit(
                    "fill_limit",
                    at_s=self.clock(),
                    reason="host_max_ratio",
                    sent=0,
                    host_pools=list(occupancy),
                )
                self.fail("fill_headroom_exhausted")
                return
            self.coverage = PressureCoverage(tuple(baseline.samples), self.config)
            self.emit("fill_baseline", at_s=self.clock(), raw_baseline=baseline.raw)
        now = self.clock()
        self.state.phase, self.state.entered = "fill", now
        self.last_progress = self.last_execution_progress = now
        self.progress, self.ready = None, False
        self.fill_sent = 0
        self.emit("phase", phase="fill", at_s=now)

    def _run_formal_supply(self):
        self.formal_origin = self.clock()
        if self.adaptive:
            self._run_adaptive_formal_supply()
            return
        due = self.arrivals.next_offset()
        formal_sequence = 0
        while self.state.phase in ("steady", "measurement", "support"):
            self._tick()
            if self.state.phase == "drain":
                break
            while (
                self.state.phase in ("steady", "measurement", "support")
                and self.clock() + 1e-8 >= self.formal_origin + due
            ):
                if not self._submit("formal", due, formal_sequence):
                    break
                formal_sequence += 1
                due = self.arrivals.next_offset()
            if self.state.phase != "drain":
                self._pause()

    def _record_adaptive_dispatch(self, outstanding, rate):
        stats = self.supply_stats
        stats["max_outstanding"] = max(stats["max_outstanding"], outstanding + 1)
        stats["rate_count"] += 1
        stats["rate_sum"] += rate
        stats["rate_min"] = (
            rate if stats["rate_min"] is None else min(stats["rate_min"], rate)
        )
        stats["rate_max"] = (
            rate if stats["rate_max"] is None else max(stats["rate_max"], rate)
        )
        region = (
            "low"
            if outstanding <= self.config.backlog_low
            else "high" if outstanding >= self.config.backlog_high else "interpolated"
        )
        stats["backlog_regions"][region] += 1

    def _run_adaptive_formal_supply(self):
        due = 0.0
        formal_sequence = 0
        while self.state.phase in ("steady", "measurement", "support"):
            self._tick()
            if self.state.phase == "drain":
                break
            while (
                self.state.phase in ("steady", "measurement", "support")
                and self.clock() + 1e-8 >= self.formal_origin + due
            ):
                outstanding = len(self.sender.pending)
                rate = self.adaptive.rate(outstanding)
                if outstanding >= self.sender.limit:
                    self.supply_stats["saturated"] = True
                    self.emit(
                        "sender_saturated",
                        at_s=self.clock(),
                        formal_sequence=formal_sequence,
                        scheduled_offset_s=due,
                        outstanding=outstanding,
                        max_outstanding=self.sender.limit,
                        effective_rate=rate,
                    )
                    self.fail("sender_saturated")
                    break
                if not self._submit(
                    "formal",
                    due,
                    formal_sequence,
                    outstanding_before=outstanding,
                    effective_rate=rate,
                ):
                    raise RuntimeError("adaptive sender rejected below hard limit")
                self._record_adaptive_dispatch(outstanding, rate)
                formal_sequence += 1
                due += 1.0 / rate
            if self.state.phase != "drain":
                self._pause(self.formal_origin + due)

    def _supply_result(self):
        stats = self.supply_stats
        return dict(
            max_outstanding=stats["max_outstanding"],
            configured_max_outstanding=self.sender.limit,
            rate_min=stats["rate_min"],
            rate_max=stats["rate_max"],
            rate_average=(
                stats["rate_sum"] / stats["rate_count"] if stats["rate_count"] else None
            ),
            rate_count=stats["rate_count"],
            backlog_regions=dict(stats["backlog_regions"]),
            saturated=stats["saturated"],
        )

    def run(self) -> dict:
        self.state = PressurePhases(self.config, self.clock())
        self.executor = self.executor_factory(
            max_workers=self.sender.limit, thread_name_prefix="pressure-request"
        )
        self.poll_executor = self.executor_factory(
            max_workers=1, thread_name_prefix="pressure-poll"
        )
        if self.diagnostics_control:
            self.control_executor = self.executor_factory(
                max_workers=1, thread_name_prefix="pressure-control"
            )
        try:
            if self.diagnostics_control:
                future = self.control_executor.submit(
                    self.diagnostics_control,
                    "enable",
                    "",
                    self.config.control_timeout_s,
                )
                if not self._control_result(
                    "enable",
                    "",
                    future,
                    timeout=self.config.control_timeout_s + 0.1,
                ):
                    raise RuntimeError("diagnostic enable failed")
            baseline = self.poll(self.config.sample_period_s)
            try:
                if baseline.raw.get("diagnostics_error"):
                    raise RuntimeError(baseline.raw["diagnostics_error"])
                if not baseline.sample_complete:
                    raise RuntimeError("initial pressure inventory is incomplete")
                validate_inventory(
                    select_targets(
                        tuple(baseline.samples),
                        tuple(sorted({target.rank for target in self.startup_targets})),
                    ),
                    self.startup_targets,
                )
                self.coverage = PressureCoverage(tuple(baseline.samples), self.config)
                # Validate reachability before sending any requests.
                for sample in baseline.samples:
                    for group in sample.groups:
                        if group.group_type == "FULL":
                            group_valid(
                                group,
                                {p.pool_id: p for p in sample.pools},
                                group,
                                self.config,
                            )
            except Exception:
                self._record_pressure(baseline)
                raise
            self.emit(
                "frozen",
                config=asdict(self.config),
                limits=asdict(self.limits),
                seed_plan=asdict(self.seed_plan),
                targets=[asdict(t) for t in self.coverage.targets],
                raw_baseline=baseline.raw,
            )
            self._record_pressure(baseline)
            self.next_poll = self.clock()
            seed = self._run_seed_phase()
            if (
                self.state.phase != "drain"
                and seed["completed"] == seed["planned"]
                and seed["failed"] == 0
                and seed["pending"] == 0
            ):
                self._start_fill()
                if (
                    self.state.phase != "drain"
                    and self.config.evidence_policy == "mock_memory"
                ):
                    self.state.advance(self.clock(), fill_ready=True)
                    self.emit("phase", phase="steady", at_s=self.clock())
            elif self.state.phase != "drain":
                self.fail("seed_incomplete")
            while self.state.phase == "fill":
                self._tick()
                if self.state.phase == "drain":
                    break
                if self.fill_headroom_reached:
                    self.fail("fill_headroom_exhausted")
                    break
                if self.fill_sent and self.ready and not self.sender.pending:
                    self._run_rewarm_phase()
                    if self.state.phase != "drain":
                        self.state.advance(self.clock(), fill_ready=True)
                        self.emit("phase", phase="steady", at_s=self.clock())
                    break
                if (
                    self.fill_sent >= self.config.fill_max_requests
                    or self.clock() - self.last_progress
                    >= self.config.fill_no_progress_s
                ):
                    self.fail("pressure_not_reached")
                    break
                count = min(
                    self.fill_batch, self.config.fill_max_requests - self.fill_sent
                )
                self._batch("fill", count)
                self.fill_sent += count
                completed = self.clock()
                while self.last_poll_s < completed and self.state.phase != "drain":
                    self._tick()
                    if self.state.phase != "drain":
                        self._pause()
            if self.state.phase == "steady":
                self._run_formal_supply()
        except Exception as exc:
            self.fail("supervisor_failed")
            self.emit("error", error=repr(exc), at_s=self.clock())
        finally:
            try:
                self._drain()
            finally:
                if self.diagnostics_control:
                    try:
                        self._disable_diagnostics()
                        self._final_steps()
                    finally:
                        self.control_executor.shutdown(wait=False, cancel_futures=True)
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.poll_executor.shutdown(wait=False, cancel_futures=True)
        profile_result = None
        if self.controller and hasattr(self.controller, "finish"):
            try:
                profile_result = self.controller.finish(getattr(self, "coverage", None))
            except Exception as exc:
                profile_result = dict(
                    trace_valid=False, reasons=[repr(exc)], pressure_valid=False
                )
            if not profile_result.get("trace_valid"):
                self.state.fail("profile_invalid")
        seed = self._seed_result()
        self.emit("seed_summary", **seed)
        result = dict(
            seed=seed,
            rewarm=self._rewarm_result(),
            failure=self.state.failure,
            measurement_start=self.state.measurement_start,
            measurement_end=self.state.measurement_end,
            supply=self._supply_result(),
        )
        if profile_result is not None:
            result["profile"] = profile_result
        if self.state.measurement_end is not None:
            result["pressure"] = asdict(
                self.coverage.assess(
                    self.state.measurement_start, self.state.measurement_end
                )
            )
            result["migration"] = asdict(
                self.coverage.assess_migration(
                    self.state.measurement_start, self.state.measurement_end
                )
            )
            if self.coverage.local_final is not None:
                result["local_writes"] = self.coverage.local_final
            if self.config.evidence_policy == "tree_then_mock_remote":
                result["mock_remote"] = self.coverage.mock_remote_summary()
            if self.config.evidence_policy == "tree_then_mock_disk":
                result["mock_cache"] = self.coverage.mock_cache_summary(
                    self.state.measurement_start, self.state.measurement_end
                )
        if self.support_started is not None and self.support_end > self.support_started:
            result["support_pressure"] = asdict(
                self.coverage.assess(self.support_started, self.support_end)
            )
        self.emit("finished", **result)
        return result

    def _drain(self):
        with self.dispatch_turn:
            self.state.phase = "drain"
            self.dispatch_turn.notify_all()
        started = self.clock()
        if getattr(self, "last_emitted_phase", None) != "drain":
            self.emit("phase", phase="drain", at_s=started)
        self.support_end = started
        deadline = min(self.supply_deadline, started + self.config.drain_timeout_s)
        self.supply_deadline = deadline
        if self.controller and not self.profile_stopped:
            self.profile_stopped = True
            try:
                self.controller.stop(self.state.failure or "complete", started)
            except Exception as exc:
                self.fail("profile_stop_failed")
                self.emit("error", error=repr(exc), at_s=self.clock())
        # Failed setup traffic cannot make progress into another phase. Request
        # cancellation immediately, then use the bounded drain to reap it.
        if self.state.failure and (
            self.state.failure == "sender_saturated"
            or any(kind in ("seed", "rewarm") for kind in self.job_kinds.values())
        ):
            self._cancel_pending()
        while self.clock() < deadline:
            self._collect()
            if hasattr(self, "coverage"):
                self._tick()
            edge = (
                self.support_end
                if self.support_started is not None
                else self.state.measurement_end
            )
            edge_pending = edge is not None and self.last_poll_s <= edge
            if not self.sender.pending and not edge_pending:
                break
            self._pause()
        with self.lock:
            self.closed = True
        if self.sender.pending:
            self.fail("drain_timeout")
            self._cancel_pending()
            self._collect()
        remote = self.config.evidence_policy == "tree_then_mock_remote"
        local = hasattr(self, "coverage") and bool(self.coverage.local_tiers)
        if local:
            self.coverage.local_final = dict(
                valid=False, reasons=["local_completions_not_drained"]
            )
        if (remote or local) and hasattr(self, "coverage"):
            try:
                # No emitter remains, and every transport response must be reaped
                # before the native owner barrier. An older poll must finish
                # first so it cannot overwrite the drained final evidence.
                if self.sender.pending:
                    raise RuntimeError("request ingress did not drain")
                if self.poll_future is not None:
                    evidence = self.poll_future.result(
                        timeout=max(0, deadline - self.clock())
                    )
                    self.poll_future = None
                    self._consume_poll(evidence)
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError("completion drain deadline expired")
                evidence = (
                    self.poll(remaining, drain_remote_completions=True)
                    if remote
                    else self.poll(remaining)
                )
                self._consume_poll(evidence)
                if (
                    evidence.raw.get("diagnostics_error")
                    or not evidence.sample_complete
                ):
                    raise RuntimeError("final drain snapshot is incomplete")
                if self.ingress_termination_uncertain:
                    raise RuntimeError(
                        "stream cancellation did not confirm stopped ingress"
                    )
                if remote:
                    barrier_raw = evidence.raw
                    if (
                        self.measurement_dispatches
                        != self.remote_request_reuse["completed_requests"]
                    ):
                        self.coverage.remote_reasons.add(
                            "remote_request_ingress_uncertain"
                        )
                    self.coverage.finish_remote_window(
                        evidence.samples, request_reuse=self.remote_request_reuse
                    )
                if local:
                    try:
                        self._drain_local_stores(evidence, deadline)
                    except Exception as exc:
                        self.coverage.local_final = dict(
                            valid=False, reasons=["local_drain_failed"]
                        )
                        self.fail("local_drain_failed")
                        self.emit(
                            "error",
                            action="local_drain",
                            error=repr(exc),
                            at_s=self.clock(),
                        )
                if remote:
                    summary = self.coverage.mock_remote_summary()
                    self.emit(
                        "remote_drained",
                        at_s=self.clock(),
                        raw=barrier_raw,
                        evidence=summary,
                    )
                    if not summary["valid"]:
                        self.fail("remote_evidence_invalid")
            except Exception as exc:
                self.fail("remote_drain_failed" if remote else "local_drain_failed")
                self.emit(
                    "error",
                    action="remote_drain" if remote else "local_drain",
                    error=repr(exc),
                    at_s=self.clock(),
                )
        if self.poll_future is not None:
            self.poll_future.cancel()

    def _drain_local_stores(self, evidence, deadline):
        completed = self.remote_request_reuse["completed_requests"]
        while True:
            summary = self.coverage.local_write_summary(evidence.samples, completed)
            if completed != self.measurement_dispatches:
                summary["reasons"].append("local_request_ingress_uncertain")
                summary["valid"] = False
            if not summary["pending"] or self.clock() >= deadline:
                break
            self.sleep(
                min(
                    self.config.sample_period_s,
                    max(0, deadline - self.clock()),
                )
            )
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            evidence = self.poll(remaining)
            self._consume_poll(evidence)
            if evidence.raw.get("diagnostics_error") or not evidence.sample_complete:
                raise RuntimeError("local drain snapshot is incomplete")
        self.coverage.local_final = summary
        self.emit(
            "local_stores_drained",
            at_s=self.clock(),
            raw=evidence.raw,
            evidence=summary,
        )
        if not summary["valid"]:
            self.fail("local_write_evidence_invalid")

    def _cancel_pending(self):
        with self.dispatch_turn:
            self.formal_turns.clear()
            self.dispatch_turn.notify_all()
        for job_id, handle in list(self.sender.pending.items()):
            if job_id in self.cancel_requested:
                continue
            self.cancel_requested.add(job_id)
            if self.job_kinds.get(job_id) == "seed":
                self.seed_cancelled += 1
            error = None
            try:
                accepted = handle.cancel()
            except Exception as exc:
                accepted, error = False, repr(exc)
            self.emit(
                "cancellation",
                job_id=job_id,
                at_s=self.clock(),
                accepted=accepted,
                error=error,
                dispatch=self.jobs.get(job_id),
            )
