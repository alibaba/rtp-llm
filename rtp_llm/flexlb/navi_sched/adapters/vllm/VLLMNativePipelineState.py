"""Read EngineCore's Native PP state without consuming or changing its futures.

The owner calls record_dispatch/record_push after the corresponding Native
operation, and capture on that same scheduler thread. The retained Native slot
objects contain host timing scalars, not worker tensors or scheduler requests.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import wraps
from types import MethodType


def native_prefill_slot_release(max_tokens, resumable):
    if type(max_tokens) is not int or max_tokens <= 0 or type(resumable) is not bool:
        raise ValueError("invalid Native request completion condition")
    return 1 if max_tokens == 1 and not resumable else 2


@dataclass(frozen=True)
class NativePipelineSpan:
    request_id: int
    computed_tokens: int
    scheduled_tokens: int


@dataclass(frozen=True)
class NativePipelineSlot:
    ordinal: int
    dispatch_age_ns: int
    expected_forward_ms: float
    last_stage_reply_age_ns: int | None
    geometry: tuple[float, ...]
    spans: tuple[NativePipelineSpan, ...]


@dataclass(frozen=True)
class NativePipelineResult:
    scheduler_output_key: int
    dispatch_age_ns: int
    slot_ordinal: int | None
    execute_reply_ready: bool
    result_ready: bool
    load_request_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class NativePipelineSnapshot:
    observed_monotonic_ns: int
    burst: int
    stages: int
    wave_open: bool
    last_real_ordinal: int
    measured_forward_ms: float
    occupied_request_slot_count: int
    slots: tuple[NativePipelineSlot, ...]
    results: tuple[NativePipelineResult, ...]

    def report_age_ns(self, now_ns: int) -> int:
        if now_ns < self.observed_monotonic_ns:
            raise ValueError("report clock precedes Native state capture")
        return now_ns - self.observed_monotonic_ns


@dataclass(frozen=True)
class NativePipelineTimingSample:
    observed_monotonic_ns: int
    burst: int
    slot_ordinal: int
    sample_reply_age_ns: int
    sample_reply_drain: int
    last_stage_reply_drain: int
    slots: tuple[NativePipelineSlot, ...]


class NativePipelineState:
    """Bounded adapter history for a Native LazyPushStepper, owned by rank0."""

    def __init__(self, stepper, execution_policy=None):
        stages = stepper.stages
        if type(stages) is not int or stages <= 1:
            raise ValueError("Native PP state requires more than one stage")
        if stepper.in_flight or stepper.wave.open:
            raise ValueError("bind Native PP state before its first dispatch")
        self._stepper = stepper
        self._stages = stages
        self._owner_thread = threading.get_ident()
        self._history = deque(maxlen=2 * stages)
        self._load_requests_by_output = {}
        # V6D's load_started_at covers query/prepare as well as worker loading.
        # Keep the actual Native metadata-dispatch clock, scoped to its query
        # generation, until that request leaves the loading phase.
        self._load_dispatches = {}
        self._loading_generations = {}
        self._burst = 0
        self.dispatch_active = False
        self.pending_decision = None
        self.failed = False
        self._observation_id = 0
        self._last_step_signature = None
        self._on_timing_sample = None
        self._on_timing_error = None
        self.late_timing_reply_count = 0
        self.failed_timing_reply_count = 0
        self.incomplete_timing_sample_count = 0
        self._projector = None
        if execution_policy is not None:
            from .VLLMExecutionGeometry import TPUStreamingStepProjector

            self._projector = TPUStreamingStepProjector(execution_policy)

    def step_features(self, spans):
        if not spans:
            return (0.0,) * 8
        if self._projector is None:
            raise ValueError("Native PP geometry has no resolved execution policy")
        scheduled = sum(span.scheduled_tokens for span in spans)
        padding = self._projector.project(scheduled)
        return (
            1.0,
            scheduled / 1024.0,
            padding.global_padded_tokens / 1024.0,
            sum(span.computed_tokens * span.scheduled_tokens for span in spans)
            / 1048576.0,
            sum(span.computed_tokens for span in spans) / 1024.0,
            sum(span.scheduled_tokens**2 for span in spans) / 1048576.0,
            float(len(spans)),
            max(span.computed_tokens for span in spans) / 1024.0,
        )

    def record_decision(self, output_key, spans, load_request_ids=()):
        self._check_thread()
        if not self.dispatch_active:
            raise RuntimeError("Native decision recorded outside its dispatch")
        if self.pending_decision is not None:
            raise RuntimeError("Native dispatch decided more than one SchedulerOutput")
        spans = tuple(spans)
        if any(
            v.request_id <= 0 or v.computed_tokens < 0 or v.scheduled_tokens <= 0
            for v in spans
        ):
            raise ValueError("invalid Native SchedulerOutput span")
        if len({v.request_id for v in spans}) != len(spans):
            raise ValueError("duplicate request in Native SchedulerOutput")
        loads = tuple(sorted(load_request_ids)) if load_request_ids else ()
        if any(type(value) is not int or value <= 0 for value in loads) or any(
            loads[index] <= loads[index - 1] for index in range(1, len(loads))
        ):
            raise ValueError("invalid Native KV load request IDs")
        self.pending_decision = (output_key, self.step_features(spans), spans, loads)

    def bind(self, on_change, on_error, on_timing_sample=None):
        """Observe dispatches; an optional callback also publishes changed step state."""
        self._on_timing_sample = on_timing_sample
        self._on_timing_error = on_error
        stepper = self._stepper
        original_dispatch, original_push = stepper._dispatch, stepper._push
        original_step = stepper.core.step_fn

        def observe(call):
            if self.failed:
                return
            try:
                call()
            except Exception:
                self.failed = True
                on_error()

        @wraps(original_dispatch)
        def dispatch(*args, **kwargs):
            self.dispatch_active = True
            self.pending_decision = None
            try:
                result = original_dispatch(*args, **kwargs)
                if result:

                    def record():
                        if self.pending_decision is None:
                            raise ValueError(
                                "Native dispatch has no observed SchedulerOutput"
                            )
                        key, geometry, spans, loads = self.pending_decision
                        if id(stepper.in_flight[-1].scheduler_output) != key:
                            raise ValueError(
                                "Native dispatch used a different SchedulerOutput"
                            )
                        self.record_dispatch(
                            geometry if spans else None, spans, load_request_ids=loads
                        )

                    observe(record)
                return result
            finally:
                self.pending_decision = None
                self.dispatch_active = False

        @wraps(original_push)
        def push(*args, **kwargs):
            result = original_push(*args, **kwargs)
            observe(self.record_push)
            return result

        @wraps(original_step)
        def step(_stepper, *args, **kwargs):
            result = original_step(*args, **kwargs)

            def publish():
                snapshot = self.capture()
                signature = self.signature(snapshot)
                if signature != self._last_step_signature:
                    on_change()
                    self._last_step_signature = signature

            observe(publish)
            return result

        stepper._dispatch, stepper._push = dispatch, push
        if on_change is not None:
            stepper.core.step_fn = MethodType(step, stepper)

    @staticmethod
    def signature(snapshot):
        observed = snapshot.observed_monotonic_ns
        return (
            snapshot.burst,
            snapshot.wave_open,
            snapshot.last_real_ordinal,
            snapshot.measured_forward_ms,
            snapshot.occupied_request_slot_count,
            tuple(
                (
                    slot.ordinal,
                    observed - slot.dispatch_age_ns,
                    slot.expected_forward_ms,
                    (
                        None
                        if slot.last_stage_reply_age_ns is None
                        else observed - slot.last_stage_reply_age_ns
                    ),
                    slot.geometry,
                    slot.spans,
                )
                for slot in snapshot.slots
            ),
            tuple(
                (
                    row.scheduler_output_key,
                    observed - row.dispatch_age_ns,
                    row.slot_ordinal,
                    row.execute_reply_ready,
                    row.result_ready,
                    row.load_request_ids,
                )
                for row in snapshot.results
            ),
        )

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Native PP state must be read on its scheduler thread")

    @staticmethod
    def _finite_nonnegative(value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid Native " + name)
        return value

    def _record_slot(self, slot, geometry, spans):
        self._check_thread()
        stepper = self._stepper
        if stepper.stages != self._stages or not stepper.wave.open:
            raise ValueError("Native PP stage or wave changed during dispatch")
        if not stepper.slots or stepper.slots[-1] is not slot:
            raise ValueError("dispatched Native slot is not the newest slot")
        if any(previous is slot for _, previous, _, _ in self._history):
            raise ValueError("Native PP slot was recorded twice")
        ordinal = stepper.first_slot + len(stepper.slots) - 1
        if ordinal == 0:
            self._history.clear()
            self._burst += 1
        elif not self._history or self._history[-1][0] + 1 != ordinal:
            raise ValueError("Native PP dispatch history has a gap")
        if len(geometry) != 8:
            raise ValueError("Native PP geometry requires eight stage features")
        shape = tuple(self._finite_nonnegative(value, "geometry") for value in geometry)
        self._history.append((ordinal, slot, shape, tuple(spans)))

    def record_dispatch(self, geometry, spans=(), *, load_request_ids=()):
        """Call only after Native _dispatch returned True; None denotes no tokens."""
        self._check_thread()
        stepper = self._stepper
        if not stepper.in_flight:
            raise ValueError("Native dispatch has no outstanding result")
        step = stepper.in_flight[-1]
        if load_request_ids or self._load_requests_by_output:
            live = {id(value.scheduler_output) for value in stepper.in_flight}
            self._load_requests_by_output = {
                key: value
                for key, value in self._load_requests_by_output.items()
                if key in live
            }
            if load_request_ids:
                self._load_requests_by_output[id(step.scheduler_output)] = tuple(
                    load_request_ids
                )
                dispatched_ns = round(
                    self._finite_nonnegative(step.dispatched, "load dispatch time") * 1e9
                )
                for request_id in load_request_ids:
                    generation = self._loading_generations.get(request_id)
                    if generation is not None:
                        self._load_dispatches[request_id] = (generation, dispatched_ns)
            if len(self._load_requests_by_output) > 2 * self._stages:
                raise ValueError(
                    "Native load carrier history exceeds its result capacity"
                )
        if step.slot is None:
            if geometry is not None:
                raise ValueError("Native metadata-only call has compute geometry")
            return
        if geometry is None or geometry[0] != 1:
            raise ValueError("Native compute slot requires real-step geometry")
        self._record_slot(step.slot, geometry, spans)
        if self._on_timing_sample is not None:
            if step.future.done():
                self.late_timing_reply_count += 1
            else:
                burst, ordinal = self._burst, self._history[-1][0]
                # At most four real Futures are outstanding. Each keeps one
                # scalar Native slot whose geometry can leave the eight-slot
                # history before this sample reply arrives.
                prefix = next(
                    (row for row in self._history if row[0] == ordinal - 3), None
                )

                def sampled(future):
                    try:
                        self._record_timing_sample(burst, ordinal, future, prefix)
                    except Exception:
                        if not self.failed:
                            self.failed = True
                            self._on_timing_error()

                step.future.add_done_callback(sampled)

    def _record_timing_sample(self, burst, ordinal, future, prefix=None):
        completed_ns = time.monotonic_ns()
        self._check_thread()
        if self.failed:
            return
        if future.cancelled() or future.exception() is not None:
            self.failed_timing_reply_count += 1
            return
        values = self._slot_values((prefix,)) if prefix is not None else []
        values.extend(self._slot_values())
        by_ordinal = {value[0]: value for value in values}
        required = range(max(0, ordinal - 3), ordinal + 3)
        own = by_ordinal.get(ordinal)
        if (
            burst != self._burst
            or own is None
            or own[2] is None
            or own[2] > completed_ns
            or any(index not in by_ordinal for index in required)
            or any(by_ordinal[index][1] > completed_ns for index in required)
            or ordinal > 0
            and by_ordinal[ordinal - 1][2] is None
            or ordinal > 0
            and by_ordinal[ordinal - 1][2] > completed_ns
        ):
            self.incomplete_timing_sample_count += 1
            return
        slot = next(slot for index, slot, _, _ in self._history if index == ordinal)
        observed_ns = time.monotonic_ns()
        sample = NativePipelineTimingSample(
            observed_monotonic_ns=observed_ns,
            burst=burst,
            slot_ordinal=ordinal,
            sample_reply_age_ns=observed_ns - completed_ns,
            sample_reply_drain=self._stepper.drains,
            last_stage_reply_drain=slot.started_drain,
            slots=tuple(
                self._freeze_slots(
                    [by_ordinal[index] for index in required], observed_ns
                )
            ),
        )
        self._on_timing_sample(sample)

    def _slot_values(self, history=None):
        values = []
        for ordinal, slot, geometry, spans in (
            self._history if history is None else history
        ):
            dispatched_ns = round(
                self._finite_nonnegative(slot.dispatched, "dispatch time") * 1e9
            )
            reply_ns = (
                None
                if slot.started_last is None
                else round(
                    self._finite_nonnegative(slot.started_last, "last-stage reply time")
                    * 1e9
                )
            )
            expected_ms = (
                self._finite_nonnegative(slot.expected, "expected forward time") * 1000
            )
            values.append(
                (ordinal, dispatched_ns, reply_ns, expected_ms, geometry, spans)
            )
        return values

    @staticmethod
    def _freeze_slots(values, observed_ns):
        slots = []
        for ordinal, dispatched_ns, reply_ns, expected_ms, geometry, spans in values:
            if (
                dispatched_ns > observed_ns
                or reply_ns is not None
                and reply_ns > observed_ns
            ):
                raise ValueError("Native slot timestamp is after its observation")
            slots.append(
                NativePipelineSlot(
                    ordinal=ordinal,
                    dispatch_age_ns=observed_ns - dispatched_ns,
                    expected_forward_ms=expected_ms,
                    last_stage_reply_age_ns=(
                        None if reply_ns is None else observed_ns - reply_ns
                    ),
                    geometry=geometry,
                    spans=spans,
                )
            )
        return slots

    def record_push(self):
        self._check_thread()
        if not self._stepper.slots:
            raise ValueError("Native push has no slot")
        self._record_slot(self._stepper.slots[-1], (0.0,) * 8, ())

    def capture(self):
        self._check_thread()
        stepper = self._stepper
        if stepper.stages != self._stages:
            raise ValueError("Native PP stage count changed")
        if (
            len(stepper.slots) > self._stages
            or len(stepper.in_flight) > 2 * self._stages
        ):
            raise ValueError("Native PP state exceeds its scheduler capacity")
        if sum(step.slot is not None for step in stepper.in_flight) > self._stages:
            raise ValueError("Native PP compute results exceed its pipeline capacity")
        scheduler = stepper.core.scheduler
        paused = scheduler.num_waiting_for_streaming_input
        if type(paused) is not int or paused < 0:
            raise ValueError("invalid Native paused streaming request count")
        occupied = len(scheduler.running) + paused
        result_values = []
        by_identity = {id(slot): ordinal for ordinal, slot, _, _ in self._history}
        if stepper.wave.open:
            for index, slot in enumerate(stepper.slots):
                if by_identity.get(id(slot)) != stepper.first_slot + index:
                    raise ValueError("Native PP state has an unrecorded slot")
        for step in stepper.in_flight:
            ordinal = None
            if step.slot is not None:
                ordinal = by_identity.get(id(step.slot))
                if ordinal is None:
                    raise ValueError("Native compute result has no retained geometry")
                if not stepper.wave.open:
                    raise ValueError(
                        "Native closed its wave before compute results drained"
                    )
            execute_ready = bool(step.exec_future.done())
            result_ready = (
                execute_ready
                if step.future is step.exec_future
                else bool(step.future.done())
            )
            dispatch_ns = round(
                self._finite_nonnegative(step.dispatched, "result dispatch time") * 1e9
            )
            result_values.append(
                (
                    id(step.scheduler_output),
                    dispatch_ns,
                    ordinal,
                    execute_ready,
                    result_ready,
                    self._load_requests_by_output.get(id(step.scheduler_output), ()),
                )
            )
        if self._load_requests_by_output:
            live = {row[0] for row in result_values}
            self._load_requests_by_output = {
                key: value
                for key, value in self._load_requests_by_output.items()
                if key in live
            }
        values = []
        if stepper.wave.open:
            if not self._history:
                raise ValueError("Native open wave has no retained slots")
            last = stepper.first_slot + len(stepper.slots) - 1
            if last != self._history[-1][0]:
                raise ValueError("Native PP state has an unrecorded dispatch")
            values = self._slot_values()
        observed_ns = time.monotonic_ns()
        results = []
        for (
            output_key,
            dispatched_ns,
            ordinal,
            execute_ready,
            result_ready,
            load_request_ids,
        ) in result_values:
            if dispatched_ns > observed_ns:
                raise ValueError("Native result was dispatched after its observation")
            results.append(
                NativePipelineResult(
                    scheduler_output_key=output_key,
                    dispatch_age_ns=observed_ns - dispatched_ns,
                    slot_ordinal=ordinal,
                    execute_reply_ready=execute_ready,
                    result_ready=result_ready,
                    load_request_ids=load_request_ids,
                )
            )
        slots = self._freeze_slots(values, observed_ns)
        return NativePipelineSnapshot(
            observed_monotonic_ns=observed_ns,
            burst=self._burst,
            stages=self._stages,
            wave_open=bool(stepper.wave.open),
            last_real_ordinal=stepper.last_real if stepper.wave.open else -1,
            measured_forward_ms=self._finite_nonnegative(
                stepper.measured, "measured forward time"
            )
            * 1000,
            occupied_request_slot_count=occupied,
            slots=tuple(slots),
            results=tuple(results),
        )

    def make_report(self, module, entries):
        """Freeze Native observations once on their owner; no Native references escape."""
        snapshot = self.capture()
        self._observation_id += 1
        span_type = module._VLLMNativePPSpan

        def spans(values):
            return [
                span_type(v.request_id, v.computed_tokens, v.scheduled_tokens)
                for v in values
            ]

        slots = [
            module._VLLMNativePPSlot(
                v.ordinal,
                v.dispatch_age_ns // 1000,
                v.expected_forward_ms,
                (
                    -1
                    if v.last_stage_reply_age_ns is None
                    else v.last_stage_reply_age_ns // 1000
                ),
                v.geometry,
                spans(v.spans),
            )
            for v in snapshot.slots
        ]
        results = [
            module._VLLMNativePPResult(
                v.scheduler_output_key,
                v.dispatch_age_ns // 1000,
                -1 if v.slot_ordinal is None else v.slot_ordinal,
                v.execute_reply_ready,
                v.result_ready,
                v.load_request_ids,
            )
            for v in snapshot.results
        ]
        queue_codes = {
            "running": 0,
            "skipped_waiting": 1,
            "waiting": 2,
            "not_queued": 3,
        }
        phase_codes = {
            "native_running": 1,
            "preempted": 2,
            "kv_loading": 3,
            "lookup_skipped_waiting": 4,
            "query_pending": 5,
            "query_ready_not_consumed": 6,
            "query_consumed_waiting": 7,
            "query_not_started": 8,
            "cache_known_waiting": 9,
            "waiting_cache_state_unknown": 10,
            "request_not_registered": 11,
        }

        def age(start):
            if start is None:
                return -1
            if start > snapshot.observed_monotonic_ns:
                raise ValueError("Native request timestamp is after observation")
            return (snapshot.observed_monotonic_ns - start) // 1000

        loading_generations = {}

        def load_age(row):
            if row.phase != "kv_loading":
                return -1
            loading_generations[row.navi_request_id] = row.query_generation
            dispatched = self._load_dispatches.get(row.navi_request_id)
            if dispatched is None or dispatched[0] != row.query_generation:
                return -1
            if dispatched[1] > snapshot.observed_monotonic_ns:
                raise ValueError("Native load dispatch is after observation")
            return (snapshot.observed_monotonic_ns - dispatched[1]) // 1000

        ordered = sorted(
            entries, key=lambda row: (queue_codes[row.queue], row.queue_index)
        )
        requests = [
            module._VLLMNativePPRequest(
                row.navi_request_id,
                queue_codes[row.queue],
                row.queue_index,
                phase_codes[row.phase],
                row.native_prompt_tokens,
                row.native_computed_tokens,
                row.native_cache_hit_tokens,
                row.native_shared_prefix_boundary,
                age(row.lookup_first_poll_mono_ns),
                load_age(row),
                row.native_prefill_slot_release,
            )
            for row in ordered
        ]
        if loading_generations != self._loading_generations:
            if self._load_dispatches:
                self._load_dispatches = {
                    request_id: value
                    for request_id, value in self._load_dispatches.items()
                    if loading_generations.get(request_id) == value[0]
                }
            self._loading_generations = loading_generations
        pending = self.pending_decision
        decision = (
            None
            if pending is None
            else module._VLLMNativePPDecision(
                pending[0], pending[1], spans(pending[2]), pending[3]
            )
        )
        value = module._VLLMNativePPState(
            self._observation_id,
            snapshot.burst,
            snapshot.wave_open,
            snapshot.last_real_ordinal,
            snapshot.measured_forward_ms,
            slots,
            results,
            requests,
            decision,
            snapshot.occupied_request_slot_count,
        )
        return value, snapshot.observed_monotonic_ns
