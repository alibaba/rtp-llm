"""Incremental diagnostic evidence, deliberately separate from fit observations."""

import json
import math
import os
import random
import threading
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path


def formal_pressure_validity(
    *,
    execution_valid,
    full_pressure_valid,
    formal_config_valid,
    bidirectional_migration_valid,
):
    """Keep diagnostic gates independent and require explicit evidence for each."""
    gates = dict(
        execution_valid=execution_valid is True,
        full_pressure_valid=full_pressure_valid is True,
        formal_config_valid=formal_config_valid is True,
        bidirectional_migration_valid=bidirectional_migration_valid is True,
    )
    return dict(gates, formal_acceptance_valid=all(gates.values()))


def server_window(start_ns, end_ns, offset_bounds_ns):
    low, high = offset_bounds_ns
    return math.ceil(start_ns - low), math.floor(end_ns - high)


def step_exclusion(step, start_ns, end_ns):
    if not step.get("complete"):
        return "incomplete"
    if step.get("decode_batch", 0):
        return "decode"
    phases, kinds = step.get("member_phases", []), step.get("member_kinds", [])
    if (
        not phases
        or len(phases) != len(kinds)
        or len(phases) != step.get("context_batch")
        or any(value is None for value in phases + kinds)
    ):
        return "unknown_membership"
    if "padding" in kinds or "padding" in phases:
        return "padding"
    if all(value == "support" for value in phases):
        return "support"
    if len(set(phases)) > 1:
        return "mixed_phase"
    if any(value != "measurement" for value in phases):
        return "other_phase"
    if any(value != "formal" for value in kinds):
        return "other_kind"
    if not start_ns <= step["begin_ns"] < step["end_ns"] <= end_ns:
        return "boundary"
    return None


def step_eligible(step, start_ns, end_ns):
    return step_exclusion(step, start_ns, end_ns) is None


def step_gap(previous, current):
    if (previous["session_id"], previous["instance"]) != (
        current["session_id"],
        current["instance"],
    ):
        return None
    if current["step_id"] != previous["step_id"] + 1:
        return None
    wall = current["forward_enter_ns"] - previous["forward_return_ns"]
    cpu = current["forward_enter_cpu_ns"] - previous["forward_return_cpu_ns"]
    if wall < 0 or cpu < 0:
        return None
    return {"wall_ns": wall, "cpu_ns": cpu}


class BoundedDistribution:
    """Algorithm R reservoir with reproducible local PRNG and exact moments."""

    def __init__(self, capacity=4096):
        if capacity < 1:
            raise ValueError("positive reservoir capacity required")
        self.capacity, self.values = capacity, []
        self.count = self.total = 0
        self.maximum = None
        self.random = random.Random(0)

    def add(self, value):
        self.count += 1
        self.total += value
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.values) < self.capacity:
            self.values.append(value)
        else:
            index = self.random.randrange(self.count)
            if index < self.capacity:
                self.values[index] = value

    def summary(self):
        ordered = sorted(self.values)

        def quantile(q):
            return ordered[math.ceil(q * len(ordered)) - 1] if ordered else None

        return dict(
            count=self.count,
            mean=self.total / self.count if self.count else None,
            max=self.maximum,
            p50=quantile(0.5),
            p95=quantile(0.95),
            p99=quantile(0.99) if self.count >= 1000 else None,
            p99_status="estimated" if self.count >= 1000 else "insufficient",
            reservoir_size=len(ordered),
            reservoir_capacity=self.capacity,
            algorithm="algorithm_r_seed_0_nearest_rank",
        )


class StepAnalysis:
    """Bounded association of persisted wire pages with actual dispatch identities."""

    def __init__(self, ranks, session_id, capacity=4096):
        self.ranks, self.session_id, self.capacity = tuple(ranks), session_id, capacity
        self.requests, self.roots = OrderedDict(), OrderedDict()
        self.pending = {rank: OrderedDict() for rank in ranks}
        self.instances, self.last_sequence, self.previous = {}, {}, {}
        self.exclusions = {rank: Counter() for rank in ranks}
        self.eligible = Counter()
        self.distributions = {
            rank: {
                name: BoundedDistribution(capacity)
                for name in (
                    "forward_wall_ns",
                    "forward_cpu_ns",
                    "preparation_wall_ns",
                    "gap_wall_ns",
                    "gap_cpu_ns",
                )
            }
            for rank in ranks
        }
        self.window = None
        self.state_evictions = self.cursor_losses = 0
        self.invalid_ranks = set()

    def _put(self, mapping, key, value):
        mapping[key] = value
        if len(mapping) > self.capacity:
            mapping.popitem(last=False)
            self.state_evictions += 1

    def dispatch(self, event):
        phase = event["phase"]
        kind = event.get(
            "kind", "formal" if phase in ("steady", "measurement", "support") else phase
        )
        self._put(self.requests, event["request_id"], (phase, kind))

    def consume(self, raw):
        pages = raw.get("response", {}).get("ranks", [])
        bounds = raw.get("rank_offset_bounds_s", {})
        if raw.get("diagnostics_error"):
            self.previous.clear()
            self.roots.clear()
            for rank in self.ranks:
                self._flush(rank, final=True)
            return
        # Root membership is decoded before any peer page, independent of wire order.
        for page in sorted(pages, key=lambda p: int(p.get("world_rank", 0))):
            rank = int(page.get("world_rank", 0))
            if rank not in self.pending:
                continue
            instance = page.get("service_instance")
            if rank in self.instances and self.instances[rank] != instance:
                self.invalid_ranks.add(rank)
            self.instances[rank] = instance
            lost = page.get("step_records_lost", False)
            if lost:
                self.cursor_losses += 1
                self._flush(rank, final=True)
                self.previous.pop(rank, None)
                if rank == 0:
                    self.roots.clear()
                    self.previous.clear()
                    for pending_rank in self.ranks:
                        self._flush(pending_rank, final=True)
            if not page.get("step_history_available", False) or page.get("error"):
                self.previous.pop(rank, None)
                continue
            for wire in page.get("steps", []):
                if wire.get("session_id") != self.session_id:
                    self.exclusions[rank]["session_mismatch"] += 1
                    self.previous.pop(rank, None)
                    continue
                identity = int(wire.get("step_id", 0))
                if rank == 0 and rank not in self.invalid_ranks:
                    members = [
                        self.requests.get(v, (None, None))
                        for v in wire.get("source_request_ids", [])
                    ]
                    self._put(
                        self.roots,
                        identity,
                        (
                            int(wire.get("context_count", 0)),
                            int(wire.get("decode_count", 0)),
                            int(wire.get("executed_tokens", 0)),
                            members,
                        ),
                    )
                if not wire.get("completed", False):
                    continue  # Active sequence zero must never move the completed cursor.
                sequence = int(wire.get("sequence", 0))
                if sequence <= self.last_sequence.get(rank, 0):
                    continue
                if self.last_sequence.get(rank, 0) + 1 != sequence:
                    # Do not create adjacency through a pending missing-record boundary.
                    self._flush(rank, final=True)
                    self.previous.pop(rank, None)
                self.last_sequence[rank] = sequence
                step = dict(
                    session_id=self.session_id,
                    instance=instance,
                    step_id=identity,
                    complete=True,
                    context_batch=int(wire.get("context_count", 0)),
                    decode_batch=int(wire.get("decode_count", 0)),
                    executed_tokens=int(wire.get("executed_tokens", 0)),
                    begin_ns=int(wire.get("preparation_begin_ns", 0)),
                    end_ns=int(wire.get("end_ns", 0)),
                    forward_enter_ns=int(wire.get("forward_enter_ns", 0)),
                    forward_return_ns=int(wire.get("forward_return_ns", 0)),
                    forward_enter_cpu_ns=int(wire.get("cpu_start_ns", 0)),
                    forward_return_cpu_ns=int(wire.get("cpu_end_ns", 0)),
                    offset_bounds_ns=tuple(v * 1e9 for v in bounds.get(str(rank), ())),
                )
                if len(self.pending[rank]) >= self.capacity:
                    _, oldest = self.pending[rank].popitem(last=False)
                    self._observe(rank, oldest, unknown=True)
                    self.state_evictions += 1
                self.pending[rank][sequence] = step
        for rank in self.ranks:
            self._flush(rank)

    def _flush(self, rank, final=False):
        pending = self.pending[rank]
        while pending:
            sequence, step = next(iter(pending.items()))
            if step["step_id"] not in self.roots and not final:
                break
            pending.pop(sequence)
            self._observe(rank, step)

    def _observe(self, rank, step, unknown=False):
        root = self.roots.get(step["step_id"])
        members = []
        if (
            not unknown
            and rank not in self.invalid_ranks
            and 0 not in self.invalid_ranks
            and root
            and root[:3]
            == (step["context_batch"], step["decode_batch"], step["executed_tokens"])
        ):
            members = root[3]
        step["member_phases"] = [v[0] for v in members]
        step["member_kinds"] = [v[1] for v in members]
        window = (0, 0)
        if self.window and len(step["offset_bounds_ns"]) == 2:
            window = server_window(*self.window, step["offset_bounds_ns"])
        reason = step_exclusion(step, *window)
        if reason is None and not (
            0
            < step["begin_ns"]
            <= step["forward_enter_ns"]
            <= step["forward_return_ns"]
            <= step["end_ns"]
            and 0 <= step["forward_enter_cpu_ns"] <= step["forward_return_cpu_ns"]
        ):
            reason = "invalid_clocks"
        if reason:
            self.exclusions[rank][reason] += 1
            self.previous.pop(rank, None)
            return
        self.eligible[rank] += 1
        dist = self.distributions[rank]
        dist["forward_wall_ns"].add(
            step["forward_return_ns"] - step["forward_enter_ns"]
        )
        dist["forward_cpu_ns"].add(
            step["forward_return_cpu_ns"] - step["forward_enter_cpu_ns"]
        )
        dist["preparation_wall_ns"].add(step["forward_enter_ns"] - step["begin_ns"])
        previous = self.previous.get(rank)
        gap = step_gap(previous, step) if previous else None
        if gap:
            dist["gap_wall_ns"].add(gap["wall_ns"])
            dist["gap_cpu_ns"].add(gap["cpu_ns"])
        self.previous[rank] = step

    def summary(self, final=True):
        if final:
            for rank in self.ranks:
                self._flush(rank, final=True)
        return dict(
            state_capacity=self.capacity,
            state_evictions=self.state_evictions,
            cursor_losses=self.cursor_losses,
            ranks={
                str(rank): dict(
                    eligible=self.eligible[rank],
                    exclusions=dict(self.exclusions[rank]),
                    **{
                        name: value.summary()
                        for name, value in self.distributions[rank].items()
                    },
                )
                for rank in self.ranks
            },
        )


class RequestWindow:
    """Terminal dispatched cohort counters; callers submit each request once."""

    def __init__(self, start_s, end_s):
        if not all(math.isfinite(v) for v in (start_s, end_s)) or end_s <= start_s:
            raise ValueError("invalid measurement window")
        self.start, self.end = start_s, end_s
        self.members = self.crossing = self.completed_inside = 0
        self.successes = self.cancelled = 0

    def observe(self, record):
        dispatch_s = record.get("dispatch_s")
        if (
            record["kind"] != "formal"
            or record["phase"] != "measurement"
            or record.get("terminal") == "not_dispatched"
            or dispatch_s is None
        ):
            return
        if not self.start <= dispatch_s < self.end:
            raise ValueError("measurement dispatch outside window")
        self.members += 1
        completed = record.get("completed_s")
        if completed is not None and completed >= self.end:
            self.crossing += 1
        elif completed is not None and self.start <= completed < self.end:
            self.completed_inside += 1
            self.successes += record.get("success") is True
        self.cancelled += record.get("terminal") == "cancellation"

    def summary(self):
        return dict(
            measurement_requests=self.members,
            crossing_requests=self.crossing,
            in_window_completions=self.completed_inside,
            request_throughput=self.completed_inside / (self.end - self.start),
            in_window_successes=self.successes,
            success_throughput=self.successes / (self.end - self.start),
            cancelled_requests=self.cancelled,
        )


class PressureWriter:
    def __init__(self, directory: Path, manifest: dict):
        self.directory = Path(directory) / "pressure"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.streams = {}
        self.closed = False
        self.manifest = dict(manifest, complete=False)
        self._atomic("manifest.json", self.manifest)

    def _atomic(self, name, value):
        encoded = json.dumps(value, allow_nan=False, indent=2) + "\n"
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)

    def append(self, kind: str, record: dict):
        if kind not in {"requests", "cache_samples", "steps"}:
            raise ValueError("unsupported pressure evidence stream")
        encoded = json.dumps(record, allow_nan=False) + "\n"
        with self.lock:
            if self.closed:
                raise ValueError("pressure writer closed")
            if kind not in self.streams:
                self.streams[kind] = (self.directory / (kind + ".jsonl")).open(
                    "a", encoding="utf-8"
                )
            stream = self.streams[kind]
            stream.write(encoded)
            # Each callback is durable to process termination; no queued records.
            stream.flush()

    def finish(self, summary: dict):
        with self.lock:
            if self.closed:
                raise ValueError("pressure writer closed")
            self._atomic("summary.json", summary)
            self.manifest.update(
                complete=True,
                validity=summary.get("validity"),
                validity_reasons=summary.get("validity_reasons"),
                classification=summary.get("classification"),
            )
            self._atomic("manifest.json", self.manifest)

    def close(self):
        with self.lock:
            for stream in self.streams.values():
                stream.flush()
                stream.close()
            self.streams.clear()
            self.closed = True


# Missing provenance remains unknown.
MANIFEST_PROJECTION = (
    "code_hash",
    "profile_hash",
    "grid_hash",
    "resolved_settings",
    "input_identity",
    "epoch",
    "clock_bounds",
    "control_times",
)

NONFORMAL_REASON_ORDER = (
    "manual_seed_count",
    "client_inflight_below_16",
    "synthetic_test_block_capacity",
    "nonstandard_fill_target",
    "explicit_smoke",
    "diagnostic_only",
    "grid_reuse_below_32768",
)


def seed_plan_reasons(plan):
    count = plan.get("seed_count")
    return [] if type(count) is int and count > 0 else ["seed_plan_count_invalid"]


def seed_outcome_reasons(seed, plan):
    """Validate terminal accounting against both the frozen and accepted work."""
    plan_reasons = seed_plan_reasons(plan)
    if seed is None:
        return plan_reasons + ["seed_outcome_missing"]
    names = ("planned", "submitted", "completed", "failed", "cancelled", "pending")
    if not isinstance(seed, dict):
        return plan_reasons + ["seed_outcome_invalid"]
    valid_counts = {
        name for name in names if type(seed.get(name)) is int and seed[name] >= 0
    }
    reasons = plan_reasons.copy()
    if len(valid_counts) != len(names):
        reasons.append("seed_outcome_invalid")
    headroom_limited = (
        not plan_reasons
        and seed.get("stop_reason") == "host_max_ratio"
        and type(seed.get("configured_limit")) is int
        and seed["configured_limit"] == plan["seed_count"]
        and type(seed.get("effective_seed_count")) is int
        and "planned" in valid_counts
        and seed["effective_seed_count"] == seed["planned"]
        and seed["planned"] <= seed["configured_limit"]
        and isinstance(seed.get("stop_host_pools"), list)
    )
    if (
        not plan_reasons
        and "planned" in valid_counts
        and seed["planned"] != plan["seed_count"]
        and not headroom_limited
    ):
        reasons.append("seed_plan_outcome_mismatch")
    if {"planned", "submitted"} <= valid_counts and seed["planned"] != seed[
        "submitted"
    ]:
        reasons.append("seed_planned_submitted_mismatch")
    if {"planned", "completed"} <= valid_counts and seed["planned"] != seed[
        "completed"
    ]:
        reasons.append("seed_planned_completed_mismatch")
    for name in ("failed", "cancelled", "pending"):
        if name in valid_counts and seed[name]:
            reasons.append("seed_" + name)
    return reasons


def assessment_reasons(event, key):
    assessment = event.get(key)
    if not isinstance(assessment, dict):
        return [key + "_evidence_missing"]
    reason_field = "epoch_reasons" if key == "pressure" else "reasons"
    reasons = list(assessment.get(reason_field, []))
    if assessment.get("valid") is not True and not reasons:
        reasons.append(key + "_invalid")
    return reasons


class PressureEvidence:
    """Serialized supervisor sink retaining only outstanding dispatched jobs."""

    def __init__(
        self,
        writer,
        downstream=None,
        *,
        profile_requested=False,
        target_ranks=(),
        session_id=None,
    ):
        self.writer, self.downstream = writer, downstream
        self.profile_requested = profile_requested
        self.pending = {}
        self.window = None
        self.duration = None
        self.frozen_processed = False
        self.execution_failed = False
        self.execution_reasons = []
        previous = writer.manifest.get("classification", {})
        self.nonformal_reasons = list(previous.get("nonformal_reasons", []))
        self.nonformal_reasons.extend(
            writer.manifest.get("formal_candidate_reasons", [])
        )
        self.formal_config_eligible = (
            previous.get("formal_config_valid") is not False
            and writer.manifest.get("formal_candidate") is not False
        )
        if "seed_plan" in writer.manifest:
            self._classification(writer.manifest["seed_plan"])
        self.lock = threading.RLock()
        self.steps = StepAnalysis(target_ranks, session_id)

    def _execution_failure(self, reason):
        self.execution_failed = True
        if reason not in self.execution_reasons:
            self.execution_reasons.append(reason)

    def _classification(self, plan):
        self.formal_config_eligible &= plan.get("formal_config_valid") is True
        self.nonformal_reasons.extend(plan.get("nonformal_reasons", []))
        self.nonformal_reasons.extend(seed_plan_reasons(plan))
        # Retain frozen inputs even if an upstream screening reason was omitted.
        for condition, reason in (
            (plan.get("source") == "manual", "manual_seed_count"),
            (
                "client_inflight" in plan and plan["client_inflight"] < 16,
                "client_inflight_below_16",
            ),
            (plan.get("test_block_num", 0) > 0, "synthetic_test_block_capacity"),
            (
                "fill_target_tokens" in plan and plan["fill_target_tokens"] != 102400,
                "nonstandard_fill_target",
            ),
            (plan.get("explicit_smoke") is True, "explicit_smoke"),
            (plan.get("diagnostic_only") is True, "diagnostic_only"),
            (
                "grid_reuse_tokens" in plan and plan["grid_reuse_tokens"] < 32768,
                "grid_reuse_below_32768",
            ),
        ):
            if condition:
                self.nonformal_reasons.append(reason)
        reasons = set(self.nonformal_reasons)
        ordered = [reason for reason in NONFORMAL_REASON_ORDER if reason in reasons]
        ordered.extend(sorted(reasons - set(NONFORMAL_REASON_ORDER)))
        if not self.formal_config_eligible and not ordered:
            ordered.append("formal_configuration_invalid")
        self.nonformal_reasons = ordered
        return dict(
            formal_config_valid=self.formal_config_eligible and not ordered,
            nonformal_reasons=ordered.copy(),
            inputs={
                name: deepcopy(plan.get(name))
                for name in (
                    "source",
                    "seed_count",
                    "client_inflight",
                    "test_block_num",
                    "fill_target_tokens",
                    "grid_reuse_tokens",
                    "explicit_smoke",
                    "diagnostic_only",
                )
            },
        )

    def abort(self, error):
        with self.lock:
            self.writer.append("steps", dict(event="error", error=error))
            for record, cancellation in self.pending.values():
                self._terminal(record, "cancellation", cancellation or {})
            self.pending.clear()

    def _terminal(self, record, terminal, event):
        row = dict(record)
        row.setdefault(
            "kind",
            (
                "formal"
                if row["phase"] in {"steady", "measurement", "support"}
                else row["phase"]
            ),
        )
        row["completed_s"] = event.get("completed_s")
        result = event.get("result", {})
        row.update(result)
        confirmed = (
            (event.get("dispatch") or {}) if terminal == "cancellation" else event
        )
        row["dispatch_s"] = confirmed.get("dispatched_s", row.pop("dispatched_s", None))
        if "schedule_lag_s" in confirmed:
            row["schedule_lag_s"] = confirmed["schedule_lag_s"]
        if terminal == "completion" and result.get("not_dispatched") is True:
            terminal = "not_dispatched"
            row.setdefault("attempted_dispatch_s", row["dispatch_s"])
            row["dispatch_s"] = row["completed_s"] = None
            row.pop("schedule_lag_s", None)
            self.steps.requests.pop(record["request_id"], None)
        row["terminal"] = terminal
        row["success"] = terminal == "completion" and row.get("success") is True
        if not row["success"]:
            self._execution_failure(
                {
                    "cancellation": "request_cancelled",
                    "not_dispatched": "request_not_dispatched",
                }.get(terminal, "request_failed")
            )
        if terminal == "completion" and "input_len" in record:
            if (
                result.get("input_len") != record["input_len"]
                or result.get("output_len") != 1
            ):
                self._execution_failure("request_shape_mismatch")
        if terminal == "cancellation":
            row.update(
                cancel_requested_s=event.get("at_s"),
                transport_cancel_accepted=event.get("accepted"),
            )
        self.writer.append("requests", row)
        if self.window:
            self.window.observe(row)

    def __call__(self, event):
        with self.lock:
            kind = event["event"]
            self.writer.append(
                "cache_samples" if kind == "pressure" else "steps", event
            )
            if kind == "frozen":
                plan = deepcopy(event["seed_plan"])
                self.duration = event["config"]["measure_s"]
                if event["config"].get("evidence_policy") in {
                    "mock_memory",
                    "tree_then_mock_memory",
                    "tree_then_mock_disk",
                    "tree_then_mock_remote",
                }:
                    self.nonformal_reasons.append("diagnostic_only")
                    self.writer.manifest["formal_candidate"] = False
                    self.writer.manifest["formal_candidate_reasons"] = list(
                        dict.fromkeys(
                            self.writer.manifest.get("formal_candidate_reasons", [])
                            + ["diagnostic_only"]
                        )
                    )
                self.writer.manifest.update(
                    session_id=event["session_id"],
                    frozen=deepcopy(event),
                    seed_plan=plan,
                    classification=self._classification(plan),
                )
                self.writer._atomic("manifest.json", self.writer.manifest)
                self.frozen_processed = True
            elif kind == "phase" and event["phase"] == "measurement":
                self.window = RequestWindow(
                    event["at_s"], event["at_s"] + self.duration
                )
                self.steps.window = (self.window.start * 1e9, self.window.end * 1e9)
            elif kind == "pressure":
                for page in event["raw"].get("response", {}).get("ranks", []):
                    for record in page.get("steps", []):
                        self.writer.append(
                            "steps",
                            dict(
                                event="step_record",
                                world_rank=page.get("world_rank", 0),
                                service_instance=page.get("service_instance"),
                                record=record,
                                offset_bounds_s=event["raw"]
                                .get("rank_offset_bounds_s", {})
                                .get(str(page.get("world_rank", 0))),
                            ),
                        )
                self.steps.consume(event["raw"])
            elif kind == "control":
                controls = self.writer.manifest.get("control_times")
                if controls is None:
                    controls = self.writer.manifest["control_times"] = {}
                controls.setdefault("diagnostics", []).append(event)
                self.writer._atomic("manifest.json", self.writer.manifest)
                if event.get("result") and not event["result"].get("success"):
                    self._execution_failure(
                        event.get("reason") or "diagnostics_control_failed"
                    )
            elif kind == "dispatch":
                self.pending[event["job_id"]] = (event, None)
                self.steps.dispatch(event)
            elif kind == "completion":
                entry = self.pending.pop(event["job_id"], None)
                if entry:
                    self._terminal(entry[0], kind, event)
                elif event.get("result", {}).get("success") is not True:
                    self._execution_failure("request_failed")
            elif kind == "cancellation":
                entry = self.pending.get(event["job_id"])
                if entry:
                    self.pending[event["job_id"]] = (entry[0], event)
            elif kind == "error":
                self._execution_failure(
                    event.get("reason") or event.get("error") or "supervisor_error"
                )
            elif kind == "finished":
                for record, cancellation in self.pending.values():
                    self._terminal(record, "cancellation", cancellation or {})
                self.pending.clear()
                failure = event.get("failure")
                profile_failure = bool(failure and failure.startswith("profile"))
                if not self.frozen_processed:
                    self._execution_failure("frozen_event_missing")
                plan = self.writer.manifest.get("seed_plan", {})
                seed = deepcopy(event.get("seed"))
                seed_reasons = seed_outcome_reasons(seed, plan)
                classification = self._classification(plan)
                execution_reasons = self.execution_reasons.copy()
                if failure and not profile_failure and failure not in execution_reasons:
                    execution_reasons.append(failure)
                execution_reasons.extend(seed_reasons)
                validity = formal_pressure_validity(
                    execution_valid=(
                        not self.execution_failed
                        and not seed_reasons
                        and (not failure or profile_failure)
                    ),
                    full_pressure_valid=event.get("pressure", {}).get("valid"),
                    formal_config_valid=classification["formal_config_valid"],
                    bidirectional_migration_valid=event.get("migration", {}).get(
                        "valid"
                    ),
                )
                validity_reasons = dict(
                    execution_valid=execution_reasons,
                    full_pressure_valid=assessment_reasons(event, "pressure"),
                    formal_config_valid=classification["nonformal_reasons"],
                    bidirectional_migration_valid=assessment_reasons(
                        event, "migration"
                    ),
                )
                validity_reasons["formal_acceptance_valid"] = list(
                    dict.fromkeys(
                        reason
                        for reasons in validity_reasons.values()
                        for reason in reasons
                    )
                )
                self.writer.manifest["seed_outcome"] = seed
                summary = dict(
                    **{
                        key: deepcopy(self.writer.manifest[key])
                        for key in (
                            *MANIFEST_PROJECTION,
                            "startup_pressure",
                            "startup_targets",
                            "seed_plan",
                            "frozen",
                        )
                        if key in self.writer.manifest
                    },
                    execution_mode="full_pressure_diagnostic_v1",
                    metrics=[],
                    seed_outcome=seed,
                    classification=classification,
                    validity_reasons=validity_reasons,
                    baseline_steps=self.steps.summary(),
                    window=self.window.summary() if self.window else None,
                    result=event,
                    validity=dict(
                        **validity,
                        pressure_valid=dict(
                            measurement=event.get("pressure", {}).get("valid"),
                            # Support coverage cannot certify the actual capture window.
                            profile=event.get("profile", {}).get("pressure_valid"),
                            support=event.get("support_pressure", {}).get("valid"),
                        ),
                        reuse_valid=None,
                        trace_valid=event.get("profile", {}).get("trace_valid"),
                        trace_applicability=(
                            ("validated" if event.get("profile") else "pending")
                            if self.profile_requested
                            else "not_applicable"
                        ),
                    ),
                )
                self.writer.finish(summary)
            if self.downstream:
                self.downstream(event)
