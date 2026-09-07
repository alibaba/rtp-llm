"""Bounded recovery observations; YAML owns fault ordering and verdicts."""

from __future__ import annotations

import re
from pathlib import Path

from ..contracts import CheckResult, StageHandler, StageOutput
from . import cancel
from . import status_protocol as status
from .engine_control import ENGINE_NAME

MAX_LOG_BYTES = 8 * 1024 * 1024
EXTRA_METRICS = {
    "topology_mismatch",
    "created",
    "retired",
    "retired_targets",
    "target_prefill_batches",
    "target_prefill_requests",
    "discovered_prefill",
    "target_stopped",
    "target_running",
    "target_inflight",
    "target_held_blocks",
    "target_accepted",
    "target_cache_keys",
    "target_kv_tokens",
    "target_leaks",
    "prefill_lack_mem",
}
METRICS = cancel.METRICS | EXTRA_METRICS


def _snapshot(ctx, ref):
    value = ctx.resource(ref, "snapshot").to_dict()
    if value["env_epoch"] != ctx.env_epoch:
        raise RuntimeError("recovery evidence belongs to a different environment")
    return value


def validate_select(params, plan):
    p = status._validate(params, plan, {"targets"}, {"targets"})
    if not isinstance(p["targets"], list) or not 1 <= len(p["targets"]) <= 8:
        raise ValueError("recovery requires 1..8 explicit engine targets")
    for name in p["targets"]:
        if isinstance(name, dict):
            plan.reference(name, "string")
        elif not isinstance(name, str) or not ENGINE_NAME.fullmatch(name):
            raise ValueError("invalid recovery target")
    return p


def execute_select(ctx, params, deadline):
    engines = status._mock(ctx, deadline)
    names = [ctx.resolve(value) for value in params["targets"]]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate recovery targets")
    targets = {}
    for name in names:
        entry = engines[name]
        if (
            entry["role"] != "prefill"
            or not entry.get("http_addr")
            or not entry.get("grpc_addr")
        ):
            raise RuntimeError("recovery target lacks its advertised Prefill addresses")
        targets[name] = {"ip_port": entry["http_addr"], "grpc_addr": entry["grpc_addr"]}
    result = status._frozen(ctx, "recovery-selection", {"targets": targets})
    return StageOutput(
        {
            "selection": result.output["snapshot"],
            "engine": names[0] if len(names) == 1 else "",
        },
        artifacts=result.artifacts,
    )


def validate_mark(params, plan):
    p = status._validate(params, plan, {"selection"}, {"selection"})
    plan.reference(p["selection"], "snapshot")
    return p


def execute_mark(ctx, params, deadline):
    deadline.check()
    selection = _snapshot(ctx, params["selection"])
    raw_path = getattr(ctx.env, "master_sync_log_path", None)
    if raw_path is None:
        raise RuntimeError("backend has not provided an isolated Master sync-log path")
    path = Path(raw_path).resolve(strict=True)
    if not path.is_relative_to(ctx.artifact_dir.resolve()):
        raise RuntimeError("Master sync log is outside the owned artifact directory")
    stat = path.stat()
    return status._frozen(
        ctx,
        "recovery-log-mark",
        {
            "path": str(path),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "offset": stat.st_size,
            "targets": selection["targets"],
        },
    )


def _log_counts(mark, deadline):
    deadline.check()
    path = Path(mark["path"])
    with path.open("rb") as stream:
        import os

        stat = os.fstat(stream.fileno())
        if (stat.st_dev, stat.st_ino) != (
            mark["device"],
            mark["inode"],
        ) or stat.st_size < mark["offset"]:
            raise RuntimeError(
                "Master sync log rotated or was truncated after its mark"
            )
        if stat.st_size - mark["offset"] > MAX_LOG_BYTES:
            raise RuntimeError("Master sync-log observation exceeds its byte budget")
        stream.seek(mark["offset"])
        raw = stream.read(MAX_LOG_BYTES + 1)
    if len(raw) > MAX_LOG_BYTES:
        raise RuntimeError("Master sync log grew beyond its byte budget")
    rows = raw.decode("utf-8", errors="strict").splitlines()
    result = {}
    for name, target in mark["targets"].items():
        address = re.escape(target["ip_port"])
        created = re.compile(
            r"Created WorkerStatus generation .+ for worker: " + address + r"(?:\s|$)"
        )
        retired = re.compile(r"worker " + address + r" marked dead(?:\s|$)")
        result[name] = {
            "created": sum(bool(created.search(row)) for row in rows),
            "retired": sum(bool(retired.search(row)) for row in rows),
        }
    return {
        "counts": result,
        "path": str(path),
        "offset": mark["offset"],
        "read_bytes": len(raw),
        "inode": stat.st_ino,
    }


def metric(frame, name):
    if name == "topology_mismatch":
        info = frame["info"]["worker_summary"]["PREFILL"]
        expected = frame["expected_prefill"]
        return abs(status._count(info["alive"]) - expected) + abs(
            status._count(info["discovered"]) - expected
        )
    if name in {"created", "retired", "retired_targets"}:
        counts = frame["log"]["counts"]
        if not counts:
            raise RuntimeError("missing target generation evidence")
        if name == "retired_targets":
            return sum(status._count(v["retired"]) > 0 for v in counts.values())
        return sum(status._count(v[name]) for v in counts.values())
    if name == "discovered_prefill":
        return status._count(frame["info"]["worker_summary"]["PREFILL"]["discovered"])
    if name.startswith("target_prefill_"):
        endpoints = frame["inflight"]["prefill_endpoints"]
        field = "inflight_batches" if name.endswith("batches") else "inflight_requests"
        total = 0
        for target in frame["targets"].values():
            matches = [e for e in endpoints if e["ip_port"] == target["ip_port"]]
            if len(matches) != 1:
                raise RuntimeError(
                    "recovered endpoint is missing or ambiguous in Master ledger"
                )
            total += status._count(matches[0][field])
        return total
    if name == "prefill_lack_mem":
        engines = [e for e in frame["mock"].values() if e["role"] == "prefill"]
        if not engines:
            raise RuntimeError("missing Prefill lack-memory census")
        return sum(status._count(e["lack_mem_rejects"]) for e in engines)
    if name.startswith("target_"):
        targets = frame["targets"]
        engines = [frame["mock"][n] for n in targets]
        if not engines:
            raise RuntimeError("empty engine recovery selection")
        field = {
            "target_stopped": "stopped",
            "target_leaks": "leak_detected",
            "target_cache_keys": "cache_key_set",
            "target_kv_tokens": "kv_tokens_used",
        }.get(name, name[len("target_") :])
        values = [e[field] for e in engines]
        if name in {"target_stopped", "target_leaks"}:
            if any(type(v) is not bool for v in values):
                raise RuntimeError("missing typed engine recovery flag")
            return sum(values)
        return sum(status._count(v) for v in values)
    return cancel.metric(frame, name)


def validate_observe(params, plan):
    p = status._validate(
        params,
        plan,
        {
            "selection",
            "log",
            "requests",
            "include",
            "duration_s",
            "interval_s",
            "until",
        },
    )
    for key in ("selection", "log"):
        if key in p:
            plan.reference(p[key], "snapshot")
    if "requests" in p:
        plan.reference(p["requests"], "requests")
    p.setdefault("include", ["inflight", "mock", "info"])
    if (
        not isinstance(p["include"], list)
        or not p["include"]
        or any(
            s not in {"inflight", "mock", "info", "log", "client_records"}
            for s in p["include"]
        )
    ):
        raise ValueError("invalid recovery observation sources")
    if "log" in p["include"] and "log" not in p:
        raise ValueError("log observation requires a fixed byte-offset mark")
    if "client_records" in p["include"] and "requests" not in p:
        raise ValueError("client observation requires a cohort")
    p["duration_s"] = status._number(p.get("duration_s", 0), plan, "duration_s", 0, 180)
    p["interval_s"] = status._number(
        p.get("interval_s", 0.2), plan, "interval_s", 0.01, 5
    )
    if "until" in p:
        u = status._validate(
            p["until"], plan, {"metric", "op", "value"}, {"metric", "op", "value"}
        )
        if u["metric"] not in METRICS or u["op"] not in {"eq", "le", "ge"}:
            raise ValueError("invalid recovery stop predicate")
        status._number(u["value"], plan, "until.value", 0, 2**63 - 1)
    return p


def execute_observe(ctx, params, deadline):
    start, epoch = ctx.clock(), ctx.env_epoch
    frames, size = [], 0
    selection = _snapshot(ctx, params["selection"]) if "selection" in params else None
    mark = _snapshot(ctx, params["log"]) if "log" in params else None
    try:
        while True:
            deadline.check()
            if ctx.env_epoch != epoch:
                raise RuntimeError("environment changed during recovery observation")
            frame = status._frame(
                ctx,
                {
                    "include": [
                        s
                        for s in params["include"]
                        if s not in {"log", "client_records"}
                    ]
                },
                deadline,
            )
            frame["expected_prefill"] = ctx.instance["environment"]["n_prefill"]
            if selection:
                frame["targets"] = selection["targets"]
            if mark:
                frame["log"] = _log_counts(mark, deadline)
            if "requests" in params:
                cohort = ctx.resource(params["requests"], "requests")
                cohort.prove_ended(deadline)
                frame["records"] = cohort.snapshot_records()
            frame["capture_finished_s"] = ctx.clock()
            size += len(str(frame).encode())
            if len(frames) >= 4000 or size > 32 * 1024 * 1024:
                raise RuntimeError("recovery observation budget exceeded")
            frames.append(frame)
            until = params.get("until")
            if until and status._compare(
                metric(frame, until["metric"]), until["op"], until["value"]
            ):
                break
            if ctx.clock() >= start + params["duration_s"]:
                break
            deadline.sleep(
                min(params["interval_s"], start + params["duration_s"] - ctx.clock())
            )
    except Exception as error:
        status._artifact(
            ctx,
            "recovery-incomplete",
            {"frames": frames, "error": repr(error), "env_epoch": epoch},
        )
        raise
    return status._frozen(
        ctx,
        "recovery-observation",
        {"frames": frames, "started_s": start, "ended_s": ctx.clock()},
    )


def validate_check(params, plan):
    p = status._validate(
        params,
        plan,
        {"snapshot", "baseline", "metric", "op", "expected"},
        {"snapshot", "metric", "op", "expected"},
    )
    for key in ("snapshot", "baseline"):
        if key in p:
            plan.reference(p[key], "snapshot")
    if p["metric"] not in METRICS or p["op"] not in {"eq", "le", "ge"}:
        raise ValueError("invalid recovery check")
    status._number(p["expected"], plan, "expected", -(2**63), 2**63 - 1)
    return p


def execute_check(ctx, params, deadline):
    deadline.check()
    source = _snapshot(ctx, params["snapshot"])
    actual = metric(source["frames"][-1], params["metric"])
    if "baseline" in params:
        actual -= metric(
            _snapshot(ctx, params["baseline"])["frames"][-1], params["metric"]
        )
    passed = status._compare(actual, params["op"], params["expected"])
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "contract",
                "PASS" if passed else "FAIL",
                "engine recovery owner contract",
                actual=actual,
                expected=params["expected"],
                evidence={"metric": params["metric"]},
            )
        ],
    )


HANDLERS = [
    StageHandler(
        "recovery_select",
        validate_select,
        execute_select,
        {"selection": "snapshot", "engine": "string"},
    ),
    StageHandler(
        "recovery_log_mark", validate_mark, execute_mark, {"snapshot": "snapshot"}
    ),
    StageHandler(
        "recovery_observe", validate_observe, execute_observe, {"snapshot": "snapshot"}
    ),
    StageHandler(
        "recovery_check",
        validate_check,
        execute_check,
        {"passed": "boolean"},
        checks=frozenset({"contract"}),
    ),
]


class GeneratePayload:
    """Keep a legacy direct-stream payload explicit without altering Schedule."""

    def __init__(self, ops, mode):
        self.ops, self.mode = ops, mode

    def __getattr__(self, name):
        return getattr(self.ops, name)

    def build_generate_input(self, rid, **shape):
        if self.mode == "legacy_default":
            return self.ops.build_generate_input(rid)
        return self.ops.build_generate_input(rid, **shape)


class RecoveryRequests(cancel.CancelRequests):
    """Concurrency bounds the complete request, not only its Schedule RPC."""

    def __init__(self, ctx, params):
        super().__init__(ctx, params)
        for child in self.children:
            count = params["unique_key_count"]
            if count:
                rid = child.prepared["wire_request_id"]
                child.params["block_keys"] = [rid * 100 + j for j in range(count)]
            child.ops = GeneratePayload(child.ops, params["generate_payload"])

    def _worker(self, deadline, done, index):
        try:
            while not self.cancelled.is_set():
                with self.queue_lock:
                    if self.next_index >= len(self.children):
                        break
                    child = self.children[self.next_index]
                    self.next_index += 1
                try:
                    child.submit(deadline)
                except Exception as error:
                    deadline.check()
                    code = getattr(error, "code", None)
                    state = getattr(code(), "name", None) if callable(code) else None
                    if state not in self.params["expected_rpc_statuses"]:
                        raise
                    status._terminal_records(
                        child.snapshot_records(), self.params["expected_rpc_statuses"]
                    )
                if self.auto_consume:
                    for entry in child.entries:
                        child._await_consumer(entry, deadline)
                    self._completed_child(child, deadline)
        except Exception as error:
            with self.queue_lock:
                self.errors.append(error)
        finally:
            with self.queue_lock:
                self.exit_records.append(
                    {"worker": index, "exited_s": self.ctx.clock()}
                )
            done.set()


def validate_prepare(params, plan):
    p = status._validate(
        params,
        plan,
        {
            "count",
            "concurrency",
            "input_len",
            "output_len",
            "consume",
            "schedule_timeout_s",
            "stream_timeout_s",
            "unique_key_count",
            "block_keys",
            "generate_payload",
        },
    )
    for key, default, high in (
        ("count", 1, 64),
        ("concurrency", 1, 10),
        ("input_len", 2048, 1048576),
        ("output_len", 2, 10000),
    ):
        p[key] = status._number(p.get(key, default), plan, key, 1, high, True)
    p["unique_key_count"] = status._number(
        p.get("unique_key_count", 0), plan, "unique_key_count", 0, 10, True
    )
    if "block_keys" in p:
        from .kv import _keys

        _keys(p["block_keys"])
        if p["unique_key_count"]:
            raise ValueError("explicit and generated keys are exclusive")
    p.setdefault("consume", "immediate")
    if p["consume"] not in {"immediate", "manual"}:
        raise ValueError("invalid recovery consumption mode")
    p.setdefault("generate_payload", "match_schedule")
    if p["generate_payload"] not in {"match_schedule", "legacy_default"}:
        raise ValueError("direct Generate payload must be explicit")
    for key, default in (("schedule_timeout_s", 30), ("stream_timeout_s", 15)):
        p[key] = status._number(p.get(key, default), plan, key, 0.05, 60)
    # Known transport failures are retained as failed requests for availability
    # rates. Untyped Python failures and absent terminal evidence remain ERROR.
    p["expected_rpc_statuses"] = [
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "INTERNAL",
        "UNKNOWN",
    ]
    p["expected_stream_statuses"] = [
        *p["expected_rpc_statuses"],
        "CANCELLED",
        "NOT_FOUND",
    ]
    return p


def execute_prepare(ctx, params, deadline):
    deadline.check()
    cohort = RecoveryRequests(ctx, params)
    return StageOutput(
        {
            "requests": ctx.register_resource(
                "requests", cohort, cleanup=cohort.cleanup, historical=True
            ),
            "count": params["count"],
        }
    )


def execute_dispatch(ctx, params, deadline):
    cohort = ctx.resource(params["requests"], "requests")
    if not isinstance(cohort, RecoveryRequests):
        raise RuntimeError("recovery dispatch requires an owned recovery cohort")
    cohort.dispatch(deadline)
    cohort.prove_ended(deadline)
    return StageOutput({"requests": ctx.resolve(params["requests"])})


HANDLERS += [
    StageHandler(
        "recovery_prepare",
        validate_prepare,
        execute_prepare,
        {"requests": "requests", "count": "integer"},
    ),
    StageHandler(
        "recovery_dispatch",
        cancel.validate_request_ref,
        execute_dispatch,
        {"requests": "requests"},
    ),
]


def validate_pause(params, plan):
    p = status._validate(params, plan, {"duration_s"}, {"duration_s"})
    status._number(p["duration_s"], plan, "duration_s", 0.001, 180)
    return p


def execute_pause(ctx, params, deadline):
    start = ctx.clock()
    deadline.sleep(params["duration_s"])
    return StageOutput({"elapsed_s": ctx.clock() - start})


HANDLERS.append(
    StageHandler(
        "recovery_pause", validate_pause, execute_pause, {"elapsed_s": "number"}
    )
)


def _ttft(source):
    from .elastic import request_success

    values = []
    for record in source["frames"][-1]["records"]:
        if not request_success(record):
            continue
        first, stream_start = (
            record["stream"]["first_output_s"],
            record["stream"]["started_s"],
        )
        if first is None or first - stream_start > 15:
            continue
        elapsed = (first - record["schedule"]["started_s"]) * 1000
        if elapsed < 0:
            raise RuntimeError("inverted TTFT timestamps")
        values.append(elapsed)
    return sorted(values)[len(values) // 2] if values else None


def validate_ttft(params, plan):
    p = status._validate(
        params, plan, {"baseline", "recovered"}, {"baseline", "recovered"}
    )
    for key in p:
        plan.reference(p[key], "snapshot")
    return p


def execute_ttft(ctx, params, deadline):
    deadline.check()
    before, after = (
        _ttft(_snapshot(ctx, params[k])) for k in ("baseline", "recovered")
    )
    passed = (
        before is not None
        and after is not None
        and (before <= 0 or after <= 1.5 * before)
    )
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "ttft",
                "PASS" if passed else "FAIL",
                "successful-request index p50 recovery bound",
                actual={"baseline_ms": before, "recovered_ms": after},
                expected="<= 1.5x baseline; missing timing fails",
            )
        ],
    )


HANDLERS.append(
    StageHandler(
        "recovery_ttft_check",
        validate_ttft,
        execute_ttft,
        {"passed": "boolean"},
        checks=frozenset({"ttft"}),
    )
)
