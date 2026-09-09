"""Bounded recovery observations; YAML owns fault ordering and verdicts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..contracts import CheckResult, StageHandler, StageOutput
from . import cancel, elastic
from . import status_protocol as status
from .engine_control import ENGINE_NAME

MAX_LOG_BYTES = 8 * 1024 * 1024
EXTRA_METRICS = {
    "observed_completed",
    "target_engine_clean_total",
    "target_family_overlap",
    "target_success_landings",
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
    if (
        name == "success"
        and "records" in frame
        and all("recovery_observed_success" in r for r in frame["records"])
    ):
        return sum(r["recovery_observed_success"] is True for r in frame["records"])
    if name == "observed_completed":
        return sum(o["completed_before_cancel"] is True for o in frame["outcomes"])
    if name == "target_engine_clean_total":
        return metric(frame, "target_inflight") + metric(frame, "target_leaks")
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
    if name == "target_success_landings":
        from .elastic import request_success

        addresses = {v["grpc_addr"] for v in frame["targets"].values()}
        return sum(
            r.get("recovery_observed_success", request_success(r))
            and r["prefill_addr"] in addresses
            for r in frame["records"]
        )
    if name == "target_family_overlap":
        keys = set(frame["keys"])
        total = 0
        for name in frame["targets"]:
            observed = frame["mock"][name]["cache_key_set"]
            if not isinstance(observed, list) or any(
                type(key) is not int for key in observed
            ):
                raise RuntimeError("missing target cache key set")
            total += len(keys & set(observed))
        return total
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
            "keys",
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
    if "keys" in p:
        from .kv import _keys

        _keys(p["keys"])
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
            if (
                frames
                and params.get("until")
                and ctx.clock() >= start + params["duration_s"]
            ):
                break
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
            if "keys" in params:
                frame["keys"] = params["keys"]
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
                params["interval_s"]
                if until
                else min(
                    params["interval_s"], start + params["duration_s"] - ctx.clock()
                )
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
    """Concurrency bounds each request observation, not only its Schedule RPC."""

    def __init__(self, ctx, params):
        # The legacy stream_timeout_s is an observation wait; its actual gRPC
        # call keeps the independent 60s transport deadline.
        self.observation_wait_s = params["stream_timeout_s"]
        self.measure_ttft = params.get("measure_ttft", False)
        super().__init__(ctx, dict(params, stream_timeout_s=60))
        for child in self.children:
            count = params["unique_key_count"]
            if count:
                rid = child.prepared["wire_request_id"]
                child.params["block_keys"] = [
                    rid * 100 + params["unique_key_start"] + j for j in range(count)
                ]
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
                    from .elastic import request_success

                    for entry in child.entries:
                        if entry["thread"] is not None:
                            if self.measure_ttft:
                                first_end = self.ctx.clock() + 15
                                observed_first = None
                                while self.ctx.clock() < first_end:
                                    deadline.check()
                                    if (
                                        entry["record"]["stream"]["first_output_s"]
                                        is not None
                                    ):
                                        observed_first = self.ctx.clock()
                                        break
                                    if entry["done"].is_set():
                                        break
                                    entry["done"].wait(
                                        min(0.002, max(0, first_end - self.ctx.clock()))
                                    )
                                child.update(
                                    entry["record"],
                                    recovery_first_output_observed=observed_first
                                    is not None,
                                    recovery_first_output_observed_s=observed_first,
                                )
                            end = self.ctx.clock() + self.observation_wait_s
                            while not entry["done"].is_set() and self.ctx.clock() < end:
                                deadline.check()
                                entry["done"].wait(min(0.05, end - self.ctx.clock()))
                            if entry["done"].is_set():
                                child._await_consumer(entry, deadline)
                                self._completed_child(child, deadline)
                        child.update(
                            entry["record"],
                            recovery_observed_success=request_success(
                                child.snapshot_records()[0]
                            ),
                            recovery_observed_s=self.ctx.clock(),
                        )
                    child.persist()
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
            "unique_key_start",
            "block_keys",
            "generate_payload",
            "measure_ttft",
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
    p["unique_key_start"] = status._number(
        p.get("unique_key_start", 0), plan, "unique_key_start", 0, 10, True
    )
    if "block_keys" in p:
        from .kv import _keys

        _keys(p["block_keys"])
        if p["unique_key_count"]:
            raise ValueError("explicit and generated keys are exclusive")
    p.setdefault("consume", "immediate")
    if p["consume"] not in {"immediate", "manual"}:
        raise ValueError("invalid recovery consumption mode")
    p.setdefault("measure_ttft", False)
    if type(p["measure_ttft"]) is not bool:
        raise ValueError("measure_ttft must be boolean")
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
        if not record.get("recovery_observed_success", request_success(record)):
            continue
        first, stream_start = (
            record["stream"]["first_output_s"],
            record["stream"]["started_s"],
        )
        if (
            first is None
            or record.get("recovery_first_output_observed") is False
            or (
                "recovery_first_output_observed" not in record
                and first - stream_start > 15
            )
        ):
            continue
        # The old helper records the polling observer's time, not the
        # consumer's earlier receive timestamp. Keep both evidence sources.
        observed_first = record.get("recovery_first_output_observed_s", first)
        if observed_first is None:
            continue
        elapsed = (observed_first - record["schedule"]["started_s"]) * 1000
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


def validate_cache_control(params, plan):
    p = status._validate(
        params,
        plan,
        {"selection", "operation", "keys", "tokens"},
        {"selection", "operation"},
    )
    plan.reference(p["selection"], "snapshot")
    if p["operation"] == "evict":
        if "tokens" in p:
            raise ValueError("tokens do not belong to cache eviction")
        if "keys" in p:
            from .kv import _keys

            _keys(p["keys"])
    elif p["operation"] == "absolute_pressure":
        if "keys" in p or "tokens" not in p:
            raise ValueError("absolute pressure requires tokens only")
        status._number(p["tokens"], plan, "tokens", 0, 2**63 - 1, True)
    else:
        raise ValueError("unsupported recovery cache control")
    return p


def execute_cache_control(ctx, params, deadline):
    from .engine_control import _http

    targets = _snapshot(ctx, params["selection"])["targets"]
    before = status._mock(ctx, deadline)
    epoch, ops = ctx.env_epoch, ctx.ops

    def pressure_restore(limit):
        if ctx.env_epoch != epoch or ctx.ops is not ops:
            raise RuntimeError(
                "pressure cleanup belongs to a different engine environment"
            )
        for name in targets:
            result = _http(
                ops, "set_kv_pressure", limit, {"engine": name, "active_kv_tokens": 0}
            )
            if result.get("status") != "ok" or result.get("engine") != name:
                raise RuntimeError("pressure cleanup lacks target acknowledgement")

    if params["operation"] == "absolute_pressure":
        ctx.add_cleanup("recovery-absolute-pressure", pressure_restore)
    receipts = []
    for name in targets:
        deadline.check()
        if params["operation"] == "evict":
            keys = params.get("keys", before[name]["cache_key_set"])
            if (
                not isinstance(keys, list)
                or len(keys) > 4096
                or any(type(k) is not int for k in keys)
            ):
                raise RuntimeError(
                    "eviction source key set is missing or exceeds budget"
                )
            endpoint, body = "cache_evict", {"engine": name, "keys": keys}
        else:
            endpoint, body = "set_kv_pressure", {
                "engine": name,
                "active_kv_tokens": params["tokens"],
            }
        receipt = {"endpoint": endpoint, "request": body, "started_s": ctx.clock()}
        response = _http(ops, endpoint, deadline, body)
        receipt.update(response=response, ended_s=ctx.clock())
        receipts.append(receipt)
        if response.get("status") != "ok" or response.get("engine") != name:
            raise RuntimeError("cache control lacks target acknowledgement")
    return status._frozen(
        ctx, "recovery-cache-control", {"receipts": receipts, "effect_verified": False}
    )


def validate_pump(params, plan):
    p = status._validate(
        params, plan, {"selection", "duration_s"}, {"selection", "duration_s"}
    )
    plan.reference(p["selection"], "snapshot")
    status._number(p["duration_s"], plan, "duration_s", 0.1, 30)
    return p


def execute_pump(ctx, params, deadline):
    selection = _snapshot(ctx, params["selection"])
    if len(selection["targets"]) != 1:
        raise RuntimeError("accepted pump requires one target")
    name = next(iter(selection["targets"]))
    before = status._count(status._mock(ctx, deadline)[name]["accepted"])
    started, frames = ctx.clock(), []
    after = before
    while ctx.clock() < started + params["duration_s"]:
        deadline.check()
        if len(frames) >= 256:
            raise RuntimeError("accepted pump exceeds 256-attempt budget")
        p = dict(
            count=1,
            concurrency=1,
            input_len=2048,
            output_len=2,
            consume="immediate",
            unique_key_count=1,
            unique_key_start=1,
            generate_payload="match_schedule",
            schedule_timeout_s=30,
            stream_timeout_s=10,
            expected_rpc_statuses=[
                "UNAVAILABLE",
                "UNKNOWN",
                "INTERNAL",
                "DEADLINE_EXCEEDED",
            ],
            expected_stream_statuses=[
                "UNAVAILABLE",
                "UNKNOWN",
                "INTERNAL",
                "DEADLINE_EXCEEDED",
                "CANCELLED",
            ],
        )
        cohort = RecoveryRequests(ctx, p)
        ctx.register_resource(
            "requests", cohort, cleanup=cohort.cleanup, historical=True
        )
        cohort.dispatch(deadline)
        cohort.prove_ended(deadline)
        after = status._count(status._mock(ctx, deadline)[name]["accepted"])
        frames.append(
            {"records": cohort.snapshot_records(), "accepted": after, "at": ctx.clock()}
        )
        if after > before:
            break
        deadline.sleep(0.2)
    if after <= before:
        after = status._count(status._mock(ctx, deadline)[name]["accepted"])
        frames.append({"records": [], "accepted": after, "at": ctx.clock()})
    return StageOutput(
        {"grew": after > before},
        [
            CheckResult(
                "accepted",
                "PASS" if after > before else "FAIL",
                "post-recovery target accepts fresh traffic",
                actual=after - before,
                expected=">0",
                evidence={"attempts": len(frames)},
            )
        ],
        [
            status._artifact(
                ctx, "recovery-pump", {"baseline": before, "frames": frames}
            )
        ],
    )


HANDLERS += [
    StageHandler(
        "recovery_cache_control",
        validate_cache_control,
        execute_cache_control,
        {"snapshot": "snapshot"},
    ),
    StageHandler(
        "recovery_pump",
        validate_pump,
        execute_pump,
        {"grew": "boolean"},
        checks=frozenset({"accepted"}),
    ),
]


def _selection_output(ctx, targets):
    result = status._frozen(ctx, "recovery-selection", {"targets": targets})
    return StageOutput(
        {"selection": result.output["snapshot"], "count": len(targets)},
        artifacts=result.artifacts,
    )


def validate_select_routed(params, plan):
    return cancel.validate_request_ref(params, plan)


def execute_select_routed(ctx, params, deadline):
    cohort = ctx.resource(params["requests"], "requests")
    addresses = {
        r["prefill_addr"]
        for r in cohort.snapshot_records()
        if r["schedule"]["status"] == "OK"
    }
    engines = status._mock(ctx, deadline)
    targets = {}
    for addr in addresses:
        matches = [
            (name, e)
            for name, e in engines.items()
            if e["role"] == "prefill" and e["grpc_addr"] == addr
        ]
        if len(matches) != 1 or not matches[0][1].get("http_addr"):
            raise RuntimeError("routed request lacks one advertised Prefill identity")
        name, e = matches[0]
        targets[name] = {"ip_port": e["http_addr"], "grpc_addr": addr}
    return _selection_output(ctx, targets)


def validate_partition(params, plan):
    p = status._validate(
        params,
        plan,
        {"selection", "snapshot", "stopped"},
        {"selection", "snapshot", "stopped"},
    )
    for key in ("selection", "snapshot"):
        plan.reference(p[key], "snapshot")
    if type(p["stopped"]) is not bool:
        raise ValueError("stopped selector must be boolean")
    return p


def execute_partition(ctx, params, deadline):
    deadline.check()
    targets = _snapshot(ctx, params["selection"])["targets"]
    engines = _snapshot(ctx, params["snapshot"])["frames"][-1]["mock"]
    if any(type(engines[name]["stopped"]) is not bool for name in targets):
        raise RuntimeError("missing stopped-state evidence")
    return _selection_output(
        ctx,
        {
            n: t
            for n, t in targets.items()
            if engines[n]["stopped"] is params["stopped"]
        },
    )


def validate_apply(params, plan):
    p = status._validate(
        params,
        plan,
        {"selection", "operation", "perf", "restore_perf"},
        {"selection", "operation"},
    )
    plan.reference(p["selection"], "snapshot")
    from .engine_control import validate

    control = {k: v for k, v in p.items() if k not in {"selection", "restore_perf"}}
    validate(dict(control, targets=["prefill-0"]), plan)
    if p["operation"] == "set_perf":
        if "restore_perf" not in p:
            raise ValueError("recovery perf change requires its explicit restoration")
        validate(
            {
                "operation": "set_perf",
                "targets": ["prefill-0"],
                "perf": p["restore_perf"],
            },
            plan,
        )
    elif "restore_perf" in p:
        raise ValueError("restore_perf belongs only to set_perf")
    return p


def execute_apply(ctx, params, deadline):
    from .engine_control import execute

    targets = list(_snapshot(ctx, params["selection"])["targets"])
    if not targets:
        return status._frozen(ctx, "empty-recovery-control", {"targets": []})
    epoch, ops = ctx.env_epoch, ctx.ops
    if params["operation"] == "set_perf":

        def restore(limit):
            if ctx.env_epoch != epoch or ctx.ops is not ops:
                raise RuntimeError("perf restore belongs to a different environment")
            execute(
                ctx,
                {
                    "operation": "set_perf",
                    "targets": targets,
                    "perf": params["restore_perf"],
                },
                limit,
            )

        ctx.add_cleanup("recovery-perf-restore", restore)
    return execute(
        ctx,
        {
            k: v
            for k, v in dict(params, targets=targets).items()
            if k not in {"selection", "restore_perf"}
        },
        deadline,
    )


def validate_arm(params, plan):
    p = status._validate(params, plan, {"selection", "mode"}, {"selection", "mode"})
    plan.reference(p["selection"], "snapshot")
    if p["mode"] not in {"first_enqueue", "next_enqueue", "disarm"}:
        raise ValueError("crash arming position must be explicit")
    return p


def execute_arm(ctx, params, deadline):
    from .engine_control import _http
    from .engine_fault import inject as execute_inject

    targets = _snapshot(ctx, params["selection"])["targets"]
    engines = status._mock(ctx, deadline)
    handles, receipts = [], []
    for name in targets:
        if params["mode"] == "disarm":
            response = _http(
                ctx.ops,
                "inject",
                deadline,
                {"engine": name, "type": "crash_after", "enabled": False},
            )
            if response.get("status") != "ok" or response.get("engine") != name:
                raise RuntimeError("crash disarm lacks target acknowledgement")
            receipts.append(response)
        else:
            n = (
                1
                if params["mode"] == "first_enqueue"
                else status._count(engines[name]["rpc_counts"]["enqueue_batch"]) + 1
            )
            result = execute_inject(
                ctx,
                {"targets": [name], "type": "crash_after", "options": {"n": n}},
                deadline,
            )
            handles.append(result.output["fault"])
            receipts.append({"engine": name, "n": n})
    return status._frozen(
        ctx,
        "recovery-crash-control",
        {
            "targets": list(targets),
            "receipts": receipts,
            "fault_handles": handles,
            "effect_verified": False,
        },
    )


def validate_consume(params, plan):
    p = status._validate(params, plan, {"requests", "wait_s"}, {"requests", "wait_s"})
    plan.reference(p["requests"], "requests")
    status._number(p["wait_s"], plan, "wait_s", 0.01, 15)
    return p


def execute_consume(ctx, params, deadline):
    from .elastic import request_success

    cohort = ctx.resource(params["requests"], "requests")
    cohort.join_submission(deadline)
    outcomes = []
    for child in cohort.children:
        for entry in child.entries:
            if entry["record"]["schedule"]["status"] != "OK":
                continue
            cohort.auto_consume = True
            child._start_consumer(entry, deadline.expires_at)
            end = min(deadline.expires_at, ctx.clock() + params["wait_s"])
            while not entry["done"].is_set() and ctx.clock() < end:
                deadline.check()
                entry["done"].wait(min(0.05, end - ctx.clock()))
            if entry["done"].is_set():
                child._await_consumer(entry, deadline)
                cohort._completed_child(child, deadline)
            completed = entry["done"].is_set() and request_success(
                child.snapshot_records()[0]
            )
            row = {
                "request_id": str(entry["record"]["wire_request_id"]),
                "prefill_addr": entry["record"]["prefill_addr"],
                "completed_before_cancel": bool(completed),
                "observed_s": ctx.clock(),
                "cancellations": [],
            }
            if not completed:
                # Borrow a one-child view; the parent remains the sole cleanup owner.
                view = object.__new__(cancel.CancelRequests)
                view.children = [child]
                handle = ctx.register_resource("requests", view, historical=True)
                owners = ["master"] + (
                    [] if entry["response"].enqueued_by_master else ["prefill"]
                )
                for owner in owners:
                    try:
                        result = cancel.execute_rpc(
                            ctx,
                            {
                                "requests": handle,
                                "destination": owner,
                                "expected_rpc_statuses": [],
                            },
                            deadline,
                        )
                        row["cancellations"].append(
                            _snapshot(ctx, result.output["snapshot"])
                        )
                    except Exception as error:
                        code = getattr(error, "code", None)
                        state = (
                            getattr(code(), "name", None) if callable(code) else None
                        )
                        if state not in {
                            "UNAVAILABLE",
                            "UNKNOWN",
                            "INTERNAL",
                            "DEADLINE_EXCEEDED",
                            "NOT_FOUND",
                        }:
                            raise
                        row["cancellations"].append(
                            {"owner": owner, "rpc_status": state}
                        )
                        # EngineOps.cancel stops when Master Cancel raises;
                        # do not add a worker cancellation absent in the old path.
                        break
                # Exit proof is mandatory even when the old observation timed out.
                child.cancel("post_observation_transport_cleanup")
                child._await_consumer(entry, deadline)
                cohort._completed_child(child, deadline)
            outcomes.append(row)
    cohort.prove_ended(deadline)
    return status._frozen(
        ctx,
        "recovery-consumption",
        {"frames": [{"outcomes": outcomes, "records": cohort.snapshot_records()}]},
    )


HANDLERS += [
    StageHandler(
        "recovery_select_routed",
        validate_select_routed,
        execute_select_routed,
        {"selection": "snapshot", "count": "integer"},
    ),
    StageHandler(
        "recovery_partition",
        validate_partition,
        execute_partition,
        {"selection": "snapshot", "count": "integer"},
    ),
    StageHandler(
        "recovery_engine_control",
        validate_apply,
        execute_apply,
        {"snapshot": "snapshot"},
    ),
    StageHandler(
        "recovery_crash_arm", validate_arm, execute_arm, {"snapshot": "snapshot"}
    ),
    StageHandler(
        "recovery_consume", validate_consume, execute_consume, {"snapshot": "snapshot"}
    ),
]


def execute_crash_trigger(ctx, params, deadline):
    targets = _snapshot(ctx, params["selection"])["targets"]
    if not targets:
        raise RuntimeError("crash trigger requires a nonempty target set")
    started, frames, attempts = ctx.clock(), [], []
    while ctx.clock() < started + params["duration_s"]:
        deadline.check()
        engines = status._mock(ctx, deadline)
        if any(type(engines[name]["stopped"]) is not bool for name in targets):
            raise RuntimeError("crash trigger lacks typed stopped-state evidence")
        frames.append({"targets": targets, "mock": engines, "at": ctx.clock()})
        if all(engines[name]["stopped"] for name in targets):
            break
        if len(attempts) >= 256:
            raise RuntimeError("crash trigger exceeds 256-attempt budget")
        p = dict(
            count=1,
            concurrency=1,
            input_len=64,
            output_len=2,
            consume="manual",
            unique_key_count=0,
            unique_key_start=0,
            generate_payload="match_schedule",
            schedule_timeout_s=30,
            stream_timeout_s=60,
            expected_rpc_statuses=[
                "UNAVAILABLE",
                "UNKNOWN",
                "INTERNAL",
                "DEADLINE_EXCEEDED",
            ],
            expected_stream_statuses=[
                "UNAVAILABLE",
                "UNKNOWN",
                "INTERNAL",
                "DEADLINE_EXCEEDED",
                "CANCELLED",
            ],
        )
        cohort = RecoveryRequests(ctx, p)
        ctx.register_resource(
            "requests", cohort, cleanup=cohort.cleanup, historical=True
        )
        cohort.dispatch(deadline)
        attempts.extend(cohort.snapshot_records())
        deadline.sleep(0.2)
    return status._frozen(
        ctx,
        "recovery-crash-trigger",
        {
            "frames": frames,
            "attempts": attempts,
            "started_s": started,
            "ended_s": ctx.clock(),
        },
    )


HANDLERS.append(
    StageHandler(
        "recovery_crash_trigger",
        validate_pump,
        execute_crash_trigger,
        {"snapshot": "snapshot"},
    )
)


def validate_all_targets(params, plan):
    p = status._validate(params, plan, {"snapshot", "metric"}, {"snapshot", "metric"})
    plan.reference(p["snapshot"], "snapshot")
    if p["metric"] != "target_stopped":
        raise ValueError("all-target check requires explicit stopped state")
    return p


def execute_all_targets(ctx, params, deadline):
    deadline.check()
    frame = _snapshot(ctx, params["snapshot"])["frames"][-1]
    expected = len(frame["targets"])
    if not expected:
        raise RuntimeError("all-target stopped evidence cannot be empty")
    actual = metric(frame, "target_stopped")
    return StageOutput(
        {},
        [
            CheckResult(
                "all_stopped",
                "PASS" if actual == expected else "FAIL",
                actual=actual,
                expected=expected,
            )
        ],
    )


def validate_retire_all(params, plan):
    p = status._validate(
        params,
        plan,
        {"selection", "log", "per_target_s"},
        {"selection", "log", "per_target_s"},
    )
    for key in ("selection", "log"):
        plan.reference(p[key], "snapshot")
    status._number(p["per_target_s"], plan, "per_target_s", 0.1, 30)
    return p


def execute_retire_all(ctx, params, deadline):
    targets = _snapshot(ctx, params["selection"])["targets"]
    mark = _snapshot(ctx, params["log"])
    if not targets or targets != mark["targets"]:
        raise RuntimeError("retirement selection and log marker must match")
    samples, verdicts = [], {}
    for name in targets:
        end = ctx.clock() + params["per_target_s"]
        retired = False
        while ctx.clock() < end:
            counts = _log_counts(mark, deadline)
            samples.append(counts)
            retired = counts["counts"][name]["retired"] > 0
            if retired:
                break
            deadline.sleep(0.2)
        verdicts[name] = retired
    passed = all(verdicts.values())
    return StageOutput(
        {},
        [
            CheckResult(
                "retired",
                "PASS" if passed else "FAIL",
                actual=verdicts,
                expected="every selected endpoint retires",
            )
        ],
        [status._artifact(ctx, "recovery-retire-targets", {"samples": samples})],
    )


def validate_residue(params, plan):
    p = status._validate(
        params,
        plan,
        {"requests", "base_residue", "settle_s", "stable_s"},
        {"requests", "base_residue", "settle_s", "stable_s"},
    )
    plan.reference(p["requests"], "requests")
    status._number(p["base_residue"], plan, "base_residue", 0, 64, True)
    status._number(p["settle_s"], plan, "settle_s", 0.1, 30)
    status._number(p["stable_s"], plan, "stable_s", 0.1, 15)
    return p


def execute_residue(ctx, params, deadline):
    cohort = ctx.resource(params["requests"], "requests")
    cohort.join_submission(deadline)
    cohort.prove_ended(deadline)
    records = cohort.snapshot_records()
    if not records:
        raise RuntimeError(
            "residue bound requires complete takeover attempt accounting"
        )
    allowed = (
        params["base_residue"] + len(records) - metric({"records": records}, "success")
    )
    frames, end = [], ctx.clock() + params["settle_s"]
    first = None
    while ctx.clock() < end:
        frame = status._frame(ctx, {"include": ["inflight"]}, deadline)
        frames.append(frame)
        first = metric(frame, "scheduler")
        if first <= allowed:
            break
        deadline.sleep(1)
    if first is None:
        raise RuntimeError("residue observation contains no scheduler sample")
    bounded = first <= allowed
    later = None
    if bounded:
        deadline.sleep(params["stable_s"])
        frame = status._frame(ctx, {"include": ["inflight"]}, deadline)
        frames.append(frame)
        later = metric(frame, "scheduler")
    stable = bounded and later <= first
    return StageOutput(
        {},
        [
            CheckResult(
                "bounded", "PASS" if bounded else "FAIL", actual=first, expected=allowed
            ),
            CheckResult(
                "non_growing",
                "PASS" if stable else "FAIL",
                actual=later,
                expected=f"<= {first}",
            ),
        ],
        [
            status._artifact(
                ctx,
                "recovery-residue",
                {"records": records, "frames": frames, "allowed": allowed},
            )
        ],
    )


HANDLERS += [
    StageHandler(
        "recovery_all_targets_check",
        validate_all_targets,
        execute_all_targets,
        {},
        checks=frozenset({"all_stopped"}),
    ),
    StageHandler(
        "recovery_retire_all",
        validate_retire_all,
        execute_retire_all,
        {},
        checks=frozenset({"retired"}),
    ),
    StageHandler(
        "recovery_residue",
        validate_residue,
        execute_residue,
        {},
        checks=frozenset({"bounded", "non_growing"}),
    ),
]


def execute_flow_stop(ctx, params, deadline):
    """Freeze legacy join20 counters before independently proving all exits."""
    flow = ctx.resource(params["flow"], "flow")
    started = ctx.clock()
    flow._stop.set()
    # Unlike BoundedFlow.stop, reaching the observation cap must not cancel a
    # still-valid RPC. A real pump completion can end this soft wait early.
    flow.done.wait(min(20, deadline.remaining()))
    frozen_at = ctx.clock()
    frozen = flow.snapshot_records()
    returned = [r for r in frozen if r["consumer_exit_s"] is not None]
    observed = elastic.completeness(returned)
    result = dict(
        observation_started_s=started,
        observation_frozen_s=frozen_at,
        observation_cap_s=20,
        observed_total=observed["issued"],
        observed_ok=observed["completed"],
        frozen_records=frozen,
        exit_verified=False,
    )
    # Keep the frozen business evidence even when the later exit proof times out.
    path = status._artifact(ctx, "recovery-flow-stop", result)
    try:
        final = flow.stop(deadline)
        if flow.pump_error:
            raise RuntimeError(flow.pump_error)
        result["exit_verified"] = True
    finally:
        result["records"] = flow.snapshot_records()
        result["final"] = elastic.completeness(result["records"])
        Path(path).write_text(json.dumps(result, indent=2, allow_nan=False))
    evidence = ctx.register_resource("snapshot", result, historical=True)
    return StageOutput(
        output={
            "complete": final["result_complete"],
            "issued": final["issued"],
            "result": evidence,
        },
        artifacts=[path],
    )


def execute_flow_assert(ctx, params, deadline):
    deadline.check()
    result = ctx.resource(params["result"], "snapshot")
    total, ok = result["observed_total"], result["observed_ok"]
    final = result["final"]
    if type(total) is not int or type(ok) is not int or not 0 <= ok <= total:
        raise ValueError("invalid frozen flow counters")
    if (
        type(final.get("result_complete")) is not bool
        or result.get("exit_verified") is not True
    ):
        raise ValueError("missing final flow exit accounting")
    rate = ok / total if total else 0
    return StageOutput(
        checks=[
            CheckResult(
                "nonempty", "PASS" if total else "FAIL", actual=total, expected=">0"
            ),
            CheckResult(
                "complete",
                "PASS" if final["result_complete"] else "FAIL",
                evidence=final,
            ),
            CheckResult(
                "success_rate",
                "PASS" if total and rate >= params["min_success_rate"] else "FAIL",
                actual=rate,
                expected=params["min_success_rate"],
                evidence=result,
            ),
        ]
    )


HANDLERS += [
    StageHandler(
        "recovery_flow_stop",
        elastic._flow_stop_validate,
        execute_flow_stop,
        {"complete": "boolean", "issued": "integer", "result": "snapshot"},
    ),
    StageHandler(
        "recovery_flow_assert",
        elastic._flow_assert_validate,
        execute_flow_assert,
        {},
        checks=frozenset({"nonempty", "complete", "success_rate"}),
    ),
]
