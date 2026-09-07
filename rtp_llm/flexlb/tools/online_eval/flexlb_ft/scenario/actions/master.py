"""Owned master lifecycle stages; process recovery is distinct from request success."""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass

from ..contracts import CheckResult, StageHandler, StageOutput


def _params(value, plan, allowed, required=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(allowed)
        or set(required) - set(value)
    ):
        raise ValueError(f"{plan.path}: invalid master parameters")
    return copy.deepcopy(value)


def _target(params):
    params.setdefault("target", "single")
    if params["target"] not in ("single", "A", "B"):
        raise ValueError("master target must be single, A, or B")
    return params


def _layout(params, plan):
    layout = getattr(plan, "environment", {}).get("master_layout", "single")
    if (params["target"] == "single") != (layout == "single"):
        raise ValueError("master target does not match the compiled master layout")


def _process(ctx, target):
    if target == "single":
        if getattr(ctx.env, "masters", {}):
            raise ValueError("single target cannot address a dual-master environment")
        proc = ctx.env.master
    else:
        if set(getattr(ctx.env, "master_specs", {})) != {"A", "B"}:
            raise ValueError(
                "A/B target requires an actual dual-standalone environment"
            )
        proc = ctx.env.masters.get(target)
    if proc is None or not proc.alive():
        raise RuntimeError(f"owned master {target} is not running")
    return proc


@dataclass
class MasterFault:
    target: str
    mode: str
    process: object
    started_s: float
    restored: bool = False

    def cleanup(self, deadline):
        # Keep the original Popen even after EnvManager removes its registry
        # slot. Never signal a PID that might have been reused after it exited.
        if self.mode == "freeze" and not self.restored and self.process.alive():
            self.process.unfreeze()
            self.restored = True
        if self.mode == "kill":
            self.process.proc.wait(timeout=deadline.remaining())


def _fault_validate(params, plan):
    p = _target(_params(params, plan, {"target", "mode"}, {"mode"}))
    if p["mode"] not in ("kill", "freeze"):
        raise ValueError("master fault mode must be kill or freeze")
    _layout(p, plan)
    return p


def _fault(ctx, params, deadline):
    deadline.check()
    process = _process(ctx, params["target"])
    fault = MasterFault(params["target"], params["mode"], process, ctx.clock())
    handle = ctx.register_resource("master_fault", fault, fault.cleanup)
    manager = ctx.backend.manager
    if fault.mode == "freeze":
        process.freeze()
    elif fault.target == "single":
        manager.kill_master9(ctx.env)
    else:
        manager.kill_master9_instance(ctx.env, fault.target)
    deadline.check()
    return StageOutput(
        {"fault": handle, "pid": process.pid, "started_s": fault.started_s}
    )


def _restore_validate(params, plan):
    p = _params(params, plan, {"fault"}, {"fault"})
    plan.reference(p["fault"], "master_fault")
    return p


def _restore(ctx, params, deadline):
    fault = ctx.resource(params["fault"], "master_fault")
    if fault.restored:
        raise ValueError("master fault has already been restored")
    deadline.check()
    if fault.mode == "freeze":
        current = _process(ctx, fault.target)
        if current is not fault.process:
            raise RuntimeError("master process changed while frozen")
        current.unfreeze()
    else:
        # Reap before reusing the lane, retaining the fault resource if start
        # fails; backend teardown owns any partially started replacement.
        fault.process.proc.wait(timeout=deadline.remaining())
        if fault.target == "single":
            current = ctx.backend.manager.start_master(ctx.env)
        else:
            current = ctx.backend.manager.restart_master_instance(ctx.env, fault.target)
    fault.restored = True
    deadline.check()
    same = current.pid == fault.process.pid
    return StageOutput(
        {"pid": current.pid, "restored_s": ctx.clock()},
        [
            CheckResult(
                "process_identity",
                "PASS" if same == (fault.mode == "freeze") else "FAIL",
                actual={"before": fault.process.pid, "after": current.pid},
                expected="same process" if fault.mode == "freeze" else "new process",
            )
        ],
    )


def _master_json(ctx, target, path, deadline, post=False):
    _process(ctx, target)
    port = (
        ctx.env.master_http_port
        if target == "single"
        else ctx.env.master_specs[target].http_port
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=b"{}" if post else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=min(5, deadline.remaining())
    ) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("master response is not a JSON object")
    return data


def _ready_validate(params, plan):
    p = _target(_params(params, plan, {"target", "inflight_zero"}))
    p.setdefault("inflight_zero", True)
    if type(p["inflight_zero"]) is not bool:
        raise ValueError("inflight_zero must be boolean")
    _layout(p, plan)
    return p


def _endpoint_loads(data):
    loads = {}
    for side in ("prefill", "decode"):
        rows = data[side + "_endpoints"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing endpoint ledger observations")
        values = []
        for row in rows:
            key = "inflight_batches" if side == "prefill" else "total_load"
            if side == "decode" and "total_load" not in row:
                key = "inflight_requests"
            value = row[key]
            if isinstance(value, list) and side == "prefill":
                value = len(value)
            if type(value) is not int or value < 0:
                raise ValueError("invalid endpoint ledger count")
            values.append(value)
        loads[side] = values
    return loads


def _ready(ctx, params, deadline):
    expected = {"PREFILL": ctx.env.spec.n_prefill, "DECODE": ctx.env.spec.n_decode}
    samples = []
    artifact = ctx.artifact_dir / f"master-readiness-{len(ctx._resources)}.json"
    # Endpoint failures/missing fields are errors, never observations of zero.
    # Valid but not-yet-converged samples are retained until the stage deadline.
    try:
        while True:
            deadline.check()
            info = _master_json(
                ctx, params["target"], "/rtp_llm/master/info", deadline, True
            )
            inflight = _master_json(
                ctx, params["target"], "/rtp_llm/inflight_status", deadline
            )
            summary = info["worker_summary"]
            count = inflight["scheduler_inflight"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid scheduler inflight count")
            counts = {}
            for role in expected:
                row = summary[role]
                values = (row["discovered"], row["alive"])
                if any(type(v) is not int or v < 0 for v in values):
                    raise ValueError("invalid topology counts")
                counts[role] = values
            samples.append(
                dict(
                    time_s=ctx.clock(),
                    topology=counts,
                    scheduler_inflight=count,
                    ready=info.get("ready"),
                )
            )
            converged = info.get("ready") is True and all(
                values == (expected[role], expected[role])
                for role, values in counts.items()
            )
            endpoint_loads = (
                _endpoint_loads(inflight) if params["inflight_zero"] else {}
            )
            samples[-1]["endpoint_loads"] = endpoint_loads
            clean = not params["inflight_zero"] or (
                count == 0
                and not any(v for values in endpoint_loads.values() for v in values)
            )
            if converged and clean:
                break
            deadline.sleep(0.5)
    finally:
        artifact.write_text(json.dumps(samples, indent=2) + "\n")
    handle = ctx.register_resource("snapshot", samples, historical=True)
    return StageOutput(
        {"snapshot": handle},
        [
            CheckResult("topology", "PASS", actual=counts, expected=expected),
            CheckResult(
                "inflight",
                "PASS",
                actual=count,
                expected=0 if params["inflight_zero"] else "observed",
            ),
        ],
        [str(artifact)],
    )


HANDLERS = [
    StageHandler(
        "master_fault",
        _fault_validate,
        _fault,
        {"fault": "master_fault", "pid": "integer", "started_s": "number"},
    ),
    StageHandler(
        "master_restore",
        _restore_validate,
        _restore,
        {"pid": "integer", "restored_s": "number"},
        checks=frozenset({"process_identity"}),
    ),
    StageHandler(
        "master_ready",
        _ready_validate,
        _ready,
        {"snapshot": "snapshot"},
        checks=frozenset({"topology", "inflight"}),
    ),
]


# HA traffic is an actual Java client subprocess. These stages do not call a
# legacy case or its ambient HA skip gate.
def _ha_validate(params, plan):
    p = _params(params, plan, {"targets", "duration_s", "timeout_ms", "fallback"})
    environment = getattr(plan, "environment", {})
    if environment.get("master_layout", "single") != "dual_standalone":
        raise ValueError("HA traffic requires compiled dual_standalone environment")
    p.setdefault("targets", ["A", "B"])
    if p["targets"] not in (["A", "B"], ["B", "A"]):
        raise ValueError("HA targets must explicitly order A and B")
    for key, default, lower, upper in (
        ("duration_s", 60, 1, 180),
        ("timeout_ms", 30000, 100, 30000),
    ):
        p.setdefault(key, default)
        if type(p[key]) is not int or not lower <= p[key] <= upper:
            raise ValueError(f"{key} is outside the bounded HA range")
    p.setdefault("fallback", False)
    if type(p["fallback"]) is not bool:
        raise ValueError("fallback must be boolean")
    return p


class OwnedHaClient:
    def __init__(self, flow):
        self.flow = flow

    def cleanup(self, deadline):
        process = self.flow.proc
        if process is not None:
            if process.alive():
                process.proc.terminate()
                try:
                    process.proc.wait(timeout=min(2, deadline.remaining()))
                except subprocess.TimeoutExpired:
                    process.proc.kill()
            process.proc.wait(timeout=deadline.remaining())

    def finish(self, deadline):
        if self.flow.proc is None:
            raise RuntimeError("HA client was not started")
        rc = self.flow.proc.proc.wait(timeout=deadline.remaining())
        if rc != 0:
            raise RuntimeError(f"HA client exit code {rc}")
        path = self.flow.out_dir / "client_events.jsonl"
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if not rows:
            raise ValueError("HA client produced no request evidence")
        for row in rows:
            if not isinstance(row, dict) or not {
                "rid",
                "route_path",
                "master_target",
                "failover",
                "error_kind",
                "status",
            } <= set(row):
                raise ValueError("HA client evidence lacks required route fields")
            if (
                row["route_path"] not in {"master", "fallback", "failed"}
                or type(row["failover"]) is not bool
            ):
                raise ValueError("invalid HA route evidence")
            timestamp = row.get("send_start_epoch_ms")
            if timestamp is None:
                timestamp = row.get("wall_clock_ts")
            if (
                type(timestamp) not in (int, float)
                or not math.isfinite(timestamp)
                or timestamp <= 0
            ):
                raise ValueError("HA request has no valid issue timestamp")
        return rows, path


def _ha_start(ctx, params, deadline):
    from ...support.ha import HaTrafficRunner

    for target in params["targets"]:
        _process(ctx, target)
    targets = [
        ctx.backend.manager.master_instance_target(ctx.env, target)
        for target in params["targets"]
    ]
    directory = ctx.artifact_dir / f"ha-client-{len(ctx._resources)}"
    directory.mkdir(parents=True, exist_ok=True)
    flow = HaTrafficRunner(
        ctx.case_context,
        ctx.env,
        directory,
        "traffic",
        targets,
        duration_s=params["duration_s"],
        timeout_ms=params["timeout_ms"],
        enable_fallback=params["fallback"],
    )
    flow._overrides["FETCH_OUTPUT_STREAM"] = "true"
    flow._overrides["ENABLE_FALLBACK"] = str(params["fallback"]).lower()
    owned = OwnedHaClient(flow)
    handle = ctx.register_resource("ha_client", owned, owned.cleanup)
    deadline.check()
    flow.start()
    deadline.check()
    return StageOutput({"client": handle})


def _ha_finish_validate(params, plan):
    p = _params(params, plan, {"client"}, {"client"})
    plan.reference(p["client"], "ha_client")
    return p


def _ha_finish(ctx, params, deadline):
    client = ctx.resource(params["client"], "ha_client")
    rows, path = client.finish(deadline)
    return StageOutput(
        {"rows": ctx.register_resource("ha_rows", rows, historical=True)},
        artifacts=[str(path)],
    )


def _mark_validate(params, plan):
    p = _params(params, plan, {"wait_s"})
    p.setdefault("wait_s", 0)
    if (
        type(p["wait_s"]) not in (int, float)
        or not math.isfinite(p["wait_s"])
        or not 0 <= p["wait_s"] <= 180
    ):
        raise ValueError("wait_s must be finite in [0,180]")
    return p


def _mark(ctx, params, deadline):
    deadline.sleep(params["wait_s"])
    return StageOutput({"epoch_s": time.time()})


def _window_validate(params, plan):
    p = _params(
        params,
        plan,
        {
            "rows",
            "from",
            "until",
            "route",
            "status",
            "error_kind",
            "from_offset_s",
            "until_offset_s",
        },
        {"rows"},
    )
    plan.reference(p["rows"], "ha_rows")
    for key in ("from", "until"):
        if key in p:
            plan.reference(p[key], "number")
    if "route" in p and p["route"] not in {"master", "fallback", "failed"}:
        raise ValueError("invalid window route filter")
    if "status" in p and p["status"] not in {"ok", "schedule_error"}:
        raise ValueError("unsupported window status filter")
    if "error_kind" in p and p["error_kind"] not in {
        "none",
        "transport",
        "business",
        "deadline",
    }:
        raise ValueError("invalid window error filter")
    for field in ("from_offset_s", "until_offset_s"):
        if field in p:
            if (
                field.split("_")[0] not in p
                or type(p[field]) not in (int, float)
                or not math.isfinite(p[field])
            ):
                raise ValueError(
                    "window offset requires finite offset and boundary reference"
                )
    return p


def _window(ctx, params, deadline):
    from ...support.ha import rows_between

    deadline.check()
    rows = ctx.resource(params["rows"], "ha_rows")
    lower = ctx.resolve(params["from"]) if "from" in params else None
    upper = ctx.resolve(params["until"]) if "until" in params else None
    if lower is not None:
        lower += params.get("from_offset_s", 0)
    if upper is not None:
        upper += params.get("until_offset_s", 0)
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("HA observation window is empty or inverted")
    selected = rows_between(rows, lower, upper)
    for param, field in (
        ("route", "route_path"),
        ("status", "status"),
        ("error_kind", "error_kind"),
    ):
        if param in params:
            selected = [r for r in selected if r[field] == params[param]]
    return StageOutput(
        {"rows": ctx.register_resource("ha_rows", selected, historical=True)}
    )


HA_METRICS = {
    "sample_count",
    "success_rate",
    "target_share",
    "route_share",
    "route_count",
    "failover_count",
    "duplicate_ids",
    "error_kind_count",
    "wrong_error_code",
    "failed_count",
    "failed_rate_above_one",
    "business_rate_above_one",
    "visible_terminal_count",
    "visible_terminal_share",
}


def _client_check_validate(params, plan):
    p = _params(
        params,
        plan,
        {
            "rows",
            "metric",
            "op",
            "expected",
            "target",
            "route",
            "error_kind",
            "code",
            "min_samples",
        },
        {"rows", "metric", "op", "expected"},
    )
    plan.reference(p["rows"], "ha_rows")
    if p["metric"] not in HA_METRICS or p["op"] not in {"eq", "ge", "le"}:
        raise ValueError("unknown client metric/comparison")
    if type(p["expected"]) not in (int, float) or not math.isfinite(p["expected"]):
        raise ValueError("client comparison needs finite numeric expected value")
    p.setdefault("min_samples", 1)
    if type(p["min_samples"]) is not int or p["min_samples"] < 1:
        raise ValueError("client check must require actual samples")
    required = {
        "target_share": "target",
        "route_share": "route",
        "route_count": "route",
        "error_kind_count": "error_kind",
        "wrong_error_code": "code",
    }.get(p["metric"])
    if required and required not in p:
        raise ValueError(f"{p['metric']} requires {required}")
    if "target" in p and p["target"] not in ("A", "B"):
        raise ValueError("client target must be A or B")
    if "route" in p and p["route"] not in {"master", "fallback", "failed"}:
        raise ValueError("invalid expected route")
    if "error_kind" in p and p["error_kind"] not in {
        "none",
        "transport",
        "business",
        "deadline",
    }:
        raise ValueError("invalid expected error kind")
    if "code" in p and (type(p["code"]) is not int or p["code"] <= 0):
        raise ValueError("error code must be positive integer")
    return p


def _client_check(ctx, params, deadline):
    from collections import Counter

    deadline.check()
    rows = ctx.resource(params["rows"], "ha_rows")
    n = len(rows)
    metric = params["metric"]
    if metric == "sample_count":
        actual = n
    elif metric == "success_rate":
        actual = sum(r["status"] == "ok" for r in rows) / n if n else 0
    elif metric == "target_share":
        target = ctx.backend.manager.master_instance_target(ctx.env, params["target"])
        actual = sum(r["master_target"] == target for r in rows) / n if n else 0
    elif metric in {"route_share", "route_count"}:
        count = sum(r["route_path"] == params["route"] for r in rows)
        actual = count / n if metric == "route_share" and n else count
    elif metric == "failover_count":
        actual = sum(r["failover"] is True for r in rows)
    elif metric == "duplicate_ids":
        actual = sum(count > 1 for count in Counter(r["rid"] for r in rows).values())
    elif metric == "error_kind_count":
        actual = sum(r["error_kind"] == params["error_kind"] for r in rows)
    elif metric in {"failed_rate_above_one", "business_rate_above_one"}:
        count = sum(
            (
                r["route_path"] == "failed"
                if metric == "failed_rate_above_one"
                else r["error_kind"] == "business"
            )
            for r in rows
        )
        actual = count / n if count > 1 and n else 0
    elif metric in {"visible_terminal_count", "visible_terminal_share"}:
        actual = sum(
            r["status"] == "ok"
            or r["error_kind"] in {"deadline", "transport", "business"}
            for r in rows
        )
        if metric == "visible_terminal_share":
            actual = actual / n if n else 0
    elif metric == "wrong_error_code":
        actual = sum(str(params["code"]) not in str(r.get("error", "")) for r in rows)
    else:
        actual = sum(r["route_path"] == "failed" for r in rows)
    expected = params["expected"]
    comparison = (
        actual == expected
        if params["op"] == "eq"
        else actual >= expected if params["op"] == "ge" else actual <= expected
    )
    passed = n >= params["min_samples"] and comparison
    return StageOutput(
        {"actual": actual},
        [
            CheckResult(
                "criterion",
                "PASS" if passed else "FAIL",
                actual=actual,
                expected=expected,
                evidence={
                    "metric": metric,
                    "sample_count": n,
                    "min_samples": params["min_samples"],
                },
            )
        ],
    )


HANDLERS += [
    StageHandler(
        "master_client_start", _ha_validate, _ha_start, {"client": "ha_client"}
    ),
    StageHandler(
        "master_client_finish", _ha_finish_validate, _ha_finish, {"rows": "ha_rows"}
    ),
    StageHandler("master_mark", _mark_validate, _mark, {"epoch_s": "number"}),
    StageHandler(
        "master_client_window", _window_validate, _window, {"rows": "ha_rows"}
    ),
    StageHandler(
        "master_client_check",
        _client_check_validate,
        _client_check,
        {"actual": "number"},
        checks=frozenset({"criterion"}),
    ),
]


def _batch_validate(params, plan):
    p = _target(
        _params(
            params,
            plan,
            {
                "target",
                "count",
                "concurrency",
                "request_timeout_s",
                "sample_after_s",
                "coldstart",
            },
        )
    )
    for key, default, minimum, maximum in (
        ("count", 20, 1, 100),
        ("concurrency", 10, 1, 10),
        ("request_timeout_s", 15, 1, 30),
        ("sample_after_s", 0, 0, 10),
    ):
        p.setdefault(key, default)
        if type(p[key]) is not int or not minimum <= p[key] <= maximum:
            raise ValueError(f"{key} outside finite batch bounds")
    p.setdefault("coldstart", False)
    if type(p["coldstart"]) is not bool:
        raise ValueError("coldstart must be boolean")
    if (
        p["coldstart"]
        and getattr(plan, "environment", {}).get("master_stable_window_s", 3) != 0
    ):
        raise ValueError("coldstart requires zero master stability window")
    _layout(p, plan)
    return p


class FiniteMasterBatch:
    def __init__(self, records, concurrency):
        from concurrent.futures import ThreadPoolExecutor

        self.records = records
        self.pool = ThreadPoolExecutor(max_workers=concurrency)
        self.futures = []
        self.rows = []
        self.artifact = None
        self.samples = []

    def submit(self, row, shape, timeout):
        self.records.update(row, consumer_started=False)
        self.rows.append(row)

        def run():
            self.records.update(row, consumer_started=True)
            self.records.run(row, shape, timeout)

        future = self.pool.submit(run)
        self.futures.append(future)

    def persist(self):
        if self.artifact is not None:
            self.artifact.write_text(
                json.dumps(
                    dict(
                        records=self.records.snapshot_records(), topology=self.samples
                    ),
                    indent=2,
                )
                + "\n"
            )

    def cleanup(self, deadline):
        from concurrent.futures import wait

        self.records.cancel_active("master_batch_cleanup")
        self.pool.shutdown(wait=False, cancel_futures=True)
        try:
            # Cancelled queued jobs never had a consumer. Do not fabricate a
            # consumer exit for them, or confuse Future.cancel with RPC exit.
            live = [future for future in self.futures if not future.cancelled()]
            _, unfinished = wait(live, timeout=deadline.remaining())
            if unfinished:
                raise TimeoutError(
                    "master batch consumers did not stop after RPC cancellation"
                )
            for row in self.rows:
                if not row["consumer_started"]:
                    self.records.update(
                        row,
                        schedule={"status": "NOT_STARTED"},
                        cancel={
                            "requested_s": self.records.clock(),
                            "reason": "queued_at_cleanup",
                        },
                    )
                elif (
                    row["consumer_exit_s"] is None
                    or row["transport_terminal_s"] is None
                ):
                    raise RuntimeError(
                        "master consumer completed without terminal evidence"
                    )
                else:
                    self.records.update(row, consumer_completion_verified=True)
        finally:
            self.persist()


def _batch(ctx, params, deadline):
    from .elastic import RecordedRequests, request_success

    _process(ctx, params["target"])
    if params["coldstart"] and ctx.env.spec.master_stable_window_s != 0:
        raise ValueError("actual environment was warmed before coldstart burst")
    if params["target"] == "single":
        ops = ctx.ops
    else:
        from ...engine_ops import EngineOps

        spec = ctx.env.master_specs[params["target"]]
        ops = EngineOps(spec.bind_ip, spec.http_port, ctx.env.mock_http_port)
        ctx.add_cleanup("master_batch_channels", lambda d: ops.close())
    records = RecordedRequests(ops, ctx.env_epoch, ctx.clock)
    batch = FiniteMasterBatch(records, params["concurrency"])
    handle = ctx.register_resource("requests", records, batch.cleanup, historical=True)
    artifact = ctx.artifact_dir / f"master-batch-{handle['id']}.json"
    samples = []
    batch.artifact, batch.samples = artifact, samples
    try:
        for _ in range(params["count"]):
            deadline.check()
            row = records.issue(ops.next_request_id(), ctx.clock)
            batch.submit(
                row,
                dict(
                    input_len=2048,
                    output_len=2,
                    block_keys=[row["wire_request_id"] * 100 + 1],
                ),
                min(params["request_timeout_s"], deadline.remaining()),
            )
        ended = None
        while True:
            deadline.check()
            if all(future.done() for future in batch.futures) and ended is None:
                ended = ctx.clock()
                for future in batch.futures:
                    future.result()
            info = _master_json(
                ctx, params["target"], "/rtp_llm/master/info", deadline, True
            )
            summary = info["worker_summary"]
            observed = {}
            for role in ("PREFILL", "DECODE"):
                values = {key: summary[role][key] for key in ("discovered", "alive")}
                if any(
                    type(value) is not int or value < 0 for value in values.values()
                ):
                    raise ValueError("master topology sample has invalid counts")
                observed[role] = values
            samples.append(dict(time_s=ctx.clock(), workers=observed))
            if ended is not None and ctx.clock() - ended >= params["sample_after_s"]:
                break
            deadline.sleep(0.5)
        batch.pool.shutdown(wait=False)
    finally:
        artifact.write_text(
            json.dumps(
                dict(records=records.snapshot_records(), topology=samples), indent=2
            )
            + "\n"
        )
    rows = records.snapshot_records()
    successes = [r for r in rows if request_success(r)]
    value = dict(
        records=rows,
        topology=samples,
        coldstart=params["coldstart"],
        success_rate=len(successes) / len(rows),
        expected_prefill=ctx.env.spec.n_prefill,
        expected_decode=ctx.env.spec.n_decode,
    )
    return StageOutput(
        {
            "requests": handle,
            "snapshot": ctx.register_resource("snapshot", value, historical=True),
            "success_rate": value["success_rate"],
        },
        artifacts=[str(artifact)],
    )


def _cold_check_validate(params, plan):
    p = _params(params, plan, {"snapshot"}, {"snapshot"})
    plan.reference(p["snapshot"], "snapshot")
    return p


def _cold_check(ctx, params, deadline):
    from collections import Counter

    from .elastic import request_success

    deadline.check()
    value = ctx.resource(params["snapshot"], "snapshot")
    if (
        value["coldstart"] is not True
        or not value["topology"]
        or len(value["records"]) != 20
    ):
        raise ValueError(
            "coldstart verdict requires its actual twenty-request sampled burst"
        )
    final = value["topology"][-1]["workers"]
    dist = Counter(
        r["prefill_addr"]
        for r in value["records"]
        if request_success(r) and r["prefill_addr"]
    )
    total = sum(dist.values())
    share = max(dist.values()) / total if total else 1
    topology = all(
        final[role]["alive"] == final[role]["discovered"] == value[key]
        for role, key in (
            ("PREFILL", "expected_prefill"),
            ("DECODE", "expected_decode"),
        )
    )
    return StageOutput(
        {"success_rate": value["success_rate"]},
        [
            CheckResult(
                "success",
                "PASS" if value["success_rate"] >= 0.8 else "FAIL",
                actual=value["success_rate"],
                expected=0.8,
            ),
            CheckResult("topology", "PASS" if topology else "FAIL", actual=final),
            CheckResult(
                "balance",
                "PASS" if len(dist) >= 2 and share <= 0.8 else "FAIL",
                actual={"distribution": dict(dist), "max_share": share},
                expected="two prefills used, maximum share <= 0.8",
            ),
        ],
    )


HANDLERS += [
    StageHandler(
        "master_request_batch",
        _batch_validate,
        _batch,
        {"requests": "requests", "snapshot": "snapshot", "success_rate": "number"},
    ),
    StageHandler(
        "master_coldstart_check",
        _cold_check_validate,
        _cold_check,
        {"success_rate": "number"},
        checks=frozenset({"success", "topology", "balance"}),
    ),
]


def _inflight_validate(params, plan):
    p = _target(_params(params, plan, {"target", "op", "value"}, {"op", "value"}))
    if (
        p["op"] not in {"eq", "ge", "le"}
        or type(p["value"]) is not int
        or p["value"] < 0
    ):
        raise ValueError(
            "scheduler inflight condition must be eq/ge/le nonnegative integer"
        )
    _layout(p, plan)
    return p


def _inflight(ctx, params, deadline):
    samples = []
    path = ctx.artifact_dir / f"scheduler-inflight-{len(ctx._resources)}.json"
    try:
        while True:
            deadline.check()
            data = _master_json(
                ctx, params["target"], "/rtp_llm/inflight_status", deadline
            )
            count = data["scheduler_inflight"]
            if type(count) is not int or count < 0:
                raise ValueError("missing or invalid scheduler inflight observation")
            samples.append(dict(time_s=ctx.clock(), count=count))
            matched = (
                count == params["value"]
                if params["op"] == "eq"
                else (
                    count >= params["value"]
                    if params["op"] == "ge"
                    else count <= params["value"]
                )
            )
            if matched:
                break
            deadline.sleep(0.5)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        {"count": count},
        [
            CheckResult(
                "scheduler_inflight", "PASS", actual=count, expected=params["value"]
            )
        ],
        [str(path)],
    )


HANDLERS.append(
    StageHandler(
        "master_wait_inflight",
        _inflight_validate,
        _inflight,
        {"count": "integer"},
        checks=frozenset({"scheduler_inflight"}),
    )
)


def _direct_validate(params, plan):
    p = _params(params, plan, {"engine"})
    p.setdefault("engine", "prefill-0")
    if not isinstance(p["engine"], str) or not re.fullmatch(
        r"prefill-[0-9]+", p["engine"]
    ):
        raise ValueError("direct request requires an explicit prefill engine")
    return p


def _direct(ctx, params, deadline):
    import grpc

    from .engine_control import _engines, _http

    entry = _engines(_http(ctx.ops, "snapshot", deadline), [params["engine"]])[
        params["engine"]
    ]
    if entry["role"] != "prefill":
        raise ValueError("direct request target is not prefill")
    state = dict(
        route="direct",
        request_id=ctx.ops.next_request_id(),
        engine=params["engine"],
        target=entry["grpc_addr"],
        method="GenerateStreamCall",
        issued_s=ctx.clock(),
        consumer_done=False,
        consumer_exit_s=None,
        error=None,
        business_finished=False,
    )
    call_box = [None]

    def cleanup(d):
        if call_box[0] is not None and not state["consumer_done"]:
            call_box[0].cancel()

    handle = ctx.register_resource("direct_request", state, cleanup, historical=True)
    artifact = ctx.artifact_dir / f"direct-{handle['id']}.json"
    try:
        deadline.check()
        stub = ctx.ops.pb2_grpc.RpcServiceStub(ctx.ops._channel(entry["grpc_addr"]))
        call = stub.GenerateStreamCall(
            ctx.ops.build_generate_input(state["request_id"], output_len=2),
            timeout=min(15, deadline.remaining()),
        )
        call_box[0] = call
        for frame in call:
            deadline.check()
            if frame.error_info.error_code:
                state["error"] = str(frame.error_info.error_message)
            if any(frame.flatten_output.finished):
                state["business_finished"] = True
    except grpc.RpcError as exc:
        state["error"] = dict(code=exc.code().name, detail=str(exc))
    finally:
        # The consumer is this synchronous stage, never an unjoined thread.
        if call_box[0] is not None:
            call_box[0].cancel()
        state["consumer_done"] = True
        state["consumer_exit_s"] = ctx.clock()
        artifact.write_text(json.dumps(state, indent=2) + "\n")
    return StageOutput(
        {
            "result": handle,
            "error": state["error"] is not None,
            "finished": state["business_finished"],
        },
        artifacts=[str(artifact)],
    )


def _direct_clean_validate(params, plan):
    return _params(params, plan, set())


def _direct_clean(ctx, params, deadline):
    from .engine_control import _engines, _http

    names = [f"prefill-{i}" for i in range(ctx.env.spec.n_prefill)]
    while True:
        deadline.check()
        engines = _engines(_http(ctx.ops, "snapshot", deadline), names)
        states = {}
        for name, row in engines.items():
            if (
                type(row.get("inflight")) is not int
                or row["inflight"] < 0
                or type(row.get("leak_detected")) is not bool
            ):
                raise ValueError("direct engine cleanup observation missing or invalid")
            states[name] = dict(
                inflight=row["inflight"], leak_detected=row["leak_detected"]
            )
        if all(
            v["inflight"] == 0 and v["leak_detected"] is False for v in states.values()
        ):
            break
        deadline.sleep(0.5)
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", states, historical=True)},
        [
            CheckResult(
                "engine_inflight",
                "PASS",
                actual=states,
                expected="all prefill engine inflight zero and leak_detected false",
            )
        ],
    )


HANDLERS += [
    StageHandler(
        "master_direct_request",
        _direct_validate,
        _direct,
        {"result": "direct_request", "error": "boolean", "finished": "boolean"},
    ),
    StageHandler(
        "master_direct_clean",
        _direct_clean_validate,
        _direct_clean,
        {"snapshot": "snapshot"},
        checks=frozenset({"engine_inflight"}),
    ),
]


def _state_validate(params, plan):
    p = _target(_params(params, plan, {"target"}))
    _layout(p, plan)
    return p


def _state(ctx, params, deadline):
    proc = _process(ctx, params["target"])
    info = _master_json(ctx, params["target"], "/rtp_llm/master/info", deadline, True)
    inflight = _master_json(ctx, params["target"], "/rtp_llm/inflight_status", deadline)
    count = inflight["scheduler_inflight"]
    if type(count) is not int or count < 0:
        raise ValueError("master state has no valid scheduler count")
    topology = {}
    for role in ("PREFILL", "DECODE"):
        value = info["worker_summary"][role]["discovered"]
        if type(value) is not int or value < 0:
            raise ValueError("master state lacks discovered count")
        topology[role] = value
    state = dict(
        target=params["target"],
        pid=proc.pid,
        ready=info.get("ready"),
        topology=topology,
        scheduler_inflight=count,
        sampled_s=ctx.clock(),
    )
    handle = ctx.register_resource("master_state", state, historical=True)
    artifact = ctx.artifact_dir / f"master-state-{handle['id']}.json"
    artifact.write_text(json.dumps(state, indent=2) + "\n")
    return StageOutput({"state": handle}, artifacts=[str(artifact)])


def _continuity_validate(params, plan):
    p = _params(params, plan, {"before", "after"}, {"before", "after"})
    for key in p:
        plan.reference(p[key], "master_state")
    return p


def _continuity(ctx, params, deadline):
    deadline.check()
    before, after = [
        ctx.resource(params[key], "master_state") for key in ("before", "after")
    ]
    if before["target"] != after["target"]:
        raise ValueError("cannot compare different master owners")
    identity = before["pid"] == after["pid"]
    topology = after["ready"] is True and all(
        after["topology"][role] >= before["topology"][role]
        and after["topology"][role] == expected
        for role, expected in (
            ("PREFILL", ctx.env.spec.n_prefill),
            ("DECODE", ctx.env.spec.n_decode),
        )
    )
    witnessed = before["scheduler_inflight"] > 0
    retained = not witnessed or after["scheduler_inflight"] >= 1
    return StageOutput(
        {"retained": identity and topology and retained},
        [
            CheckResult(
                "same_process",
                "PASS" if identity else "FAIL",
                actual=after["pid"],
                expected=before["pid"],
            ),
            CheckResult(
                "discovered_continuity",
                "PASS" if topology else "FAIL",
                actual=after["topology"],
                expected=before["topology"],
            ),
            CheckResult(
                "scheduler_continuity",
                "PASS" if retained else "FAIL",
                actual=after["scheduler_inflight"],
                expected="nonzero if nonzero before freeze",
                evidence={
                    "before": before["scheduler_inflight"],
                    "nonzero_before_observed": witnessed,
                },
            ),
        ],
    )


HANDLERS += [
    StageHandler("master_state", _state_validate, _state, {"state": "master_state"}),
    StageHandler(
        "master_continuity",
        _continuity_validate,
        _continuity,
        {"retained": "boolean"},
        checks=frozenset(
            {"same_process", "discovered_continuity", "scheduler_continuity"}
        ),
    ),
]


def _short_validate(params, plan):
    p = _params(
        params,
        plan,
        {"hang", "burst", "post", "target"},
        {"hang", "burst", "post", "target"},
    )
    for key in ("hang", "burst", "post"):
        plan.reference(p[key], "ha_rows")
    if p["target"] not in ("A", "B"):
        raise ValueError("short-hang target must be an actual HA master")
    return p


def _short(ctx, params, deadline):
    deadline.check()
    hang, burst, post = [
        ctx.resource(params[key], "ha_rows") for key in ("hang", "burst", "post")
    ]
    target = ctx.backend.manager.master_instance_target(ctx.env, params["target"])
    complete = lambda rows: all(
        r["status"] == "ok" and r["master_target"] == target for r in rows
    )
    # A saturated eight-client flow can legitimately issue zero NEW requests
    # while frozen. A nonempty post-thaw burst remains mandatory evidence.
    passed = (
        complete(hang)
        and len(burst) >= 3
        and complete(burst)
        and all(r["master_target"] == target for r in post)
    )
    return StageOutput(
        {"passed": passed},
        [
            CheckResult(
                "short_hang",
                "PASS" if passed else "FAIL",
                actual={
                    "hang_rows": len(hang),
                    "burst_rows": len(burst),
                    "post_rows": len(post),
                },
                expected="all issued hang and >=3 burst requests complete on the same master",
            )
        ],
    )


HANDLERS.append(
    StageHandler(
        "master_short_hang_check",
        _short_validate,
        _short,
        {"passed": "boolean"},
        checks=frozenset({"short_hang"}),
    )
)
