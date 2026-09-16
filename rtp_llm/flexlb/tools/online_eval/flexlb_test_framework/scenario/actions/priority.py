"""Bounded concurrent priority cohorts and observed dispatch-order checks."""

import copy
import json
import math
import threading
import urllib.error
import urllib.request
import uuid

from ..backend import RequestBatch
from ..contracts import CheckResult, StageHandler, StageOutput
from ..runtime import Deadline
from .engine_control import _http


def _params(value, allowed, required=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(allowed)
        or set(required) - set(value)
    ):
        raise ValueError("invalid priority action parameters")
    return copy.deepcopy(value)


def _number(value, low, high, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError("priority parameter outside finite bounds")
    return value


def _empty(p, plan):
    return _params(p, ())


def _fleet(ctx, p, deadline):
    rows = _http(ctx.ops, "snapshot", deadline).get("engines")
    if not isinstance(rows, list):
        raise ValueError("missing engine inventory")
    names = [
        r.get("name")
        for r in rows
        if r.get("role") == "prefill" and r.get("stopped") is False
    ]
    if len(names) != 1 or not isinstance(names[0], str):
        raise ValueError("priority queue choreography requires one live Prefill")
    return StageOutput(dict(prefill=names[0]))


def _perf_params(p, plan):
    p = _params(
        p,
        {"prefill_fixed_ms", "expected_prefill"},
        {"prefill_fixed_ms", "expected_prefill"},
    )
    _number(p["prefill_fixed_ms"], 0, 60000)
    _number(p["expected_prefill"], 1, 8, True)
    return p


def _prefill_perf(ctx, p, deadline):
    from .engine_control import execute

    rows = _http(ctx.ops, "snapshot", deadline).get("engines")
    if not isinstance(rows, list):
        raise ValueError("missing Prefill inventory")
    targets = [r.get("name") for r in rows if r.get("role") == "prefill"]
    if len(targets) != p["expected_prefill"]:
        raise ValueError("Prefill inventory differs from explicit choreography")
    result = execute(
        ctx,
        dict(
            operation="set_perf",
            targets=targets,
            perf=dict(prefill_fixed_ms=p["prefill_fixed_ms"]),
        ),
        deadline,
    )
    return StageOutput(artifacts=result.artifacts)


def _wave_params(p, plan):
    p = _params(
        p, {"requests", "gap_s", "serial_schedule", "defer_batch"}, {"requests"}
    )
    if not isinstance(p["requests"], list) or not 1 <= len(p["requests"]) <= 32:
        raise ValueError("priority cohort must contain 1..32 requests")
    tags = set()
    for row in p["requests"]:
        _params(
            row, {"tag", "priority", "qos_level", "input_len", "output_len"}, {"tag"}
        )
        if not isinstance(row["tag"], str) or not row["tag"] or row["tag"] in tags:
            raise ValueError("priority tags must be unique nonempty strings")
        tags.add(row["tag"])
        for field in ("priority", "qos_level"):
            if field in row:
                _number(row[field], -(2**31), 2**31 - 1, True)
        for field, default in (("input_len", 2048), ("output_len", 2)):
            row.setdefault(field, default)
            _number(row[field], 1, 1000000, True)
    p.setdefault("gap_s", 0.15)
    _number(p["gap_s"], 0, 2)
    p.setdefault("serial_schedule", False)
    if type(p["serial_schedule"]) is not bool:
        raise ValueError("serial_schedule must be boolean")
    p.setdefault("defer_batch", False)
    if type(p["defer_batch"]) is not bool:
        raise ValueError("defer_batch must be boolean")
    return p


class PriorityWave:
    """Schedule concurrently; preserve separate bounded consumer-drain clocks."""

    def __init__(self, ctx, p):
        self.ctx, self.p = ctx, p
        self.entries = []
        self.path = ctx.artifact_dir / f"priority-wave-{uuid.uuid4().hex}.json"
        self.complete = False

    def start(self, deadline):
        for shape in self.p["requests"]:
            deadline.check()
            params = {k: v for k, v in shape.items() if k != "tag"}
            params.update(
                count=1,
                consume=(
                    "deferred"
                    if self.p.get("defer_batch", False)
                    and self.ctx.instance["environment"]["resolved_config"][
                        "dispatcher"
                    ]["type"]
                    == "BATCH"
                    else "immediate"
                ),
                schedule_timeout_s=90,
                stream_timeout_s=120,
            )
            batch = RequestBatch(self.ctx, params)
            batch.artifact = (
                self.ctx.artifact_dir / f"priority-request-{uuid.uuid4().hex}.json"
            )
            item = dict(
                tag=shape["tag"], batch=batch, done=threading.Event(), error=None
            )
            self.entries.append(item)
            # Independent 90 s Schedule budget, bounded by the instance. The
            # start-stage budget governs issuance, not the lifetime of this RPC.
            end = min(self.ctx.instance_deadline_s, self.ctx.clock() + 90)
            item["thread"] = threading.Thread(
                target=self._submit, args=(item, end), daemon=True
            )
            item["thread"].start()
            if self.p.get("serial_schedule", False):
                self._join(item, deadline)
                if item["error"] is not None:
                    raise item["error"]
            deadline.sleep(self.p["gap_s"])

    def _submit(self, item, end):
        try:
            item["submitted_s"] = self.ctx.clock()
            item["batch"].submit(Deadline(end, self.ctx.clock, self.ctx.sleeper))
        except Exception as exc:
            item["error"] = exc
        finally:
            item["done"].set()

    def _join(self, item, deadline):
        while not item["done"].is_set():
            item["done"].wait(min(0.05, deadline.remaining()))
        item["thread"].join(deadline.remaining())
        if item["thread"].is_alive():
            raise RuntimeError("Schedule worker lacks exit evidence")

    def wait(self, deadline):
        for item in self.entries:
            self._join(item, deadline)
        for item in self.entries:
            if item["error"] is not None:
                raise item["error"]
            # Legacy _drain gives each consumer a fresh 35 s wait after all
            # Schedule calls settle. Streams themselves opened with 120 s RPC.
            item["batch"].wait(
                Deadline(
                    min(deadline.expires_at, self.ctx.clock() + 35),
                    self.ctx.clock,
                    self.ctx.sleeper,
                )
            )
        self.complete = True
        self.persist()

    def records(self):
        rows = []
        for item in self.entries:
            for record in item["batch"].snapshot_records():
                rows.append(
                    dict(tag=item["tag"], submitted_s=item.get("submitted_s"), **record)
                )
        return rows

    def persist(self):
        self.path.write_text(
            json.dumps(self.records(), indent=2, allow_nan=False) + "\n"
        )

    def cleanup(self, deadline):
        errors = []
        for item in self.entries:
            item["batch"].cancel("priority_cleanup")
        for item in self.entries:
            try:
                self._join(item, deadline)
                item["batch"].cleanup(deadline)
            except Exception as exc:
                errors.append(exc)
        self.persist()
        if errors:
            raise errors[0]


def _start(ctx, p, deadline):
    wave = PriorityWave(ctx, p)
    handle = ctx.register_resource("requests", wave, cleanup=wave.cleanup)
    wave.start(deadline)
    return StageOutput(dict(requests=handle))


def _reference(p, plan):
    p = _params(p, {"requests"}, {"requests"})
    plan.reference(p["requests"], "requests")
    return p


def _wave(ctx, p):
    wave = ctx.resource(p["requests"], "requests")
    if not isinstance(wave, PriorityWave):
        raise ValueError("priority action requires a priority cohort")
    return wave


def _wait(ctx, p, deadline):
    wave = _wave(ctx, p)
    wave.wait(deadline)
    return StageOutput(artifacts=[str(wave.path)])


def _settled(ctx, p, deadline):
    wave = _wave(ctx, p)
    for item in wave.entries:
        wave._join(item, deadline)
        if item["error"] is not None:
            raise item["error"]
    wave.persist()
    return StageOutput(artifacts=[str(wave.path)])


def _completion_params(p, plan):
    p = _params(p, {"requests"}, {"requests"})
    if not isinstance(p["requests"], list) or not 1 <= len(p["requests"]) <= 4:
        raise ValueError("completion requires 1..4 cohorts")
    for ref in p["requests"]:
        plan.reference(ref, "requests")
    return p


def _completion(ctx, p, deadline):
    from .elastic import request_success

    records = []
    for ref in p["requests"]:
        wave = _wave(ctx, {"requests": ref})
        if not wave.complete:
            raise ValueError("completion check requires settled cohort")
        rows = wave.records()
        if len(rows) != len(wave.p["requests"]):
            raise ValueError("completion cohort lost a request record")
        records.extend(rows)
    ids = [r["wire_request_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate priority request identity")
    groups = {}
    for row in records:
        priority = str(row["request_shape"].get("priority", "unset"))
        group = groups.setdefault(
            priority, {"issued": 0, "completed": 0, "latencies_s": []}
        )
        group["issued"] += 1
        group["completed"] += int(request_success(row))
        start, end = row["schedule"]["started_s"], row["transport_terminal_s"]
        if start is not None and end is not None:
            group["latencies_s"].append(end - start)
    return StageOutput(
        checks=[
            CheckResult(
                "P6",
                (
                    "PASS"
                    if records and all(request_success(r) for r in records)
                    else "FAIL"
                ),
                actual=groups,
            )
        ]
    )


def _pending_params(p, plan):
    p = _params(p, {"prefill", "requests"}, {"prefill", "requests"})
    plan.reference(p["prefill"], "string")
    plan.reference(p["requests"], "requests")
    return p


def _pending(ctx, p, deadline):
    wave = _wave(ctx, p)
    for item in wave.entries:
        wave._join(item, deadline)
        if item["error"] is not None:
            raise item["error"]
        response = item["batch"].entries[0]["response"]
        if response.code != 200 or not response.success:
            return StageOutput(
                checks=[CheckResult("dispatched", "FAIL", "placeholder rejected")]
            )
    name = ctx.resolve(p["prefill"])
    samples = []
    path = ctx.artifact_dir / f"priority-pending-{uuid.uuid4().hex}.json"
    try:
        while True:
            deadline.check()
            raw = _http(ctx.ops, "snapshot", deadline)
            rows = [
                r
                for r in raw.get("engines", [])
                if r.get("name") == name and r.get("role") == "prefill"
            ]
            if len(rows) != 1:
                raise ValueError("pending Prefill missing or ambiguous")
            row = rows[0]
            count = _number(row.get("waiting"), 0, 1e9) + _number(
                row.get("running"), 0, 1e9
            )
            samples.append(dict(time_s=ctx.clock(), raw=raw))
            if count >= 1:
                return StageOutput(
                    checks=[CheckResult("dispatched", "PASS", actual=count)],
                    artifacts=[str(path)],
                )
            deadline.sleep(0.1)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")


def _expiry_params(p, plan):
    p = _params(p, {"placeholder", "wave"}, {"placeholder", "wave"})
    for key in p:
        plan.reference(p[key], "requests")
    return p


def _expiry(ctx, p, deadline):
    from ...grade import GradeReport
    from .elastic import request_success

    placeholder = _wave(ctx, {"requests": p["placeholder"]})
    wave = _wave(ctx, {"requests": p["wave"]})
    if not placeholder.complete or not wave.complete:
        raise ValueError("expiry check requires completed waits")
    ph, rows = placeholder.records(), wave.records()
    if len(ph) != 1 or len(rows) != 4:
        raise ValueError("expiry cohort must contain one placeholder and four peers")
    codes = []
    for item in wave.entries:
        if len(item["batch"].entries) != 1:
            raise ValueError("missing expiry response")
        response = item["batch"].entries[0]["response"]
        if response is None:
            raise ValueError("missing Schedule result")
        codes.append(int(response.code))
    low = [r for r in rows if r["request_shape"].get("priority") == 30]
    if len(low) != 3:
        raise ValueError("expiry ratio requires three priority30 peers")
    times = []
    for row in low:
        start = _number(row["submitted_s"], 0, 1e18)
        end = _number(row["schedule"]["ended_s"], start, 1e18)
        times.append(end - start)
    ratio = max(times) / 8.0
    report = GradeReport(run_grade=ctx.instance.get("grade", "normal"))
    report.check("PR8", ratio)
    return StageOutput(
        checks=[
            CheckResult("PR8", "PASS" if report.passed else "FAIL", actual=ratio),
            CheckResult(
                "P6",
                "PASS" if request_success(ph[0]) and codes == [8511] * 4 else "FAIL",
                actual=codes,
            ),
        ],
        artifacts=[str(placeholder.path), str(wave.path)],
    )


def _dispatch_observation(ctx, rows, deadline, missing_last=False):
    raw = _http(ctx.ops, "snapshot", deadline)
    engines = raw.get("engines")
    if not isinstance(engines, list):
        raise ValueError("missing dispatch snapshot")
    settled = sorted(
        rows, key=lambda r: (r["schedule"]["ended_s"], r["wire_request_id"])
    )
    ranks = {r["wire_request_id"]: i for i, r in enumerate(settled)}
    dispatch = []
    for row in rows:
        rid = row["wire_request_id"]
        matches = [
            e.get("request_lifecycle", {}).get(str(rid))
            for e in engines
            if e.get("role") == "prefill"
        ]
        matches = [m for m in matches if m is not None]
        if missing_last and (
            not matches or (len(matches) == 1 and matches[0].get("running_ms") is None)
        ):
            dispatch.append((1 << 60, ranks[rid], rid))
            continue
        if len(matches) != 1:
            raise ValueError("missing or ambiguous Prefill lifecycle")
        running = _number(matches[0].get("running_ms"), 0, 1e18)
        dispatch.append((running, ranks[rid], rid))
    return raw, dispatch


def _fifo(ctx, p, deadline):
    wave = _wave(ctx, p)
    if not wave.complete:
        raise ValueError("priority FIFO requires completed cohort observation")
    rows = wave.records()
    if len(rows) != len(wave.entries):
        raise ValueError("priority cohort lost a request record")
    raw, dispatch = _dispatch_observation(ctx, rows, deadline)
    expected = [r["wire_request_id"] for r in rows]
    actual = [r[2] for r in sorted(dispatch)]
    from .elastic import request_success

    success = all(request_success(r) for r in rows)
    evidence = ctx.artifact_dir / f"priority-dispatch-{uuid.uuid4().hex}.json"
    evidence.write_text(
        json.dumps(dict(snapshot=raw, dispatch=dispatch, records=rows), indent=2) + "\n"
    )
    return StageOutput(
        checks=[
            CheckResult(
                "PR2",
                "PASS" if success and actual == expected else "FAIL",
                actual=actual,
                expected=expected,
            ),
            CheckResult("P6_terminal", "PASS" if success else "FAIL"),
        ],
        artifacts=[str(wave.path), str(evidence)],
    )


def _order_params(p, plan):
    p = _params(
        p, {"placeholder", "wave", "first_peer_exempt"}, {"placeholder", "wave"}
    )
    _expiry_params({key: p[key] for key in ("placeholder", "wave")}, plan)
    if type(p.setdefault("first_peer_exempt", True)) is not bool:
        raise ValueError("first_peer_exempt must be boolean")
    return p


def _order_basic(ctx, p, deadline):
    from ...grade import GradeReport
    from .elastic import request_success

    placeholder = _wave(ctx, {"requests": p["placeholder"]})
    wave = _wave(ctx, {"requests": p["wave"]})
    if not placeholder.complete or not wave.complete:
        raise ValueError("priority ordering requires completed cohorts")
    ph, peers = placeholder.records(), wave.records()
    if len(ph) != 1 or len(peers) != 6:
        raise ValueError("basic ordering expects one placeholder and six peers")
    rows = ph + peers
    raw, dispatch = _dispatch_observation(ctx, rows, deadline)
    ordered = [r[2] for r in sorted(dispatch)]
    peer_ids = [r["wire_request_id"] for r in peers]
    if len(set(ordered)) != 7:
        raise ValueError("duplicate request identity in dispatch ordering")
    priorities = {r["wire_request_id"]: r["request_shape"]["priority"] for r in rows}
    # The case declares whether the first peer could reach an idle engine.
    # With an admitted placeholder, every peer belongs to the priority queue.
    exempt = p.get("first_peer_exempt", True)
    scored = peer_ids[1:] if exempt else peer_ids
    expected = ([peer_ids[0]] if exempt else []) + sorted(
        scored, key=lambda rid: (-priorities[rid], peer_ids.index(rid))
    )
    actual = [rid for rid in ordered if rid in peer_ids]
    shape = actual == expected
    position = {rid: i for i, rid in enumerate(ordered)}
    pairs = [
        (a, b)
        for i, a in enumerate(scored)
        for b in scored[i + 1 :]
        if priorities[a] != priorities[b]
    ]
    inversions = sum(
        (priorities[a] - priorities[b]) * (position[a] - position[b]) > 0
        for a, b in pairs
    )
    ratio = inversions / len(pairs) if pairs else 0.0
    fifo = all(
        [position[rid] for rid in peer_ids if priorities[rid] == priority]
        == sorted(position[rid] for rid in peer_ids if priorities[rid] == priority)
        for priority in set(priorities.values())
    )
    success = all(request_success(r) for r in rows)
    codes = [
        entry["response"].code
        for cohort in (placeholder, wave)
        for item in cohort.entries
        for entry in item["batch"].entries
    ]
    report = GradeReport(run_grade=ctx.instance.get("grade", "normal"))
    report.check("PR1", ratio)
    path = ctx.artifact_dir / f"priority-order-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            dict(
                snapshot=raw,
                dispatch=dispatch,
                records=rows,
                expected=expected,
                first_peer_exempt=exempt,
                scored=scored,
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput(
        checks=[
            CheckResult("PR1", "PASS" if report.passed else "FAIL", actual=ratio),
            CheckResult(
                "PR2",
                "PASS" if shape and fifo else "FAIL",
                actual=actual,
                expected=expected,
            ),
            CheckResult(
                "PR6",
                "PASS" if shape and success and codes == [200] * 7 else "FAIL",
                actual=codes,
            ),
            CheckResult("P6_terminal", "PASS" if success else "FAIL"),
        ],
        artifacts=[str(path), str(placeholder.path), str(wave.path)],
    )


def _normalize_wait(ctx, p, deadline):
    wave = _wave(ctx, p)
    if ctx.instance["environment"]["resolved_config"]["dispatcher"]["type"] != "BATCH":
        return _wait(ctx, p, deadline)
    _settled(ctx, p, deadline)
    if not wave.p.get("defer_batch"):
        raise ValueError(
            "Schedule-only BATCH normalization requires explicit deferred issuance"
        )
    if any(r.get("fetch_invocations", 0) for r in wave.records()):
        raise ValueError("legacy normalization BATCH segment must not Fetch")
    wave.complete = True
    wave.persist()
    return StageOutput(artifacts=[str(wave.path)])


def _normalize_metrics_wait(ctx, p, deadline):
    """Observe exited streams without adding a business-success metric gate."""
    wave = _wave(ctx, p)
    _settled(ctx, p, deadline)
    server_errors = {
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
        "UNAUTHENTICATED",
    }
    for item in wave.entries:
        deadline.check()
        try:
            item["batch"].wait(
                Deadline(
                    min(deadline.expires_at, ctx.clock() + 35), ctx.clock, ctx.sleeper
                )
            )
        except RuntimeError:
            rows = item["batch"].snapshot_records()
            if len(rows) != 1:
                raise
            row = rows[0]
            if not (
                row["schedule"]["status"] == "OK"
                and row["stream"]["status"] in server_errors
                and row.get("consumer_done") is True
                and row.get("consumer_completion_verified") is True
                and row.get("consumer_exit_s") is not None
                and row.get("transport_terminal_s") is not None
                and row["stream"]["ended_s"] is not None
                and row["cancel"]["requested_s"] is None
            ):
                raise
        deadline.check()
    wave.complete = True
    wave.persist()
    return StageOutput(artifacts=[str(wave.path)])


def _normalize_params(p, plan):
    p = _params(
        p,
        {"requests", "expected_tags", "batch_admission"},
        {"requests", "expected_tags"},
    )
    _completion_params({"requests": p["requests"]}, plan)
    expected = p["expected_tags"]
    if (
        not isinstance(expected, list)
        or not 1 <= len(expected) <= 32
        or any(not isinstance(x, str) for x in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("normalization expected tags must be unique strings")
    p.setdefault("batch_admission", False)
    if type(p["batch_admission"]) is not bool:
        raise ValueError("batch_admission must be boolean")
    return p


def _normalize_order(ctx, p, deadline):
    from .elastic import request_success

    rows = []
    for ref in p["requests"]:
        wave = _wave(ctx, {"requests": ref})
        if not wave.complete:
            raise ValueError("normalization observation requires settled cohort")
        rows.extend(wave.records())
    tags = {r["wire_request_id"]: r["tag"] for r in rows}
    if len(tags) != len(rows) or set(tags.values()) != set(p["expected_tags"]):
        raise ValueError("normalization request identities missing/duplicated")
    raw, dispatch = _dispatch_observation(ctx, rows, deadline, missing_last=True)
    actual = [tags[r[2]] for r in sorted(dispatch)]

    def okay(row):
        if p["batch_admission"] and row.get("enqueued_by_master"):
            return (
                row["schedule"]["status"] == "OK"
                and row.get("fetch_invocations", 0) == 0
            )
        return request_success(row)

    success = all(okay(row) for row in rows)
    path = ctx.artifact_dir / f"priority-normalize-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(dict(raw=raw, rows=rows, dispatch=dispatch), indent=2) + "\n"
    )
    return StageOutput(
        checks=[
            CheckResult(
                "PR3",
                "PASS" if success and actual == p["expected_tags"] else "FAIL",
                actual=actual,
                expected=p["expected_tags"],
            ),
            CheckResult("P6_terminal", "PASS" if success else "FAIL"),
        ],
        artifacts=[str(path)],
    )


def _metric_text(ctx, deadline):
    from ...engine_ops import MASTER_PROMETHEUS_PATHS

    for path in MASTER_PROMETHEUS_PATHS:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{ctx.ops.master_management_port}/{path}",
                timeout=min(5, deadline.remaining()),
            ) as response:
                data = response.read(4000001)
                if len(data) > 4000000:
                    raise ValueError("metrics response exceeds evidence bound")
                return data.decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError):
            deadline.check()
    return None


def _normalization_metrics(ctx, p, deadline):
    from ...engine_ops import parse_prometheus_samples

    wave = _wave(ctx, p)
    if not wave.complete:
        raise ValueError("metrics must follow settled normalization requests")
    warm = Deadline(min(deadline.expires_at, ctx.clock() + 180), ctx.clock, ctx.sleeper)
    while _metric_text(ctx, warm) is None:
        warm.sleep(2)
    # Preserve the old one-shot availability warmup followed by a fresh scrape;
    # do not retry until the expected bucket values happen to appear.
    body = _metric_text(ctx, deadline)
    if body is None:
        raise ValueError("missing normalization metric source")
    samples = parse_prometheus_samples(body, "")
    buckets = {}
    for priority in (30, 50, 70):
        values = [
            value
            for name, labels, value in samples
            if "auto_tpm_request" in name and labels.get("priority") == str(priority)
        ]
        for value in values:
            _number(value, 0, 1e18)
        buckets[str(priority)] = sum(values) if values else None
    path = ctx.artifact_dir / f"priority-metrics-{uuid.uuid4().hex}.txt"
    path.write_text(body)
    return StageOutput(
        checks=[
            CheckResult(
                "PR3",
                "PASS" if all(v == 1 for v in buckets.values()) else "FAIL",
                actual=buckets,
            )
        ],
        artifacts=[str(path), str(wave.path)],
    )


HANDLERS = [
    StageHandler("priority_normalize_wait", _reference, _normalize_wait, {}),
    StageHandler(
        "priority_normalize_metrics_wait", _reference, _normalize_metrics_wait, {}
    ),
    StageHandler(
        "priority_normalize_order",
        _normalize_params,
        _normalize_order,
        {},
        checks=frozenset({"PR3", "P6_terminal"}),
    ),
    StageHandler(
        "priority_normalize_metrics",
        _reference,
        _normalization_metrics,
        {},
        checks=frozenset({"PR3"}),
    ),
    StageHandler(
        "priority_order_basic",
        _order_params,
        _order_basic,
        {},
        checks=frozenset({"PR1", "PR2", "PR6", "P6_terminal"}),
    ),
    StageHandler("priority_settled", _reference, _settled, {}),
    StageHandler(
        "priority_pending",
        _pending_params,
        _pending,
        {},
        checks=frozenset({"dispatched"}),
    ),
    StageHandler(
        "priority_expiry", _expiry_params, _expiry, {}, checks=frozenset({"PR8", "P6"})
    ),
    StageHandler("priority_prefill_perf", _perf_params, _prefill_perf, {}),
    StageHandler("priority_fleet", _empty, _fleet, {"prefill": "string"}),
    StageHandler("priority_start", _wave_params, _start, {"requests": "requests"}),
    StageHandler("priority_wait", _reference, _wait, {}),
    StageHandler(
        "priority_completion",
        _completion_params,
        _completion,
        {},
        checks=frozenset({"P6"}),
    ),
    StageHandler(
        "priority_fifo", _reference, _fifo, {}, checks=frozenset({"PR2", "P6_terminal"})
    ),
]
