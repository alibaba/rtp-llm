"""Bounded concurrent priority cohorts and observed dispatch-order checks."""

import copy
import json
import math
import threading
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


def _wave_params(p, plan):
    p = _params(p, {"requests", "gap_s"}, {"requests"})
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
                consume="immediate",
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
            deadline.sleep(self.p["gap_s"])

    def _submit(self, item, end):
        try:
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
                rows.append(dict(tag=item["tag"], **record))
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


def _fifo(ctx, p, deadline):
    wave = _wave(ctx, p)
    if not wave.complete:
        raise ValueError("priority FIFO requires completed cohort observation")
    rows = wave.records()
    if len(rows) != len(wave.entries):
        raise ValueError("priority cohort lost a request record")
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
        if len(matches) != 1:
            raise ValueError("missing or ambiguous Prefill lifecycle")
        running = _number(matches[0].get("running_ms"), 0, 1e18)
        dispatch.append((running, ranks[rid], rid))
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


HANDLERS = [
    StageHandler("priority_fleet", _empty, _fleet, {"prefill": "string"}),
    StageHandler("priority_start", _wave_params, _start, {"requests": "requests"}),
    StageHandler("priority_wait", _reference, _wait, {}),
    StageHandler(
        "priority_fifo", _reference, _fifo, {}, checks=frozenset({"PR2", "P6_terminal"})
    ),
]
