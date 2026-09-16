"""Bounded scale-out traffic and observations collected before Decode admission."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from ..contracts import CheckResult, StageHandler, StageOutput
from .elastic import BoundedFlow, _snapshot, _validate
from .master import _master_json


def sample(ctx, deadline):
    engines = _snapshot(ctx, deadline)
    master = _master_json(ctx, "single", "/rtp_llm/inflight_status", deadline)
    fields = (
        "running",
        "waiting",
        "accepted",
        "completed",
        "kv_admission_fails",
        "available_blocks",
        "cache_blocks",
        "referenced_blocks",
    )
    decodes = {}
    for name, engine in engines.items():
        if engine.get("role") != "decode":
            continue
        row = {k: engine.get(k) for k in fields}
        if any(type(v) is not int or v < 0 for v in row.values()):
            raise ValueError(f"{name}: missing or invalid Decode counters: {row}")
        if type(engine.get("stopped")) is not bool:
            raise ValueError(f"{name}: missing Decode health")
        row.update(stopped=engine["stopped"], http_addr=engine["http_addr"])
        decodes[name] = row
    endpoints = master.get("decode_endpoints")
    limit = master.get("decode_max_engine_requests")
    if not isinstance(endpoints, list) or type(limit) is not int or limit <= 0:
        raise ValueError("missing live Decode capacity configuration")
    slots = {}
    for endpoint in endpoints:
        used = endpoint.get("engine_capacity_used")
        if type(used) is not int or used < 0:
            raise ValueError("missing live Decode dispatch capacity")
        slots[endpoint["ip_port"]] = used
    return dict(time_s=ctx.clock(), engines=decodes, capacity=slots, limit=limit)


def assess(samples, engine, limit, window_start, window_s):
    """Keep transient violations; later recovery must not erase an earlier peak."""
    errors = []
    selected = []
    seen = False
    seen_master = False
    for item in samples:
        row = item["engines"].get(engine)
        if row is None:
            if seen:
                errors.append("new Decode disappeared")
            continue
        seen = True
        used = item["capacity"].get(row["http_addr"])
        if used is not None:
            seen_master = True
        elif seen_master:
            errors.append("new Decode disappeared from master capacity view")
        if item["limit"] != limit:
            errors.append("live capacity limit differs from case configuration")
        if row["stopped"]:
            errors.append("new Decode stopped")
        if row["running"] + row["waiting"] > limit or (
            used is not None and used > limit
        ):
            errors.append("new Decode exceeded engine capacity")
        if (
            row["kv_admission_fails"]
            or row["available_blocks"] > row["cache_blocks"]
            or row["referenced_blocks"] > row["cache_blocks"]
        ):
            errors.append("new Decode KV admission/capacity failure")
        if selected and any(
            row[k] < selected[-1][1][k] for k in ("accepted", "completed")
        ):
            errors.append("new Decode counters reset")
        selected.append((item["time_s"], row, used))
    if not selected or not seen_master:
        errors.append("new Decode lacks engine/master observations")
    # Four separate windows require continued forward progress, not one lucky request.
    progress = []
    for index in range(4):
        lo = window_start + index * window_s / 4
        hi = window_start + (index + 1) * window_s / 4
        rows = [r for t, r, _ in selected if lo <= t <= hi]
        delta = rows[-1]["completed"] - rows[0]["completed"] if len(rows) >= 2 else 0
        progress.append(delta)
    if any(n <= 0 for n in progress):
        errors.append(
            "new Decode did not complete requests in every observation window"
        )
    # Routed Decode work enters scheduleDecodeCompletion directly; the accepted
    # counter belongs to the public RPC entry and remains zero on this path.
    # Completion deltas prove that the newcomer actually received and served work.
    if not selected or selected[-1][1]["completed"] <= selected[0][1]["completed"]:
        errors.append("new Decode received no requests")
    gaps = [b["time_s"] - a["time_s"] for a, b in zip(samples, samples[1:])]
    if not gaps or max(gaps) > 1.0:
        errors.append("observation gap exceeds 1s")
    return dict(
        errors=sorted(set(errors)),
        progress=progress,
        sample_count=len(selected),
        peak_running_waiting=max(
            (r["running"] + r["waiting"] for _, r, _ in selected), default=0
        ),
        peak_engine_capacity_used=max(
            (n for _, _, n in selected if n is not None), default=0
        ),
        max_sample_gap_s=max(gaps, default=0),
    )


class DecodeFlow(BoundedFlow):
    def __init__(self, ctx, concurrency, output_len):
        super().__init__(
            ctx.ops,
            ctx.env_epoch,
            [],
            interval_s=0.01,
            max_inflight=concurrency,
            clock=ctx.clock,
        )
        self.ctx, self.output_len = ctx, output_len
        self.samples, self.observer_error = [], None
        self.watch_stop = threading.Event()
        self.watch = threading.Thread(
            target=self._observe, name="decode-scale-out-observer", daemon=True
        )
        self.artifact_path = ctx.artifact_dir / "decode-scale-out-requests.json"
        self.observation_path = ctx.artifact_dir / "decode-scale-out-observations.json"

    def _observe(self):
        from ..runtime import Deadline

        end = self.clock() + 180
        try:
            while not self.watch_stop.is_set():
                if self.clock() >= end:
                    raise TimeoutError("Decode observer exceeded its 180s lifetime")
                row = sample(
                    self.ctx, Deadline(min(end, self.clock() + 0.9), clock=self.clock)
                )
                with self._lock:
                    self.samples.append(row)
                self.watch_stop.wait(0.1)
        except Exception as exc:
            self.observer_error = repr(exc)

    def observations(self):
        with self._lock:
            return copy.deepcopy(self.samples)

    def start(self):
        self.watch.start()
        super().start()

    def _pump(self):
        end = self.clock() + 150
        futures = []
        try:
            with ThreadPoolExecutor(max_workers=self.max_inflight) as pool:
                while not self._stop.is_set():
                    if self.clock() >= end:
                        raise TimeoutError("Decode flow exceeded its 150s lifetime")
                    for future in futures:
                        if future.done():
                            future.result()
                    futures = [f for f in futures if not f.done()]
                    if len(futures) < self.max_inflight:
                        rid = self.ops.next_request_id()
                        record = self.issue(rid, self.clock)
                        futures.append(
                            pool.submit(
                                self.run,
                                record,
                                dict(
                                    input_len=512,
                                    output_len=self.output_len,
                                    block_keys=[rid * 100 + 1],
                                ),
                                timeout_s=45,
                                stream_timeout_s=30,
                            )
                        )
                    self._stop.wait(self.interval_s)
                for future in futures:
                    future.result()
        except Exception as exc:
            self.pump_error = repr(exc)
        finally:
            self.done.set()

    def stop(self, deadline, cancel=False):
        try:
            return super().stop(deadline, cancel)
        finally:
            self.watch_stop.set()
            if self.watch.ident is not None:
                self.watch.join(timeout=max(0, deadline.expires_at - self.clock()))
                if self.watch.is_alive():
                    raise TimeoutError("Decode observer did not stop")
            self.observation_path.write_text(
                json.dumps(
                    dict(samples=self.observations(), error=self.observer_error),
                    indent=2,
                )
            )
            self.artifact_path.write_text(json.dumps(self.snapshot_records(), indent=2))


def _start_validate(params, plan):
    p = _validate(
        params, plan, {"concurrency", "output_len"}, {"concurrency", "output_len"}
    )
    for key, low, high in (("concurrency", 2, 128), ("output_len", 64, 512)):
        if type(p[key]) is not int or not low <= p[key] <= high:
            raise ValueError(f"{key} outside bounded Decode flow limits")
    return p


def _start(ctx, params, deadline):
    flow = DecodeFlow(ctx, **params)
    handle = ctx.register_resource(
        "flow", flow, cleanup=lambda d: flow.stop(d, cancel=True)
    )
    # A synchronous baseline precedes both issuance and membership mutation.
    flow.samples.append(sample(ctx, deadline))
    flow.start()
    return StageOutput(output={"flow": handle})


def _validate_flow(params, plan, fields):
    p = _validate(params, plan, fields | {"flow"}, fields | {"flow"})
    plan.reference(p["flow"], "flow")
    return p


def _loaded_validate(params, plan):
    p = _validate_flow(params, plan, {"limit"})
    if type(p["limit"]) is not int or not 2 <= p["limit"] <= 64:
        raise ValueError("Decode capacity must be between 2 and 64")
    return p


def _loaded(ctx, params, deadline):
    flow = ctx.resource(params["flow"], "flow")
    while True:
        deadline.check()
        if flow.observer_error or flow.pump_error:
            raise RuntimeError(flow.observer_error or flow.pump_error)
        rows = flow.observations()[-1]["engines"]
        if len(rows) != ctx.env.spec.n_decode:
            raise ValueError("initial Decode membership differs from configuration")
        loaded = all(
            not r["stopped"]
            and r["completed"] > 0
            and r["running"] + r["waiting"] >= params["limit"] / 2
            for r in rows.values()
        )
        if loaded:
            return StageOutput(
                checks=[CheckResult("old_decodes_loaded", "PASS", actual=rows)]
            )
        deadline.sleep(0.1)


def _window_validate(params, plan):
    p = _validate_flow(params, plan, {"seconds"})
    if type(p["seconds"]) is not int or not 20 <= p["seconds"] <= 60:
        raise ValueError("Decode observation window must be 20..60 seconds")
    return p


def _window(ctx, params, deadline):
    flow = ctx.resource(params["flow"], "flow")
    started = ctx.clock()
    deadline.sleep(params["seconds"])
    if flow.observer_error or flow.pump_error:
        raise RuntimeError(flow.observer_error or flow.pump_error)
    return StageOutput(output={"started_s": started})


def _check_validate(params, plan):
    p = _validate_flow(params, plan, {"engine", "limit", "started_s", "window_s"})
    plan.reference(p["engine"], "string")
    plan.reference(p["started_s"], "number")
    if (
        type(p["limit"]) is not int
        or not 2 <= p["limit"] <= 64
        or type(p["window_s"]) is not int
        or not 20 <= p["window_s"] <= 60
    ):
        raise ValueError("invalid Decode protection limits")
    return p


def _check(ctx, params, deadline):
    flow = ctx.resource(params["flow"], "flow")
    if flow.observer_error or flow.pump_error:
        raise RuntimeError(flow.observer_error or flow.pump_error)
    engine = ctx.resolve(params["engine"])
    rows = flow.observations()
    result = assess(
        rows,
        engine,
        params["limit"],
        ctx.resolve(params["started_s"]),
        params["window_s"],
    )
    now = sample(ctx, deadline)
    current = now["engines"].get(engine)
    idle = (
        bool(current)
        and not current["stopped"]
        and current["running"]
        == current["waiting"]
        == current["referenced_blocks"]
        == 0
    )
    old = set(rows[0]["engines"])
    window = [
        r
        for r in rows
        if ctx.resolve(params["started_s"])
        <= r["time_s"]
        <= ctx.resolve(params["started_s"]) + params["window_s"]
    ]
    old_progress = len(window) >= 2 and all(
        n in window[-1]["engines"]
        and window[-1]["engines"][n]["completed"] > window[0]["engines"][n]["completed"]
        for n in old
    )
    path = ctx.artifact_dir / "decode-scale-out-verdict.json"
    path.write_text(json.dumps(dict(result=result, final=now), indent=2))
    return StageOutput(
        checks=[
            CheckResult(
                "new_decode_protected",
                "FAIL" if result["errors"] else "PASS",
                actual=result,
            ),
            CheckResult(
                "new_decode_drained", "PASS" if idle else "FAIL", actual=current
            ),
            CheckResult("old_decodes_progress", "PASS" if old_progress else "FAIL"),
        ],
        artifacts=[str(path), str(flow.observation_path), str(flow.artifact_path)],
    )


HANDLERS = [
    StageHandler("decode_scale_flow", _start_validate, _start, {"flow": "flow"}),
    StageHandler(
        "decode_scale_loaded",
        _loaded_validate,
        _loaded,
        {},
        checks=frozenset({"old_decodes_loaded"}),
    ),
    StageHandler(
        "decode_scale_window", _window_validate, _window, {"started_s": "number"}
    ),
    StageHandler(
        "decode_scale_check",
        _check_validate,
        _check,
        {},
        checks=frozenset(
            {"new_decode_protected", "new_decode_drained", "old_decodes_progress"}
        ),
    ),
]
