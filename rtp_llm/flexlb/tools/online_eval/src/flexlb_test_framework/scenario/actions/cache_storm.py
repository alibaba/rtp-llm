"""Single-hot-family spill probes; all verdict metrics come from engine evidence."""

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, wait

from ..contracts import CheckResult, StageHandler, StageOutput
from .elastic import RecordedRequests, _snapshot, _validate, request_success
from .engine_control import _http


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(windows, records, events, hot_blocks, leader):
    """Issue-window cohorts include every attempt, even when completion is later."""
    index = {}
    for event in events:
        if event.get("event") == "prefill_done":
            if event["rid"] in index:
                raise ValueError("duplicate prefill completion evidence")
            index[event["rid"]] = event
    metrics = []
    for window in windows:
        cohort = [r for r in records if r["window"] == window["id"]]
        hot = [r for r in cohort if r["kind"] == "hot"]
        if not hot or any(not request_success(r) for r in cohort):
            raise ValueError("empty hot cohort or failed request: invalid construction")
        hits, landed, executions = [], [], []
        for record in cohort:
            event = index.get(record["wire_request_id"])
            if event is None or event.get("cancelled") is not False:
                raise ValueError("missing successful engine completion evidence")
            if event["engine_name"] not in window["samples"][0]["engines"]:
                raise ValueError("unknown engine landing")
            address = window["samples"][0]["engines"][event["engine_name"]]["grpc_addr"]
            if address != record["prefill_addr"]:
                raise ValueError("request landing disagrees with engine event")
            if record["kind"] == "hot":
                if event["input_len"] != hot_blocks * 1024:
                    raise ValueError("hot request shape changed across phases")
                hits.append(event["cache_hit_tokens"] >= hot_blocks * 1024)
                landed.append(event["engine_name"])
                if event["engine_name"] == leader:
                    executions.append(event["exec_ms"])
        samples = window["samples"]
        first, last = samples[0], samples[-1]
        evictions, retention = {}, {}
        for name in first["engines"]:
            for field, target in (
                ("cache_evictions", evictions),
                ("cache_retention_evictions", retention),
            ):
                values = [s["engines"][name][field] for s in samples]
                if any(b < a for a, b in zip(values, values[1:])):
                    raise ValueError("cache counter reset within observation window")
                target[name] = values[-1] - values[0]
        elapsed = last["time_s"] - first["time_s"]
        if elapsed <= 0:
            raise ValueError("empty observation duration")
        holders = [
            sum(e["hot_keys"] > 0 for e in s["engines"].values()) for s in samples
        ]
        equivalents = [
            sum(e["hot_keys"] for e in s["engines"].values()) / hot_blocks
            for s in samples
        ]
        metrics.append(
            dict(
                id=window["id"],
                phase=window["phase"],
                index=window["index"],
                hot_requests=len(hot),
                all_requests=len(cohort),
                hits=sum(hits),
                hit_rate=sum(hits) / len(hits),
                leader_share=landed.count(leader) / len(landed),
                landed_counts={n: landed.count(n) for n in sorted(set(landed))},
                eviction_count=sum(evictions.values()),
                eviction_rate=sum(evictions.values()) / elapsed,
                retention_evictions=sum(retention.values()),
                per_engine_evictions=evictions,
                hot_holders=max(holders),
                hot_copy_equivalents=max(equivalents),
                leader_exec_ms=executions,
                leader_busy_share=sum(r["leader_busy"] for r in hot) / len(hot),
                duration_s=elapsed,
            )
        )
    return metrics


def recovery_window(metrics, baseline_rate, consecutive):
    recovery = [m for m in metrics if m["phase"] == "recovery"]
    for i in range(len(recovery) - consecutive + 1):
        if all(m["hit_rate"] >= baseline_rate for m in recovery[i : i + consecutive]):
            return i + 1
    return None


class Storm:
    def __init__(self, ctx, config):
        self.ctx, self.config = ctx, config
        self.leader = "prefill-0"
        self.hot = list(range(910000, 910000 + config["hot_blocks"]))
        self.background = {
            f"prefill-{i}": list(
                range(920000 + i * 100, 920000 + i * 100 + config["background_blocks"])
            )
            for i in range(1, ctx.env.spec.n_prefill)
        }
        self.records = RecordedRequests(ctx.ops, ctx.env_epoch, clock=ctx.clock)
        self.pool = ThreadPoolExecutor(max_workers=config["concurrency"])
        self.futures, self.windows, self.seeds, self.controls = [], [], [], []
        self.issue_index = 0
        self.sampler = ThreadPoolExecutor(max_workers=1)
        self.sample_future = None
        self.latest_sample = None
        self.next_issue = None
        self.last_phase = None
        self.master_config = ctx.env.run_dir / "master_config.json"
        self.master_digest = digest(self.master_config)
        self.summary = None

    def snapshot(self, deadline):
        sample_started = self.ctx.clock()
        all_engines = _snapshot(self.ctx, deadline)
        engines = {}
        for name, row in all_engines.items():
            if row["role"] != "prefill":
                continue
            fields = (
                "cache_evictions",
                "cache_retention_evictions",
                "cache_retention_blocks",
                "cache_blocks",
                "running",
                "waiting",
                "accepted",
                "completed",
                "kv_admission_fails",
                "lack_mem_rejects",
            )
            selected = {k: row.get(k) for k in fields}
            if (
                any(type(v) is not int or v < 0 for v in selected.values())
                or row.get("stopped") is not False
            ):
                raise ValueError("missing cache counters or unavailable prefill engine")
            keys = row.get("cache_key_set")
            if not isinstance(keys, list) or any(type(k) is not int for k in keys):
                raise ValueError("missing engine cache key set")
            selected.update(
                grpc_addr=row["grpc_addr"],
                hot_keys=len(set(keys) & set(self.hot)),
                cache_key_set=keys,
            )
            engines[name] = selected
        if len(engines) != self.ctx.env.spec.n_prefill:
            raise ValueError("prefill membership changed")
        return dict(
            time_s=self.ctx.clock(),
            started_s=sample_started,
            epoch_s=time.time(),
            engines=engines,
        )

    def sample_async(self, deadline):
        # One bounded observer; HTTP latency must not serialize request emission.
        if self.latest_sample is None:
            self.latest_sample = self.snapshot(deadline)
        if self.sample_future is not None and self.sample_future.done():
            self.latest_sample = self.sample_future.result()
            self.sample_future = None
        if self.sample_future is None:
            self.sample_future = self.sampler.submit(self.snapshot, deadline)
        if (
            self.ctx.clock() - self.latest_sample["started_s"]
            > self.config["max_sample_age_s"]
        ):
            raise ValueError("storm engine observation became stale")
        return self.latest_sample

    def control(self, name, deadline, **fields):
        before = digest(self.master_config)
        response = _http(
            self.ctx.ops, "set_perf", deadline, dict(engine=name, **fields)
        )
        after = digest(self.master_config)
        if (
            response.get("status") != "ok"
            or response.get("engine") != name
            or before != self.master_digest
            or after != before
        ):
            raise ValueError("control failed or changed master estimator configuration")
        self.controls.append(
            dict(
                engine=name,
                fields=fields,
                time_s=self.ctx.clock(),
                epoch_s=time.time(),
                master_before=before,
                master_after=after,
                response=response,
            )
        )

    def seed(self, name, keys, deadline):
        ops = self.ctx.ops
        engines = _snapshot(self.ctx, deadline)
        target = engines[name]["grpc_addr"]
        decode = engines["decode-0"]
        host, port = decode["grpc_addr"].rsplit(":", 1)
        rid = ops.next_request_id()
        inp = ops.build_generate_input(
            rid, input_len=len(keys) * 1024, output_len=2, block_keys=keys
        )
        inp.generate_config.role_addrs.add(
            role="DECODE",
            ip=host,
            grpc_port=int(port),
            http_port=int(decode["http_addr"].rsplit(":", 1)[1]),
        )
        started = time.time()
        row = dict(
            rid=rid,
            request_id=rid,
            engine=name,
            keys=keys,
            successful=False,
            purpose="preconditioning",
            route_path="direct",
            prefill=target,
            decode=decode["grpc_addr"],
            input_len=len(keys) * 1024,
            output_len=2,
            send_start_epoch_ms=started * 1000,
            status="pending",
            error="",
        )
        self.seeds.append(row)
        finished = False
        try:
            for out in ops.pb2_grpc.RpcServiceStub(
                ops._channel(target)
            ).GenerateStreamCall(inp, timeout=min(20, deadline.remaining())):
                if out.HasField("error_info") and out.error_info.error_code:
                    raise ValueError("seed request returned business error")
                finished |= any(out.flatten_output.finished)
            if not finished:
                raise ValueError("seed request did not finish")
            row.update(successful=True, status="ok")
        except BaseException as exc:
            row.update(status="exception", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row.update(
                wall_clock_ts=time.time(), total_ms=(time.time() - started) * 1000
            )

    def issue(self, phase, window, before):
        self.futures = [f for f in self.futures if not f.done()]
        if len(self.futures) >= self.config["concurrency"]:
            raise ValueError(
                "traffic generator reached concurrency cap; invalid cadence"
            )
        index = self.issue_index
        self.issue_index += 1
        is_background = (
            index % self.config["background_every"]
            == self.config["background_every"] - 1
        )
        backgrounds = list(self.background.values())
        keys = (
            backgrounds[(index // self.config["background_every"]) % len(backgrounds)]
            if is_background
            else self.hot
        )
        row = self.records.issue(self.ctx.ops.next_request_id(), self.ctx.clock)
        row.update(
            kind="background" if is_background else "hot",
            phase=phase,
            window=window,
            leader_busy=before["engines"][self.leader]["running"]
            + before["engines"][self.leader]["waiting"]
            > 0,
        )
        self.futures.append(
            self.pool.submit(
                self.records.run,
                row,
                dict(input_len=len(keys) * 1024, output_len=2, block_keys=keys),
                timeout_s=60,
                schedule_timeout_s=30,
                stream_timeout_s=45,
            )
        )

    def drain(self, deadline):
        done, pending = wait(self.futures, timeout=deadline.remaining())
        if pending:
            raise TimeoutError("storm requests did not drain")
        for future in done:
            future.result()
        self.futures.clear()

    def evidence_snapshot(self):
        from online_eval.requests import completeness

        rows = self.records.snapshot_records()
        complete = completeness(rows)["complete"] and all(
            "wall_clock_ts" in r for r in self.seeds
        )
        return dict(
            records=[*rows, *[dict(r) for r in self.seeds]],
            complete=complete,
            errors=[] if complete else ["storm producer did not complete all requests"],
            producer_kind="python",
        )

    def save(self):
        path = self.ctx.artifact_dir / "cache-storm-raw.json"
        path.write_text(
            json.dumps(
                dict(
                    config=self.config,
                    windows=self.windows,
                    seeds=self.seeds,
                    controls=self.controls,
                    requests=self.records.snapshot_records(),
                ),
                indent=2,
            )
        )
        return str(path)

    def close(self, deadline):
        self.records.cancel_active("cleanup")
        try:
            if self.sample_future is not None:
                self.sample_future.result(timeout=deadline.remaining())
            self.drain(deadline)
        finally:
            self.sampler.shutdown(wait=False, cancel_futures=True)
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.save()


CONFIG_LIMITS = {
    "hot_blocks": (10, 2, 64),
    "background_blocks": (4, 1, 32),
    "receiver_retention_blocks": (6, 1, 64),
    "concurrency": (128, 2, 256),
    "background_every": (8, 2, 100),
    "normal_ms": (100, 1, 1000),
    "slow_ms": (15000, 1000, 30000),
    "window_s": (2, 1, 10),
    "baseline_windows": (3, 2, 10),
    "saturation_windows": (6, 2, 20),
    "recovery_windows": (10, 2, 30),
    "recovery_consecutive": (2, 1, 5),
    "require_insufficient_retention": (1, 0, 1),
    "calibration_runs": (0, 0, 100),
}


def _prepare_validate(params, plan):
    allowed = (
        set(CONFIG_LIMITS)
        | {"interval_s", "steady_interval_s", "min_busy_share", "max_sample_age_s"}
        | {
            f"{m}_{g}"
            for m in ("hit", "eviction", "holders")
            for g in ("strict", "normal", "loose")
        }
    )
    p = _validate(params, plan, allowed, allowed)
    for key, (_, low, high) in CONFIG_LIMITS.items():
        if type(p[key]) is not int or not low <= p[key] <= high:
            raise ValueError(f"invalid storm configuration {key}")
    for key in allowed - set(CONFIG_LIMITS):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"invalid storm configuration {key}")
    if not 0 < p["max_sample_age_s"] < p["window_s"]:
        raise ValueError("snapshot age budget must be smaller than a window")
    for metric in ("hit", "eviction", "holders"):
        values = [p[f"{metric}_{g}"] for g in ("strict", "normal", "loose")]
        if values != sorted(values, reverse=metric == "hit"):
            raise ValueError("storm grade bands out of order")
    return p


def _prepare(ctx, params, deadline):
    storm = Storm(ctx, params)
    handle = ctx.register_resource("flow", storm, cleanup=storm.close)
    for name in [storm.leader, *storm.background]:
        fields = dict(prefill_fixed_ms=params["normal_ms"])
        if name != storm.leader:
            fields["cache_retention_blocks"] = params["receiver_retention_blocks"]
        storm.control(name, deadline, **fields)
    storm.seed(storm.leader, storm.hot, deadline)
    for name, keys in storm.background.items():
        storm.seed(name, keys, deadline)
    deadline.sleep(2)
    state = storm.snapshot(deadline)
    if state["engines"][storm.leader]["hot_keys"] != len(storm.hot):
        raise ValueError("leader does not hold the complete hot prefix")
    for name, keys in storm.background.items():
        row = state["engines"][name]
        if not set(keys) <= set(row["cache_key_set"]) or row["hot_keys"]:
            raise ValueError("background cache seeding failed")
        if params["require_insufficient_retention"] and not row[
            "cache_retention_blocks"
        ] < len(storm.hot):
            raise ValueError("receiver retention can hold the entire hot family")
    return StageOutput(
        {"storm": handle}, [CheckResult("seeded", "PASS", actual=state)], [storm.save()]
    )


def _ref_validate(params, plan, extra=()):
    p = _validate(params, plan, {"storm", *extra}, {"storm", *extra})
    plan.reference(p["storm"], "flow")
    return p


def _window_validate(params, plan):
    p = _ref_validate(params, plan, ("phase", "index"))
    if (
        p["phase"] not in ("baseline", "saturation", "recovery")
        or type(p["index"]) is not int
        or not 0 <= p["index"] < 30
    ):
        raise ValueError("invalid storm window")
    return p


def _window(ctx, params, deadline):
    storm = ctx.resource(params["storm"], "flow")
    if storm.last_phase != params["phase"]:
        # Drains and speed controls between phases are deliberate quiet periods.
        # Start their next observation/cadence clock only after a fresh snapshot.
        if storm.sample_future is not None:
            storm.sample_future.result(timeout=deadline.remaining())
            storm.sample_future = None
        storm.latest_sample = storm.snapshot(deadline)
    state = storm.sample_async(deadline)
    started = ctx.clock()
    row = dict(
        id=f'{params["phase"]}-{params["index"]}',
        phase=params["phase"],
        index=params["index"],
        samples=[state],
    )
    storm.windows.append(row)
    interval = storm.config[
        "interval_s" if params["phase"] == "saturation" else "steady_interval_s"
    ]
    if storm.last_phase != params["phase"]:
        storm.next_issue = started
        storm.last_phase = params["phase"]
    while ctx.clock() - started < storm.config["window_s"]:
        state = storm.sample_async(deadline)
        if state is not row["samples"][-1]:
            row["samples"].append(state)
        if ctx.clock() >= storm.next_issue:
            if ctx.clock() - storm.next_issue > interval:
                raise ValueError("arrival cadence missed a complete interval")
            storm.issue(params["phase"], row["id"], state)
            storm.next_issue += interval
        deadline.sleep(0.03)
    # Keep cadence across adjacent windows. Synchronous final sampling and JSON
    # persistence used to consume the next window's first emission interval.
    state = storm.sample_async(deadline)
    if state is not row["samples"][-1]:
        row["samples"].append(state)
    return StageOutput()


def _speed_validate(params, plan):
    p = _ref_validate(params, plan, ("slow",))
    if type(p["slow"]) is not bool:
        raise ValueError("slow must be boolean")
    return p


def _speed(ctx, params, deadline):
    storm = ctx.resource(params["storm"], "flow")
    storm.control(
        storm.leader,
        deadline,
        prefill_fixed_ms=storm.config["slow_ms" if params["slow"] else "normal_ms"],
    )
    return StageOutput(artifacts=[storm.save()])


def _drain(ctx, params, deadline):
    storm = ctx.resource(params["storm"], "flow")
    storm.drain(deadline)
    good = all(
        request_success(r) and r["consumer_exit_s"] is not None
        for r in storm.records.snapshot_records()
    )
    return StageOutput(
        checks=[CheckResult("zero_errors", "PASS" if good else "FAIL")],
        artifacts=[storm.save()],
    )


def _verdict(ctx, params, deadline):
    storm = ctx.resource(params["storm"], "flow")
    deadline.check()
    events_path = ctx.env.run_dir / "engine_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    config = storm.config
    metrics = summarize(
        storm.windows,
        storm.records.snapshot_records(),
        events,
        config["hot_blocks"],
        storm.leader,
    )
    baseline = [m for m in metrics if m["phase"] == "baseline"]
    saturation = [m for m in metrics if m["phase"] == "saturation"]
    recovery = recovery_window(
        metrics, min(m["hit_rate"] for m in baseline), config["recovery_consecutive"]
    )
    normal_exec = [x for m in baseline for x in m["leader_exec_ms"]]
    changes = [c for c in storm.controls if c["engine"] == storm.leader]
    slow_start, slow_end = changes[-2]["epoch_s"], changes[-1]["epoch_s"]
    slow_exec = [
        e["exec_ms"]
        for e in events
        if e.get("event") == "prefill_done"
        and e["engine_name"] == storm.leader
        and slow_start * 1000 <= e["prefill_start_ms"] < slow_end * 1000
    ]
    baseline_ok = all(m["hit_rate"] == 1 and m["leader_share"] == 1 for m in baseline)
    busy = sum(m["leader_busy_share"] * m["hot_requests"] for m in saturation) / sum(
        m["hot_requests"] for m in saturation
    )
    injection = (
        bool(normal_exec and slow_exec)
        and min(slow_exec) >= 0.9 * config["slow_ms"]
        and max(normal_exec) <= 1.1 * config["normal_ms"]
        and busy >= config["min_busy_share"]
        and digest(storm.master_config) == storm.master_digest
    )
    summary = dict(
        config=config,
        p=ctx.env.spec.n_prefill,
        profile=ctx.instance["profile"],
        metrics=metrics,
        baseline_valid=baseline_ok,
        injection_valid=injection,
        leader_busy_share=busy,
        recovery_window=recovery,
        recovery_valid=baseline_ok and recovery is not None,
        hit_min=min(m["hit_rate"] for m in saturation),
        eviction_peak=max(m["eviction_rate"] for m in saturation),
        holders_peak=max(m["hot_holders"] for m in saturation),
        spread_engines=sorted({n for m in saturation for n in m["landed_counts"]}),
        successful_requests=len(storm.records.snapshot_records()) + len(storm.seeds),
        master_config_sha256=storm.master_digest,
    )
    storm.summary = summary
    path = ctx.artifact_dir / "cache-storm-summary.json"
    path.write_text(json.dumps(summary, indent=2))
    # A portable, bounded copy avoids dependence on the environment teardown path.
    event_copy = ctx.artifact_dir / "cache-storm-engine-events.jsonl"
    event_copy.write_text(events_path.read_text())
    return StageOutput(
        checks=[
            CheckResult(
                "baseline",
                "PASS" if baseline_ok else "FAIL",
                actual=[m["hit_rate"] for m in baseline],
            ),
            CheckResult(
                "hidden_slowdown",
                "PASS" if injection else "FAIL",
                actual=dict(
                    normal_exec=normal_exec, slow_exec=slow_exec, leader_busy_share=busy
                ),
            ),
            CheckResult(
                "recovery",
                "PASS" if baseline_ok and recovery is not None else "FAIL",
                actual=recovery,
            ),
            CheckResult(
                "calibrated",
                "PASS" if config["calibration_runs"] >= 3 else "FAIL",
                actual=config["calibration_runs"],
            ),
        ],
        artifacts=[str(path), str(event_copy)],
    )


def _health(ctx, params, deadline):
    storm = ctx.resource(params["storm"], "flow")
    grade = ctx.instance.get("grade", "normal")
    summary, cfg = storm.summary, storm.config
    checks = []
    for name, metric, upper in (
        ("hit", "hit_min", False),
        ("eviction", "eviction_peak", True),
        ("holders", "holders_peak", True),
    ):
        actual, expected = summary[metric], cfg[f"{name}_{grade}"]
        passed = actual <= expected if upper else actual >= expected
        checks.append(
            CheckResult(
                name, "PASS" if passed else "FAIL", actual=actual, expected=expected
            )
        )
    resolved = all(check.status == "PASS" for check in checks)
    message = (
        "REVIEW REQUIRED: healthy storm bands passed; verify the scheduler fix and retire the probe marker."
        if resolved
        else "Healthy cache contract violated; finding confirmed."
    )
    path = ctx.artifact_dir / "cache-storm-probe.json"
    path.write_text(
        json.dumps(
            dict(
                state="resolved" if resolved else "confirmed",
                review_required=resolved,
                message=message,
            ),
            indent=2,
        )
    )
    if resolved:
        print(message, flush=True)
    return StageOutput(checks=checks, artifacts=[str(path)])


HANDLERS = [
    StageHandler(
        "storm_prepare",
        _prepare_validate,
        _prepare,
        {"storm": "flow"},
        checks=frozenset({"seeded"}),
    ),
    StageHandler("storm_window", _window_validate, _window, {}),
    StageHandler("storm_speed", _speed_validate, _speed, {}),
    StageHandler(
        "storm_drain", _ref_validate, _drain, {}, checks=frozenset({"zero_errors"})
    ),
    StageHandler(
        "storm_validate",
        _ref_validate,
        _verdict,
        {},
        checks=frozenset({"baseline", "hidden_slowdown", "recovery", "calibrated"}),
    ),
    StageHandler(
        "storm_health",
        _ref_validate,
        _health,
        {},
        checks=frozenset({"hit", "eviction", "holders"}),
    ),
]
