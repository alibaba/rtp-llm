"""Bounded balance windows and explicit steady-recovery predicates."""

import json
import math
import statistics
import threading
import time

from ..contracts import CheckResult, StageHandler, StageOutput


def empty(params, plan):
    from .elastic import _validate

    return _validate(params, plan, set())


def flow_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"interval_s", "stream_timeout_s"}, {"interval_s"})
    if type(p["interval_s"]) not in (int, float) or p["interval_s"] not in (
        0.1,
        0.2,
        0.5,
    ):
        raise ValueError("balance flow interval must be .1, .2 or .5 seconds")
    p.setdefault("stream_timeout_s", 30)
    if type(p["stream_timeout_s"]) not in (int, float) or p["stream_timeout_s"] not in (
        10,
        30,
    ):
        raise ValueError("balance flow stream timeout must be 10 or 30")
    return p


def flow_start(ctx, params, deadline):
    from .elastic import BoundedFlow

    class Flow(BoundedFlow):
        def _pump(self):
            end = self.clock() + 1200
            try:
                while not self._stop.is_set():
                    if self.clock() >= end:
                        raise TimeoutError("balance pump exceeded 1200s lifetime")
                    rid = self.ops.next_request_id()
                    record = self.issue(rid, self.clock)
                    self.run(
                        record,
                        dict(input_len=2048, output_len=2, block_keys=[rid * 100 + 1]),
                        timeout_s=min(
                            30 + params["stream_timeout_s"], end - self.clock()
                        ),
                        schedule_timeout_s=30,
                        stream_timeout_s=params["stream_timeout_s"],
                    )
                    self._stop.wait(self.interval_s)
            except BaseException as exc:
                self.pump_error = repr(exc)
            finally:
                self.done.set()

    deadline.check()
    flow = Flow(
        ctx.ops,
        ctx.env_epoch,
        [],
        interval_s=params["interval_s"],
        max_inflight=1,
        clock=ctx.clock,
    )
    path = ctx.artifact_dir / f"elastic-balance-flow-{time.time_ns()}.json"

    def finish(d):
        try:
            flow.stop(d, cancel=True)
            if flow.pump_error:
                raise RuntimeError(flow.pump_error)
        finally:
            path.write_text(json.dumps(flow.snapshot_records(), indent=2))

    handle = ctx.register_resource("flow", flow, cleanup=finish)
    try:
        flow.start()
    except BaseException:
        flow.done.set()
        raise
    return StageOutput(output=dict(flow=handle))


def observe_start(ctx, params, deadline):
    from .elastic import ElasticMetrics

    deadline.check()
    started_s = ctx.clock()
    metrics = ElasticMetrics(ctx, max_duration_s=1200)
    handle = ctx.register_resource("observation", metrics, cleanup=metrics.stop)
    try:
        metrics.thread.start()
    except BaseException:
        metrics.done.set()
        raise
    return StageOutput(output=dict(observation=handle, started_s=started_s))


def window_validate(params, plan):
    from .elastic import _validate

    p = _validate(
        params,
        plan,
        {"observation", "duration_s", "since"},
        {"observation", "duration_s"},
    )
    plan.reference(p["observation"], "observation")
    if "since" in p:
        plan.reference(p["since"], "number")
    if type(p["duration_s"]) not in (int, float) or p["duration_s"] not in (20, 60):
        raise ValueError("balance window must be 20 or 60 seconds")
    return p


def window(ctx, params, deadline):
    metrics = ctx.resource(params["observation"], "observation")
    now = ctx.clock()
    start = ctx.resolve(params["since"]) if "since" in params else now
    if (
        type(start) not in (int, float)
        or not math.isfinite(start)
        or not 0 <= now - start <= 1200
    ):
        raise ValueError("balance window has an invalid start anchor")
    deadline.sleep(params["duration_s"])
    end = ctx.clock()
    result = dict(start_s=start, end_s=end, data=metrics.snapshot())
    path = ctx.artifact_dir / f"elastic-balance-window-{time.time_ns()}.json"
    path.write_text(json.dumps(result, indent=2))
    return StageOutput(
        output=dict(window=ctx.register_resource("snapshot", result, historical=True)),
        artifacts=[str(path)],
    )


def finish_window_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"window", "observation"}, {"window", "observation"})
    plan.reference(p["window"], "snapshot")
    plan.reference(p["observation"], "observation")
    return p


def finish_window(ctx, params, deadline):
    """Extend an existing window through a completed flow-stop stage."""
    deadline.check()
    previous = ctx.resource(params["window"], "snapshot")
    metrics = ctx.resource(params["observation"], "observation")
    end = ctx.clock()
    if not previous["end_s"] <= end <= previous["start_s"] + 1200:
        raise ValueError("invalid final balance window end")
    result = dict(start_s=previous["start_s"], end_s=end, data=metrics.snapshot())
    path = ctx.artifact_dir / f"elastic-balance-final-window-{time.time_ns()}.json"
    path.write_text(json.dumps(result, indent=2))
    return StageOutput(
        output=dict(window=ctx.register_resource("snapshot", result, historical=True)),
        artifacts=[str(path)],
    )


def points(window, names, metric, start=None, end=None):
    """Keep missing series, counter resets and sample gaps distinguishable from zero."""
    start = window["start_s"] if start is None else start
    end = window["end_s"] if end is None else end
    if start >= end:
        raise ValueError("empty/inverted balance window")
    data = window["data"]
    if any(start <= e["time_s"] <= end for e in data["errors"]):
        raise ValueError("balance observation contains acquisition errors")
    samples = [s for s in data["samples"] if start <= s["time_s"] <= end]
    if len(samples) < max(2, int((end - start) / 2)):
        raise ValueError("balance window has insufficient samples")
    stamps = [start] + [s["time_s"] for s in samples] + [end]
    if any(b < a or b - a > 2.5 for a, b in zip(stamps, stamps[1:])):
        raise ValueError("balance window has unordered samples or excessive gaps")
    result = {name: [] for name in names}
    for sample in samples:
        for name in names:
            row = sample["engines"].get(name)
            if not isinstance(row, dict):
                raise ValueError(f"balance window lacks engine {name}")
            if metric == "occupancy":
                total, available = row.get("mock_engine_cache_blocks"), row.get(
                    "mock_engine_available_blocks"
                )
                if (
                    any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in (total, available)
                    )
                    or not 0 <= available <= total
                    or total <= 0
                ):
                    raise ValueError(f"invalid occupancy counters for {name}")
                value = (total - available) / total
            else:
                value = row.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"missing/invalid {metric} for {name}")
            result[name].append((sample["time_s"], value))
    return result


def deltas(series):
    result = {}
    for name, seq in series.items():
        if any(b[1] < a[1] for a, b in zip(seq, seq[1:])):
            raise ValueError(f"counter reset for {name}")
        result[name] = seq[-1][1] - seq[0][1]
    return result


def shares(window, names, start=None, end=None):
    values = deltas(points(window, names, "mock_engine_completed_total", start, end))
    total = sum(values.values())
    return {n: v / total for n, v in values.items()} if total else None


def spread(series):
    means = [statistics.mean(v for _, v in seq) for seq in series.values()]
    return max(means) - min(means)


def baseline_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"window"}, {"window"})
    plan.reference(p["window"], "snapshot")
    return p


def baseline(ctx, params, deadline):
    deadline.check()
    before = ctx.resource(params["window"], "snapshot")
    names = [f"decode-{i}" for i in range(4)]
    share = shares(before, names)
    occ = spread(points(before, names, "occupancy"))
    return StageOutput(
        checks=[
            CheckResult(
                "nonempty_decode_traffic",
                "PASS" if share else "FAIL",
                actual=share,
                evidence=dict(occupancy_spread=occ),
            )
        ]
    )


def verdict_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"baseline", "steady"}, {"baseline", "steady"})
    for name in p:
        plan.reference(p[name], "snapshot")
    return p


def oscillations(subshares, target):
    bad = []
    for name, seq in subshares.items():
        for (i, s1), (j, s2) in zip(seq, seq[1:]):
            d1, d2 = s1 - target, s2 - target
            if j == i + 1 and abs(d1) > 0.10 and abs(d2) > 0.10 and d1 * d2 > 0:
                bad.append(dict(engine=name, first=i, second=j, deviations=[d1, d2]))
    return bad


def optional_stat(compute):
    try:
        return dict(value=compute(), unavailable=None)
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return dict(value=None, unavailable=str(exc))


def observations(window, names, start, end):
    def cv():
        means = [
            statistics.mean(v for _, v in seq)
            for seq in points(
                window, names, "mock_engine_decode_ms_avg", start, end
            ).values()
        ]
        return statistics.stdev(means) / statistics.mean(means)

    def hit():
        engine_names = sorted(
            {
                n
                for s in window["data"]["samples"]
                if start <= s["time_s"] <= end
                for n in s["engines"]
            }
        )
        hits = sum(
            deltas(
                points(
                    window, engine_names, "mock_engine_cache_key_hits_total", start, end
                )
            ).values()
        )
        asked = sum(
            deltas(
                points(
                    window,
                    engine_names,
                    "mock_engine_cache_keys_requested_total",
                    start,
                    end,
                )
            ).values()
        )
        return hits / asked

    def tps():
        means = [
            statistics.mean(v for _, v in seq)
            for seq in points(
                window, names, "rtp_llm_generate_tps", start, end
            ).values()
        ]
        return max(means) / min(means)

    return dict(
        exec_cv=optional_stat(cv),
        hit_rate=optional_stat(hit),
        generate_tps_ratio=optional_stat(tps),
    )


def verdict(ctx, params, deadline):
    deadline.check()
    base, steady = (ctx.resource(params[k], "snapshot") for k in ("baseline", "steady"))
    old, survivors = [f"decode-{i}" for i in range(4)], [
        f"decode-{i}" for i in range(1, 4)
    ]
    base_share = shares(base, old)
    if base_share is None:
        raise ValueError("baseline decode traffic is empty")
    if abs((steady["end_s"] - steady["start_s"]) - 60) > 1:
        raise ValueError("steady recovery requires a pure 60s window")
    tail = steady["start_s"] + 40
    tail_share = shares(steady, survivors, tail)
    base_spread = spread(points(base, old, "occupancy"))
    occ = points(steady, survivors, "occupancy", tail)
    depth = points(steady, survivors, "mock_engine_waiting", tail)
    cap = max(max(base_share.values()) + 0.10, 1 / 3 + 0.15)
    subshares = {n: [] for n in survivors}
    for i in range(20):
        lo = steady["start_s"] + i * 3
        share = shares(steady, survivors, lo, lo + 3)
        if share is not None:
            for name, value in share.items():
                subshares[name].append((i, value))
    bad = oscillations(subshares, 1 / 3)
    base_deviations = []
    for i in range(6):
        lo = base["start_s"] + i * 3
        share = shares(base, old, lo, lo + 3)
        if share is not None:
            base_deviations.extend(abs(v - 0.25) for v in share.values())
    base_obs = observations(base, old, base["start_s"], base["end_s"])
    steady_obs = observations(steady, survivors, tail, steady["end_s"])
    swing = max(
        (abs(v - 1 / 3) for seq in subshares.values() for _, v in seq), default=None
    )
    base_swing = max(base_deviations, default=None)
    base_cv = base_obs["exec_cv"]["value"]
    base_hit = base_obs["hit_rate"]["value"]
    evidence = dict(
        baseline_share=base_share,
        tail_share=tail_share,
        share_cap=cap,
        baseline_occ_spread=base_spread,
        tail_occ_spread=spread(occ),
        subshares=subshares,
        oscillation_violations=bad,
        observations=dict(
            baseline=base_obs,
            steady=steady_obs,
            swing=swing,
            swing_cap=base_swing + 0.10 if base_swing is not None else None,
            exec_cv_cap=max((base_cv or 0) + 0.10, 0.25),
            hit_floor=base_hit - 0.15 if base_hit is not None else None,
            tps_ratio_cap=1.5,
        ),
    )
    predicates = dict(
        nonempty_decode_traffic=tail_share is not None,
        share_max=tail_share is not None and max(tail_share.values()) <= cap,
        share_min=tail_share is not None and min(tail_share.values()) >= 0.10,
        waiting_peak=max(v for seq in depth.values() for _, v in seq) <= 2,
        occupancy_spread=spread(occ) <= base_spread + 0.05,
        occupancy_peak=max(v for seq in occ.values() for _, v in seq) <= 0.95,
        oscillation=not bad,
    )
    path = ctx.artifact_dir / f"elastic-steady-verdict-{time.time_ns()}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if v else "FAIL", evidence=evidence)
            for k, v in predicates.items()
        ],
        artifacts=[str(path)],
    )


def remove_validate(params, plan):
    from .elastic import _remove_validate

    return _remove_validate(params, plan)


def remove(ctx, params, deadline):
    from .elastic import _mutation
    from .elastic_concurrent import mutation_http

    return _mutation(ctx, params, deadline, "remove", request_http=mutation_http)


def mark(ctx, params, deadline):
    deadline.check()
    return StageOutput(output=dict(time_s=ctx.clock()))


HANDLERS = [
    StageHandler(
        "elastic_balance_finish_window",
        finish_window_validate,
        finish_window,
        {"window": "snapshot"},
    ),
    StageHandler("elastic_balance_mark", empty, mark, {"time_s": "number"}),
    StageHandler(
        "elastic_balance_remove",
        remove_validate,
        remove,
        {"engine": "string", "port": "integer", "mutation": "snapshot"},
        checks=frozenset({"membership"}),
    ),
    StageHandler("elastic_balance_flow", flow_validate, flow_start, {"flow": "flow"}),
    StageHandler(
        "elastic_balance_observe",
        empty,
        observe_start,
        {"observation": "observation", "started_s": "number"},
    ),
    StageHandler(
        "elastic_balance_window", window_validate, window, {"window": "snapshot"}
    ),
    StageHandler(
        "elastic_steady_baseline",
        baseline_validate,
        baseline,
        {},
        checks=frozenset({"nonempty_decode_traffic"}),
    ),
    StageHandler(
        "elastic_steady_verdict",
        verdict_validate,
        verdict,
        {},
        checks=frozenset(
            {
                "nonempty_decode_traffic",
                "share_max",
                "share_min",
                "waiting_peak",
                "occupancy_spread",
                "occupancy_peak",
                "oscillation",
            }
        ),
    ),
]
