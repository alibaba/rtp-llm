"""Pre-request hit evidence, window masks and cache counters for staged KV programs."""

import math

from ...grade import GradeReport
from ..contracts import CheckResult, StageHandler, StageOutput
from ..runtime import Deadline
from .elastic import request_success
from .kv import _artifact, _engine, _fields, _keys, _target


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bands(value, upper=False, ceiling=1):
    if (
        not isinstance(value, dict)
        or set(value) != {"strict", "normal", "loose"}
        or any(not _finite(v) or not 0 <= v <= ceiling for v in value.values())
    ):
        raise ValueError("invalid explicit KV grade bands")
    ordered = (
        value["strict"] <= value["normal"] <= value["loose"]
        if upper
        else value["strict"] >= value["normal"] >= value["loose"]
    )
    if not ordered:
        raise ValueError("KV grade bands are not ordered")


def _one(ctx, reference):
    rows = ctx.resource(reference, "requests").snapshot_records()
    if (
        len(rows) != 1
        or rows[0].get("consumer_completion_verified") is not True
        or not isinstance(rows[0].get("prefill_addr"), str)
        or not rows[0]["prefill_addr"]
    ):
        raise ValueError("KV sample requires one verified terminal landing")
    return rows[0]


def _hit_validate(params, plan):
    p = _fields(
        params,
        {"snapshot", "requests", "keys", "min_contiguous"},
        {"snapshot", "requests", "keys", "min_contiguous"},
    )
    plan.reference(p["snapshot"], "kv_snapshot")
    plan.reference(p["requests"], "requests")
    _keys(p["keys"])
    if type(p["min_contiguous"]) is not int or not 1 <= p["min_contiguous"] <= len(
        p["keys"]
    ):
        raise ValueError("invalid contiguous-prefix hit threshold")
    return p


def hit_observe(ctx, params, deadline):
    deadline.check()
    before = ctx.resource(params["snapshot"], "kv_snapshot")
    row = _one(ctx, params["requests"])
    if (
        not _finite(before.get("sampled_s"))
        or not _finite(row.get("issued_s"))
        or before["sampled_s"] > row["issued_s"]
    ):
        raise ValueError("cache hit evidence must precede request issue")
    if row.get("request_shape", {}).get("block_keys") != params["keys"]:
        raise ValueError("hit observation keys differ from issued request shape")
    matches = [
        (name, engine)
        for name, engine in before["engines"].items()
        if engine["grpc_addr"] == row["prefill_addr"]
    ]
    if len(matches) != 1:
        raise ValueError("pre-request snapshot lacks a unique landing identity")
    name, engine = matches[0]
    cached = set(engine["cache_key_set"])
    run = 0
    for key in params["keys"]:
        if key not in cached:
            break
        run += 1
    sample = dict(
        record=row,
        before=before,
        keys=params["keys"],
        landed=name,
        contiguous=run,
        hit=run >= params["min_contiguous"],
        success=request_success(row),
        min_contiguous=params["min_contiguous"],
    )
    return StageOutput(
        {"sample": ctx.register_resource("kv_hit", sample)},
        artifacts=[_artifact(ctx, "kv-pre-request-hit", sample)],
    )


def _rate_validate(params, plan):
    p = _fields(
        params, {"samples", "min_samples", "bands"}, {"samples", "min_samples", "bands"}
    )
    if not isinstance(p["samples"], list) or not 1 <= len(p["samples"]) <= 200:
        raise ValueError("hit rate needs bounded explicit observations")
    for ref in p["samples"]:
        plan.reference(ref, "kv_hit")
    if type(p["min_samples"]) is not int or not 1 <= p["min_samples"] <= 200:
        raise ValueError("hit rate needs a positive sample floor")
    _bands(p["bands"])
    return p


def hit_rate(ctx, params, deadline):
    deadline.check()
    samples = [ctx.resource(ref, "kv_hit") for ref in params["samples"]]
    if len({s["record"]["wire_request_id"] for s in samples}) != len(samples):
        raise ValueError("hit rate cannot count duplicate requests")
    rate = sum(s["hit"] for s in samples) / len(samples)
    complete = len(samples) >= params["min_samples"] and all(
        s["success"] for s in samples
    )
    report = GradeReport(run_grade=ctx.instance.get("grade", "normal"))
    passed = report.check("M3", rate, bands=params["bands"])
    evidence = dict(
        complete=True,
        sample_count=len(samples),
        min_samples=params["min_samples"],
        samples=samples,
        grade=report.run_grade,
        achieved=report.achieved,
        bands=params["bands"],
    )
    return StageOutput(
        {"rate": rate},
        [
            CheckResult(
                "P6",
                "PASS" if complete else "FAIL",
                actual=complete,
                expected=True,
                evidence=evidence,
            ),
            CheckResult(
                "M3",
                "PASS" if complete and passed else "FAIL",
                actual=rate,
                expected=params["bands"][report.run_grade],
                evidence=evidence,
            ),
        ],
        [_artifact(ctx, "kv-hit-rate", evidence)],
    )


def _families(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ValueError("expected bounded explicit cache families")
    seen = set()
    for keys in value:
        _keys(keys)
        if seen & set(keys):
            raise ValueError("cache families must be disjoint")
        seen.update(keys)


def _mask(data, families):
    return {
        name: [bool(set(keys) & set(engine["cache_key_set"])) for keys in families]
        for name, engine in data["engines"].items()
    }


def _replication_validate(params, plan):
    p = _fields(
        params,
        {"snapshot", "families", "bands", "max_holders"},
        {"snapshot", "families", "bands", "max_holders"},
    )
    plan.reference(p["snapshot"], "kv_snapshot")
    _families(p["families"])
    _bands(p["bands"], upper=True, ceiling=32)
    if type(p["max_holders"]) is not int or not 1 <= p["max_holders"] <= 32:
        raise ValueError("invalid holder structural cap")
    return p


def replication(ctx, params, deadline):
    deadline.check()
    data = ctx.resource(params["snapshot"], "kv_snapshot")
    masks = _mask(data, params["families"])
    counts = [
        sum(values[f] for values in masks.values())
        for f in range(len(params["families"]))
    ]
    mean = sum(counts) / len(counts)
    maximum = max(counts)
    report = GradeReport(run_grade=ctx.instance.get("grade", "normal"))
    passed = report.check("P5", mean, bands=params["bands"])
    evidence = dict(
        complete=True,
        sample_count=len(counts),
        min_samples=len(params["families"]),
        snapshot=data,
        masks=masks,
        counts=counts,
        bands=params["bands"],
        grade=report.run_grade,
        achieved=report.achieved,
    )
    return StageOutput(
        {"mean": mean, "maximum": maximum},
        [
            CheckResult(
                "P5",
                "PASS" if passed else "FAIL",
                actual=mean,
                expected=params["bands"][report.run_grade],
                evidence=evidence,
            ),
            CheckResult(
                "structural_cap",
                "PASS" if maximum <= params["max_holders"] else "FAIL",
                actual=maximum,
                expected=params["max_holders"],
                evidence=evidence,
            ),
        ],
        [_artifact(ctx, "kv-replication", evidence)],
    )


def _transitions_validate(params, plan):
    p = _fields(params, {"snapshots", "families"}, {"snapshots", "families"})
    if not isinstance(p["snapshots"], list) or not 2 <= len(p["snapshots"]) <= 100:
        raise ValueError("window transitions need two or more bounded snapshots")
    for ref in p["snapshots"]:
        plan.reference(ref, "kv_snapshot")
    _families(p["families"])
    return p


def transitions(ctx, params, deadline):
    deadline.check()
    if len({ctx.resolve(ref)["id"] for ref in params["snapshots"]}) != len(
        params["snapshots"]
    ):
        raise ValueError("window transitions require distinct snapshots")
    snapshots = [ctx.resource(ref, "kv_snapshot") for ref in params["snapshots"]]
    times = [s["sampled_s"] for s in snapshots]
    if any(not _finite(t) for t in times) or times != sorted(times):
        raise ValueError("window snapshots must be chronological")
    if any(set(s["engines"]) != set(snapshots[0]["engines"]) for s in snapshots):
        raise ValueError(
            "window membership changed without an explicit migration contract"
        )
    masks = [_mask(s, params["families"]) for s in snapshots]
    flips = sum(
        before[n][f] != after[n][f]
        for before, after in zip(masks, masks[1:])
        for n in before
        for f in range(len(params["families"]))
    )
    evidence = dict(
        snapshots=snapshots,
        masks=masks,
        flips=flips,
        complete=True,
        sample_count=len(masks),
        min_samples=2,
    )
    return StageOutput(
        {"flips": flips}, artifacts=[_artifact(ctx, "kv-window-transitions", evidence)]
    )


def _statistics_validate(params, plan):
    p = _fields(params, {"snapshot", "engine"}, {"snapshot", "engine"})
    plan.reference(p["snapshot"], "kv_snapshot")
    _engine(p["engine"], plan)
    return p


def statistics(ctx, params, deadline):
    deadline.check()
    data = ctx.resource(params["snapshot"], "kv_snapshot")
    name = _target(ctx, params["engine"])
    row = data["engines"][name]
    values = {k: row.get(k) for k in ("cache_keys", "cache_evictions")}
    if any(type(v) is not int or v < 0 for v in values.values()):
        raise ValueError("cache counters unavailable or malformed")
    return StageOutput(
        {"keys": values["cache_keys"], "evictions": values["cache_evictions"]},
        artifacts=[
            _artifact(
                ctx,
                "kv-cache-counters",
                dict(snapshot=data, engine=name, values=values),
            )
        ],
    )


def _retry_validate(params, plan):
    p = _fields(
        params,
        {
            "anchor",
            "candidate",
            "keys",
            "input_len",
            "output_len",
            "settle_s",
            "request_timeout_s",
        },
        {
            "anchor",
            "candidate",
            "keys",
            "input_len",
            "output_len",
            "settle_s",
            "request_timeout_s",
        },
    )
    plan.reference(p["anchor"], "requests")
    plan.reference(p["candidate"], "requests")
    _keys(p["keys"])
    for k in ("input_len", "output_len"):
        if type(p[k]) is not int or not 1 <= p[k] <= 1000000:
            raise ValueError("invalid retry shape")
    if (
        not _finite(p["settle_s"])
        or not 0 <= p["settle_s"] <= 10
        or not _finite(p["request_timeout_s"])
        or not 0 < p["request_timeout_s"] <= 60
    ):
        raise ValueError("invalid bounded retry timing")
    return p


def retry_misdirected(ctx, params, deadline):
    anchor, candidate = (_one(ctx, params[k]) for k in ("anchor", "candidate"))
    if not request_success(anchor) or not request_success(candidate):
        raise ValueError(
            "affinity retry is only for successful but misdirected traffic"
        )
    if anchor["wire_request_id"] == candidate["wire_request_id"]:
        raise ValueError("retry anchor and candidate must be distinct requests")
    expected = dict(
        input_len=params["input_len"],
        output_len=params["output_len"],
        block_keys=params["keys"],
    )
    if any(
        any(row.get("request_shape", {}).get(k) != v for k, v in expected.items())
        for row in (anchor, candidate)
    ):
        raise ValueError("retry shape differs from anchor or candidate")
    handle = ctx.resolve(params["candidate"])
    retried = anchor["prefill_addr"] != candidate["prefill_addr"]
    if retried:
        deadline.sleep(params["settle_s"])
        shape = dict(
            count=1,
            input_len=params["input_len"],
            output_len=params["output_len"],
            block_keys=params["keys"],
            consume="immediate",
            schedule_timeout_s=30,
            stream_timeout_s=params["request_timeout_s"],
        )
        handle = ctx.backend.start_requests(ctx, shape, deadline)
        wait_deadline = Deadline(
            min(deadline.expires_at, ctx.clock() + params["request_timeout_s"]),
            ctx.clock,
            ctx.sleeper,
            deadline.cancelled,
        )
        ctx.backend.wait_requests(ctx, ctx.resource(handle, "requests"), wait_deadline)
    final = _one(ctx, handle)
    return StageOutput(
        {"requests": handle, "retried": retried},
        artifacts=[
            _artifact(
                ctx,
                "kv-single-affinity-retry",
                dict(anchor=anchor, candidate=candidate, final=final, retried=retried),
            )
        ],
    )


HANDLERS = [
    StageHandler("kv_hit_observe", _hit_validate, hit_observe, {"sample": "kv_hit"}),
    StageHandler(
        "kv_hit_rate_check",
        _rate_validate,
        hit_rate,
        {"rate": "number"},
        checks=frozenset({"P6", "M3"}),
    ),
    StageHandler(
        "kv_replication_check",
        _replication_validate,
        replication,
        {"mean": "number", "maximum": "integer"},
        checks=frozenset({"P5", "structural_cap"}),
    ),
    StageHandler(
        "kv_window_transitions",
        _transitions_validate,
        transitions,
        {"flips": "integer"},
    ),
    StageHandler(
        "kv_cache_statistics",
        _statistics_validate,
        statistics,
        {"keys": "integer", "evictions": "integer"},
    ),
    StageHandler(
        "kv_retry_misdirected",
        _retry_validate,
        retry_misdirected,
        {"requests": "requests", "retried": "boolean"},
    ),
]


def _phase_observation_validate(params, plan):
    p = _fields(
        params,
        {
            "saturation",
            "recovery_windows",
            "family_keys",
            "leader",
            "ready_rate",
            "phase_snapshots",
            "families",
        },
        {
            "saturation",
            "recovery_windows",
            "family_keys",
            "leader",
            "ready_rate",
            "phase_snapshots",
            "families",
        },
    )
    _keys(p["family_keys"])
    _families(p["families"])
    _engine(p["leader"], plan)
    if not _finite(p["ready_rate"]) or not 0 <= p["ready_rate"] <= 1:
        raise ValueError("invalid recovery-ready observation rate")
    if not isinstance(p["saturation"], list) or not 1 <= len(p["saturation"]) <= 200:
        raise ValueError("saturation observations must be bounded")
    if (
        not isinstance(p["recovery_windows"], list)
        or not 1 <= len(p["recovery_windows"]) <= 100
    ):
        raise ValueError("recovery observation windows must be bounded")
    for window in p["recovery_windows"]:
        if not isinstance(window, list) or not 1 <= len(window) <= 100:
            raise ValueError("recovery observation window is empty or unbounded")
    for ref in p["saturation"] + [
        r for window in p["recovery_windows"] for r in window
    ]:
        plan.reference(ref, "kv_hit")
    if not isinstance(p["phase_snapshots"], dict) or set(p["phase_snapshots"]) != {
        "steer",
        "baseline",
        "saturation",
        "recovery",
    }:
        raise ValueError("four explicit phase snapshots are required")
    for ref in p["phase_snapshots"].values():
        plan.reference(ref, "kv_snapshot")
    return p


def phase_observation(ctx, params, deadline):
    deadline.check()
    saturation = [ctx.resource(ref, "kv_hit") for ref in params["saturation"]]
    windows = [
        [ctx.resource(ref, "kv_hit") for ref in window]
        for window in params["recovery_windows"]
    ]
    all_samples = saturation + [sample for window in windows for sample in window]
    if len({s["record"]["wire_request_id"] for s in all_samples}) != len(all_samples):
        raise ValueError("phase observations cannot repeat request IDs")
    leader = _target(ctx, params["leader"])
    selected = [
        sample for sample in saturation if sample["keys"] == params["family_keys"]
    ]
    if not selected:
        raise ValueError("saturation contains no observations for the declared family")
    spill_share = sum(s["landed"] != leader for s in selected) / len(selected)
    recovery_rates = [sum(s["hit"] for s in window) / len(window) for window in windows]
    ready = next(
        (
            str(i + 1)
            for i, rate in enumerate(recovery_rates)
            if rate >= params["ready_rate"]
        ),
        f">{len(windows)}",
    )
    snapshots = {
        phase: ctx.resource(ref, "kv_snapshot")
        for phase, ref in params["phase_snapshots"].items()
    }
    if leader not in snapshots["steer"]["engines"]:
        raise ValueError("spill leader is not present in the steering snapshot")
    digests = {}
    for phase, snapshot in snapshots.items():
        digests[phase] = {}
        for name, engine in snapshot["engines"].items():
            cached = set(engine["cache_key_set"])
            runs = []
            for family in params["families"]:
                run = 0
                for key in family:
                    if key not in cached:
                        break
                    run += 1
                runs.append(run)
            digests[phase][name] = runs
    evidence = dict(
        saturation=saturation,
        recovery_windows=windows,
        recovery_rates=recovery_rates,
        spill_share=spill_share,
        recovery_window=ready,
        phase_snapshots=snapshots,
        contiguous_digests=digests,
    )
    return StageOutput(
        {"spill_share": spill_share, "recovery_window": ready},
        artifacts=[_artifact(ctx, "kv-phase-observations", evidence)],
    )


HANDLERS.append(
    StageHandler(
        "kv_phase_observation",
        _phase_observation_validate,
        phase_observation,
        {"spill_share": "number", "recovery_window": "string"},
    )
)


def _completeness_validate(params, plan):
    p = _fields(params, {"samples", "min_samples"}, {"samples", "min_samples"})
    if not isinstance(p["samples"], list) or not 1 <= len(p["samples"]) <= 200:
        raise ValueError("completion check requires bounded explicit observations")
    for ref in p["samples"]:
        plan.reference(ref, "kv_hit")
    if type(p["min_samples"]) is not int or not 1 <= p["min_samples"] <= 200:
        raise ValueError("completion check requires a positive sample floor")
    return p


def hit_completeness(ctx, params, deadline):
    deadline.check()
    samples = [ctx.resource(ref, "kv_hit") for ref in params["samples"]]
    if len({s["record"]["wire_request_id"] for s in samples}) != len(samples):
        raise ValueError("completion check cannot count repeated requests")
    passed = len(samples) >= params["min_samples"] and all(
        s["success"] for s in samples
    )
    evidence = dict(
        complete=True,
        sample_count=len(samples),
        min_samples=params["min_samples"],
        samples=samples,
    )
    return StageOutput(
        {"complete": passed},
        [
            CheckResult(
                "P6",
                "PASS" if passed else "FAIL",
                actual=passed,
                expected=True,
                evidence=evidence,
            )
        ],
        [_artifact(ctx, "kv-all-request-completion", evidence)],
    )


HANDLERS.append(
    StageHandler(
        "kv_hit_completeness",
        _completeness_validate,
        hit_completeness,
        {"complete": "boolean"},
        checks=frozenset({"P6"}),
    )
)
