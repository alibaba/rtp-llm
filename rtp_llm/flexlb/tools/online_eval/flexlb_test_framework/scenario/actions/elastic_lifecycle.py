"""Finite measurement and traffic stages for the ordered elastic lifecycle."""

import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..contracts import CheckResult, StageHandler, StageOutput


def _params(params, plan, fields, required=()):
    from .elastic import _validate

    return _validate(params, plan, fields, required)


def _name(value, plan):
    if isinstance(value, dict):
        plan.reference(value, "string")
    elif not isinstance(value, str) or not value:
        raise ValueError("engine name must be a string or typed reference")


def timeline_validate(params, plan):
    p = _params(params, plan, {"engines", "offsets_s"}, {"engines", "offsets_s"})
    if not isinstance(p["engines"], list) or not 1 <= len(p["engines"]) <= 32:
        raise ValueError("timeline requires 1..32 engines")
    for value in p["engines"]:
        _name(value, plan)
    offsets = p["offsets_s"]
    if not isinstance(offsets, list) or not 1 <= len(offsets) <= 16 or offsets[0] != 0:
        raise ValueError("timeline requires up to 16 offsets starting at zero")
    if any(type(x) not in (int, float) or not 0 <= x <= 60 for x in offsets) or any(
        a >= b for a, b in zip(offsets, offsets[1:])
    ):
        raise ValueError("timeline offsets must be finite and strictly increasing")
    return p


def _master_get(ctx, endpoint, deadline):
    deadline.check()
    with urllib.request.urlopen(
        f"http://127.0.0.1:{ctx.ops.master_http_port}/{endpoint}",
        timeout=min(5, deadline.remaining()),
    ) as response:
        if response.status != 200:
            raise ValueError(f"master returned HTTP {response.status}")
        return json.load(response)


def timeline(ctx, params, deadline):
    from .elastic import _snapshot

    names = [ctx.resolve(value) for value in params["engines"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate resolved timeline engine")
    start = ctx.clock()
    evidence = dict(
        started_s=start, engines=names, samples=[], health=[], complete=False
    )
    path = ctx.artifact_dir / f"elastic-timeline-{time.time_ns()}.json"
    try:
        _master_get(ctx, "rtp_llm/inflight_status", deadline)
        evidence["health"].append(dict(time_s=ctx.clock(), status=200))
        for offset in params["offsets_s"]:
            while ctx.clock() < start + offset:
                _master_get(ctx, "rtp_llm/inflight_status", deadline)
                evidence["health"].append(dict(time_s=ctx.clock(), status=200))
                remaining = start + offset - ctx.clock()
                if remaining > 0:
                    deadline.sleep(min(1, remaining))
            snap = _snapshot(ctx, deadline)
            counts = {}
            for name in names:
                value = snap.get(name, {}).get("accepted")
                if type(value) is not int or value < 0:
                    raise ValueError(f"missing accepted counter for {name}")
                if (
                    evidence["samples"]
                    and value < evidence["samples"][-1]["counts"][name]
                ):
                    raise ValueError(f"accepted counter reset for {name}")
                counts[name] = value
            evidence["samples"].append(
                dict(offset_s=offset, time_s=ctx.clock(), counts=counts)
            )
        evidence["complete"] = True
        full = ctx.register_resource("snapshot", evidence, historical=True)
        first = ctx.register_resource(
            "snapshot", evidence["samples"][0], historical=True
        )
        last = ctx.register_resource(
            "snapshot", evidence["samples"][-1], historical=True
        )
        return StageOutput(
            output=dict(series=full, first=first, last=last),
            checks=[
                CheckResult(
                    "master_http",
                    "PASS",
                    actual=len(evidence["health"]),
                    evidence=evidence,
                )
            ],
            artifacts=[str(path)],
        )
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["ended_s"] = ctx.clock()
        path.write_text(json.dumps(evidence, indent=2))


def share_validate(params, plan):
    p = _params(
        params,
        plan,
        {
            "before",
            "after",
            "series",
            "engine",
            "max_share",
            "old_floor",
            "exclusive",
            "require_new",
        },
        {"engine", "max_share", "old_floor", "exclusive", "require_new"},
    )
    _name(p["engine"], plan)
    if "series" in p:
        if "before" in p or "after" in p:
            raise ValueError("share uses series or before/after")
        plan.reference(p["series"], "snapshot")
    else:
        for key in ("before", "after"):
            plan.reference(p[key], "snapshot")
    for key in ("max_share", "old_floor"):
        if type(p[key]) not in (int, float) or not 0 <= p[key] <= 1:
            raise ValueError("share bounds must be finite fractions")
    if any(type(p[key]) is not bool for key in ("exclusive", "require_new")):
        raise ValueError("share comparison flags must be boolean")
    return p


def share(ctx, params, deadline):
    deadline.check()
    name = ctx.resolve(params["engine"])
    observations = {}
    if "series" in params:
        series = ctx.resource(params["series"], "snapshot")
        samples = series["samples"]
        if [s["offset_s"] for s in samples] != [0, 5, 10, 17, 24, 31, 38, 45]:
            raise ValueError("preference requires the original 45s measurement offsets")
        before, after = samples[2], samples[-1]

        def portion(a, b):
            counts = {n: b["counts"][n] - a["counts"][n] for n in a["counts"]}
            total = sum(counts.values())
            return counts[name] / total if total > 0 else None

        transient = [portion(samples[i], samples[i + 1]) for i in (0, 1)]
        steady = [portion(samples[i], samples[i + 1]) for i in range(2, 7)]
        observations = dict(
            transient_share=portion(samples[0], samples[2]),
            transient_peak=max(transient) if None not in transient else None,
            steady_subwindow_shares=steady,
            steady_swing=max(steady) - min(steady) if None not in steady else None,
            gates=False,
        )
    else:
        before, after = (
            ctx.resource(params[k], "snapshot") for k in ("before", "after")
        )
    if set(before["counts"]) != set(after["counts"]) or name not in before["counts"]:
        raise ValueError("share endpoints changed within measurement window")
    counts = {n: after["counts"][n] - before["counts"][n] for n in before["counts"]}
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("invalid accepted counter delta")
    total = sum(counts.values())
    shares = {n: v / total if total else 0 for n, v in counts.items()}
    ceiling = (
        shares[name] < params["max_share"]
        if params["exclusive"]
        else shares[name] <= params["max_share"]
    )
    newcomer = total > 0 and ceiling and (not params["require_new"] or counts[name] > 0)
    old = total > 0 and all(
        v >= params["old_floor"] for n, v in shares.items() if n != name
    )
    evidence = dict(
        counts=counts,
        shares=shares,
        denominator=total,
        observations=observations,
        before=before,
        after=after,
    )
    path = ctx.artifact_dir / f"elastic-shares-{time.time_ns()}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult("nonempty", "PASS" if total else "FAIL", actual=total),
            CheckResult(
                "new_share",
                "PASS" if newcomer else "FAIL",
                actual=shares[name],
                expected=params["max_share"],
                evidence=evidence,
            ),
            CheckResult(
                "old_floor",
                "PASS" if old else "FAIL",
                actual={n: v for n, v in shares.items() if n != name},
                expected=params["old_floor"],
                evidence=evidence,
            ),
        ],
        artifacts=[str(path)],
    )


def window_received_validate(params, plan):
    p = _params(
        params, plan, {"before", "after", "engine"}, {"before", "after", "engine"}
    )
    _name(p["engine"], plan)
    for key in ("before", "after"):
        plan.reference(p[key], "snapshot")
    return p


def window_received(ctx, params, deadline):
    deadline.check()
    name = ctx.resolve(params["engine"])
    before, after = (ctx.resource(params[k], "snapshot") for k in ("before", "after"))
    values = [s.get("counts", {}).get(name) for s in (before, after)]
    if any(type(v) is not int or v < 0 for v in values) or values[1] < values[0]:
        raise ValueError("invalid newcomer window accepted counters")
    delta = values[1] - values[0]
    return StageOutput(
        checks=[
            CheckResult(
                "received",
                "PASS" if delta > 0 else "FAIL",
                actual=delta,
                expected=">0",
                evidence=dict(engine=name, before=before, after=after),
            )
        ]
    )


def batch_validate(params, plan):
    p = _params(params, plan, {"count"}, {"count"})
    if type(p["count"]) is not int or p["count"] != 50:
        raise ValueError("rebalance batch preserves exactly 50 requests")
    return p


def batch(ctx, params, deadline):
    return bounded_batch(ctx, params["count"], deadline)


def bounded_batch(ctx, count, deadline, min_success_rate=1.0, expected_method=None):
    from .elastic import RecordedRequests, completeness

    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    done, stop = threading.Event(), threading.Event()
    error = []
    end = ctx.clock() + deadline.remaining()
    path = ctx.artifact_dir / f"elastic-batch-{time.time_ns()}.json"

    def run(record):
        if stop.is_set():
            records.update(
                record,
                cancel=dict(requested_s=ctx.clock(), reason="cleanup_before_rpc"),
                consumer_exit_s=ctx.clock(),
                transport_terminal_s=ctx.clock(),
            )
            return
        rid = record["wire_request_id"]
        records.run(
            record,
            dict(
                input_len=2048,
                output_len=2,
                block_keys=[rid * 100 + j for j in range(3)],
            ),
            timeout_s=min(45, end - ctx.clock()),
            stream_timeout_s=15,
        )

    def pump():
        try:
            issued = [
                records.issue(ctx.ops.next_request_id(), ctx.clock)
                for _ in range(count)
            ]
            with ThreadPoolExecutor(max_workers=min(10, count)) as pool:
                list(pool.map(run, issued))
        except BaseException as exc:
            error.append(f"{type(exc).__name__}: {exc}")
        finally:
            done.set()

    def finish(d, cancel=True):
        if cancel:
            stop.set()
            records.cancel_active()
        try:
            if not done.wait(max(0, d.remaining())):
                stop.set()
                records.cancel_active("batch_deadline")
                raise TimeoutError("batch consumers did not finish")
            if error:
                raise RuntimeError(error[0])
            if any(
                r["consumer_exit_s"] is None or r["transport_terminal_s"] is None
                for r in records.snapshot_records()
            ):
                raise RuntimeError("batch completion lacks final records")
        finally:
            path.write_text(json.dumps(records.snapshot_records(), indent=2))

    ctx.register_resource("requests", records, cleanup=finish)
    thread = threading.Thread(target=pump, name="elastic-bounded-batch", daemon=True)
    try:
        thread.start()
    except BaseException as exc:
        error.append(f"thread start failed: {exc}")
        done.set()
        raise
    finish(deadline, cancel=False)
    result = completeness(records.snapshot_records())
    handle = ctx.register_resource("snapshot", result, historical=True)
    output = StageOutput(
        output=dict(result=handle),
        checks=[
            CheckResult(
                "complete",
                (
                    "PASS"
                    if result["result_complete"] and result["issued"] == count
                    else "FAIL"
                ),
                evidence=result,
            ),
            CheckResult(
                "no_errors" if min_success_rate == 1.0 else "success_rate",
                "PASS" if result["completed"] / count >= min_success_rate else "FAIL",
                actual=result["completed"] / count,
                expected=min_success_rate,
                evidence=result,
            ),
        ],
        artifacts=[str(path)],
    )
    if expected_method is not None:
        admitted = [
            r
            for r in records.snapshot_records()
            if r["schedule"]["status"] == "OK" and r["prefill_addr"]
        ]
        methods = [r["stream"]["method"] for r in admitted]
        output.checks.append(
            CheckResult(
                "protocol",
                (
                    "PASS"
                    if methods and all(m == expected_method for m in methods)
                    else "FAIL"
                ),
                actual=methods,
                expected=expected_method,
            )
        )
    return output


def pause_validate(params, plan):
    p = _params(params, plan, {"seconds"}, {"seconds"})
    if type(p["seconds"]) not in (int, float) or not 0 < p["seconds"] <= 3:
        raise ValueError("pause must be in (0,3]")
    return p


def pause(ctx, params, deadline):
    deadline.sleep(params["seconds"])
    return StageOutput()


def accounting_validate(params, plan):
    return _params(params, plan, set())


def accounting(ctx, params, deadline):
    return accounting_window(ctx, deadline, 95)


def accounting_window(ctx, deadline, budget_s, legacy_prefill_defaults=False):
    start = ctx.clock()
    evidence = dict(started_s=start, samples=[], budget_s=budget_s)
    path = ctx.artifact_dir / f"elastic-accounting-{time.time_ns()}.json"

    class ProbeDeadline:
        def check(self):
            self.remaining()

        def remaining(self):
            outer_remaining = deadline.remaining()
            local_remaining = start + budget_s - ctx.clock()
            remaining = min(outer_remaining, local_remaining)
            if remaining <= 0:
                raise TimeoutError("accounting observation budget expired")
            return remaining

    try:
        while True:
            if evidence["samples"] and ctx.clock() - start >= budget_s:
                break
            data = _master_get(ctx, "rtp_llm/inflight_status", ProbeDeadline())
            evidence["samples"].append(dict(time_s=ctx.clock(), data=data))
            if (
                not isinstance(data, dict)
                or type(data.get("scheduler_inflight")) is not int
            ):
                raise ValueError("scheduler inflight evidence missing")
            sched = data["scheduler_inflight"]
            prefill, decode = data.get("prefill_endpoints"), data.get(
                "decode_endpoints"
            )
            if (
                not isinstance(prefill, list)
                or not prefill
                or not isinstance(decode, list)
                or not decode
            ):
                raise ValueError("endpoint accounting lists missing or empty")
            pvalues = [
                (
                    row.get("inflight_batches", 0)
                    if legacy_prefill_defaults
                    else row.get("inflight_batches")
                )
                for row in prefill
            ]
            dvalues = []
            for row in decode:
                values = [
                    row[k] for k in ("inflight_requests", "total_load") if k in row
                ]
                if not values:
                    raise ValueError(
                        "decode inflight_requests/total_load evidence missing"
                    )
                dvalues.extend(values)
            if any(type(v) is not int or v < 0 for v in [sched, *pvalues, *dvalues]):
                raise ValueError("invalid owner-specific inflight counters")
            local_remaining = start + budget_s - ctx.clock()
            within_budget = local_remaining >= 0
            clean = dict(
                scheduler=sched == 0 and within_budget,
                prefill_batches=all(v == 0 for v in pvalues) and within_budget,
                decode_load=all(v == 0 for v in dvalues) and within_budget,
            )
            if all(clean.values()) or local_remaining <= 0:
                break
            deadline.sleep(min(0.5, local_remaining))
        return StageOutput(
            checks=[
                CheckResult(k, "PASS" if v else "FAIL", evidence=evidence)
                for k, v in clean.items()
            ],
            artifacts=[str(path)],
        )
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["ended_s"] = ctx.clock()
        path.write_text(json.dumps(evidence, indent=2))


def accepted_timed_validate(params, plan):
    from .elastic import _accepted_validate

    return _accepted_validate(params, plan)


def accepted_timed(ctx, params, deadline):
    from .elastic import _accepted_wait

    result = _accepted_wait(ctx, params, deadline)
    result.output["observed_s"] = ctx.clock()
    return result


def add_availability_validate(params, plan):
    p = _params(
        params,
        plan,
        {"flow", "mutation", "received_s"},
        {"flow", "mutation", "received_s"},
    )
    for key, kind in (
        ("flow", "flow"),
        ("mutation", "snapshot"),
        ("received_s", "number"),
    ):
        plan.reference(p[key], kind)
    return p


def add_availability(ctx, params, deadline):
    from .elastic import completeness

    deadline.check()
    flow = ctx.resource(params["flow"], "flow")
    mutation = ctx.resource(params["mutation"], "snapshot")
    start, end = mutation["started_s"] - 1, ctx.resolve(params["received_s"]) + 1
    if ctx.clock() < end:
        raise ValueError("add availability cohort has not reached its closing time")
    cohort = flow.snapshot_cohort(start, end, basis="issued")
    result = completeness(cohort["records"])
    rate = result["completed"] / result["issued"] if result["issued"] else 0
    evidence = dict(cohort=cohort, result=result, window=[start, end])
    path = ctx.artifact_dir / f"elastic-add-availability-{time.time_ns()}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult(
                "nonempty",
                "PASS" if result["issued"] else "FAIL",
                actual=result["issued"],
            ),
            CheckResult(
                "complete",
                "PASS" if result["result_complete"] else "FAIL",
                evidence=evidence,
            ),
            CheckResult(
                "success_rate",
                "PASS" if result["issued"] and rate >= 0.9 else "FAIL",
                actual=rate,
                expected=0.9,
                evidence=evidence,
            ),
        ],
        artifacts=[str(path)],
    )


HANDLERS = [
    StageHandler(
        "elastic_window_received",
        window_received_validate,
        window_received,
        {},
        checks=frozenset({"received"}),
    ),
    StageHandler(
        "elastic_accepted_timed",
        accepted_timed_validate,
        accepted_timed,
        {"accepted": "integer", "observed_s": "number"},
        checks=frozenset({"received"}),
    ),
    StageHandler(
        "elastic_add_availability",
        add_availability_validate,
        add_availability,
        {},
        checks=frozenset({"nonempty", "complete", "success_rate"}),
    ),
    StageHandler(
        "elastic_timeline",
        timeline_validate,
        timeline,
        {"series": "snapshot", "first": "snapshot", "last": "snapshot"},
        checks=frozenset({"master_http"}),
    ),
    StageHandler(
        "elastic_share",
        share_validate,
        share,
        {},
        checks=frozenset({"nonempty", "new_share", "old_floor"}),
    ),
    StageHandler(
        "elastic_batch",
        batch_validate,
        batch,
        {"result": "snapshot"},
        checks=frozenset({"complete", "no_errors"}),
    ),
    StageHandler("elastic_pause", pause_validate, pause, {}),
    StageHandler(
        "elastic_accounting",
        accounting_validate,
        accounting,
        {},
        checks=frozenset({"scheduler", "prefill_batches", "decode_load"}),
    ),
]

from .elastic_rebalance import HANDLERS as REBALANCE_HANDLERS

HANDLERS += REBALANCE_HANDLERS

from .elastic_combined import HANDLERS as COMBINED_HANDLERS

HANDLERS += COMBINED_HANDLERS
