"""Serial traffic probes for the added-worker lifecycle's legacy contract."""

import json
import time

from ..contracts import CheckResult, StageHandler, StageOutput


def validate(params, plan):
    from .elastic import _validate

    p = _validate(
        params, plan, {"engine", "window_s", "method"}, {"engine", "window_s", "method"}
    )
    plan.reference(p["engine"], "string")
    if p["window_s"] not in (15, 20):
        raise ValueError("added-worker probe window must be 15 or 20 seconds")
    if p["method"] not in ("FetchResponse", "GenerateStreamCall"):
        raise ValueError("unsupported added-worker consumer")
    return p


def pump(ctx, params, deadline):
    from .elastic import RecordedRequests, _snapshot, completeness

    name = ctx.resolve(params["engine"])
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    path = ctx.artifact_dir / f"elastic-added-probe-{time.time_ns()}.json"
    evidence = dict(engine=name, requests=[], samples=[])

    def persist():
        evidence["requests"] = records.snapshot_records()
        path.write_text(json.dumps(evidence, indent=2))

    def cleanup(d):
        records.cancel_active("added_probe_cleanup")
        persist()

    ctx.register_resource("requests", records, cleanup=cleanup)

    def accepted():
        value = _snapshot(ctx, deadline).get(name, {}).get("accepted")
        if type(value) is not int or value < 0:
            raise ValueError("missing added-worker accepted counter")
        evidence["samples"].append(dict(time_s=ctx.clock(), accepted=value))
        return value

    try:
        baseline = accepted()
        end = ctx.clock() + params["window_s"]
        evidence.update(baseline=baseline, issuance_end_s=end)
        current = baseline
        while ctx.clock() < end:
            deadline.check()
            rid = ctx.ops.next_request_id()
            record = records.issue(rid, ctx.clock)
            records.run(
                record,
                dict(output_len=2, block_keys=[rid * 100 + 1]),
                timeout_s=min(40, deadline.remaining()),
                schedule_timeout_s=30,
                stream_timeout_s=10,
            )
            current = accepted()
            if current > baseline:
                break
            # Legacy waits after completion, even when the final call crossed
            # its issuance cutoff; it then performs one final counter sample.
            deadline.sleep(0.2)
        else:
            current = accepted()
        rows = records.snapshot_records()
        admitted = [
            r for r in rows if r["schedule"]["status"] == "OK" and r["prefill_addr"]
        ]
        methods = [r["stream"]["method"] for r in admitted]
        complete = completeness(rows)["complete"]
        return StageOutput(
            output=dict(accepted=current),
            checks=[
                CheckResult(
                    "received",
                    "PASS" if current > baseline else "FAIL",
                    actual=current - baseline,
                    expected=">0",
                ),
                CheckResult(
                    "complete", "PASS" if complete else "FAIL", actual=complete
                ),
                CheckResult(
                    "protocol",
                    (
                        "PASS"
                        if methods and all(m == params["method"] for m in methods)
                        else "FAIL"
                    ),
                    actual=methods,
                    expected=params["method"],
                ),
            ],
            artifacts=[str(path)],
        )
    finally:
        persist()


def growth_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"before", "after"}, {"before", "after"})
    for value in p.values():
        plan.reference(value, "integer")
    return p


def growth(ctx, params, deadline):
    deadline.check()
    before, after = ctx.resolve(params["before"]), ctx.resolve(params["after"])
    return StageOutput(
        checks=[
            CheckResult(
                "increased",
                "PASS" if after > before else "FAIL",
                actual=dict(before_stop=before, after_restart=after),
                expected="after > before",
            )
        ]
    )


HANDLERS = [
    StageHandler(
        "elastic_added_growth",
        growth_validate,
        growth,
        {},
        checks=frozenset({"increased"}),
    ),
    StageHandler(
        "elastic_added_probe",
        validate,
        pump,
        {"accepted": "integer"},
        checks=frozenset({"received", "complete", "protocol"}),
    ),
]
