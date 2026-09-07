"""Compare explicit request cohorts; fault selection and recovery stay in YAML."""

import copy
import json
import math
import uuid

from ..contracts import CheckResult, StageHandler, StageOutput
from .elastic import request_success

CHECKS = frozenset(
    {
        "baseline_success",
        "delayed_success",
        "recovery_success",
        "latency_increased",
        "latency_recovered",
    }
)


def validate(params, plan):
    allowed = {
        "baseline",
        "delayed",
        "recovery",
        "metric",
        "min_delta_s",
        "recovery_slack_s",
    }
    if not isinstance(params, dict) or set(params) != allowed:
        raise ValueError("rpc_latency_check requires explicit cohorts and thresholds")
    for name in ("baseline", "delayed", "recovery"):
        plan.reference(params[name], "requests")
    if params["metric"] not in ("stream_ttft", "request_total"):
        raise ValueError("unsupported RPC latency measurement")
    for name in ("min_delta_s", "recovery_slack_s"):
        value = params[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("latency thresholds must be finite and nonnegative")
    return copy.deepcopy(params)


def _sample(ctx, reference, metric):
    records = ctx.resource(reference, "requests").snapshot_records()
    if len(records) != 1:
        raise ValueError("latency cohort requires exactly one accounted request")
    record = records[0]
    if (
        record.get("consumer_completion_verified") is not True
        or record.get("consumer_done") is not True
        or record.get("consumer_exit_s") is None
        or record.get("transport_terminal_s") is None
    ):
        raise ValueError("latency cohort lacks consumer terminal evidence")
    for key in ("consumer_exit_s", "transport_terminal_s"):
        value = record[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("latency cohort has invalid terminal timestamps")
    if metric == "stream_ttft":
        start, end = record["stream"]["started_s"], record["stream"]["first_output_s"]
    else:
        start, end = record["issued_s"], record["transport_terminal_s"]
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end))
        or end < start
    ):
        raise ValueError("latency cohort has missing or inverted timestamps")
    return record, end - start


def execute(ctx, params, deadline):
    deadline.check()
    samples = {
        name: _sample(ctx, params[name], params["metric"])
        for name in ("baseline", "delayed", "recovery")
    }
    rids = [record["wire_request_id"] for record, value in samples.values()]
    if len(set(rids)) != 3:
        raise ValueError("baseline, delayed and recovery must be distinct requests")
    baseline, delayed, recovery = (
        samples[name][1] for name in ("baseline", "delayed", "recovery")
    )
    evidence = dict(
        metric=params["metric"],
        samples={
            name: dict(record=record, latency_s=value)
            for name, (record, value) in samples.items()
        },
        complete=True,
        sample_count=3,
        min_samples=3,
        env_epoch=ctx.env_epoch,
    )
    path = ctx.artifact_dir / f"rpc-latency-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2))
    checks = [
        CheckResult(
            name + "_success",
            "PASS" if request_success(record) else "FAIL",
            actual=request_success(record),
            expected=True,
            evidence=evidence,
        )
        for name, (record, value) in samples.items()
    ]
    checks.extend(
        [
            CheckResult(
                "latency_increased",
                "PASS" if delayed - baseline >= params["min_delta_s"] else "FAIL",
                actual=delayed - baseline,
                expected=params["min_delta_s"],
                evidence=evidence,
            ),
            CheckResult(
                "latency_recovered",
                "PASS" if recovery - baseline <= params["recovery_slack_s"] else "FAIL",
                actual=recovery - baseline,
                expected=params["recovery_slack_s"],
                evidence=evidence,
            ),
        ]
    )
    return StageOutput(
        {"delta_s": delayed - baseline, "recovery_delta_s": recovery - baseline},
        checks,
        [str(path)],
    )


HANDLERS = [
    StageHandler(
        "rpc_latency_check",
        validate,
        execute,
        {"delta_s": "number", "recovery_delta_s": "number"},
        checks=CHECKS,
    )
]
