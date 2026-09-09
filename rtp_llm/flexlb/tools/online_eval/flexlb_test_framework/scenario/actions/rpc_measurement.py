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
    observation = record.get("rpc_observation", {})
    if observation.get("schedule_rejected") is True:
        if (
            record["schedule"]["status"] != "REJECTED"
            or observation.get("legacy_success") is not False
            or observation.get("complete") is not True
            or observation.get("mode") != metric
        ):
            raise ValueError("invalid rejected Schedule observation")
        start, end = observation.get("operation_started_s"), observation.get(
            "operation_ended_s"
        )
        if (
            any(
                type(x) not in (int, float) or not math.isfinite(x)
                for x in (start, end)
            )
            or end < start
        ):
            raise ValueError("rejected Schedule lacks caller operation timestamps")
        return record, -1.0 if metric == "stream_ttft" else end - start
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
    observation = record.get("rpc_observation")
    if not isinstance(observation, dict) or observation.get("complete") is not True:
        raise ValueError("latency cohort lacks caller observation evidence")
    if (
        observation.get("mode") != metric
        or type(observation.get("legacy_success")) is not bool
    ):
        raise ValueError("latency cohort has incompatible observation mode")
    if metric == "stream_ttft":
        if observation.get("got_first") is True:
            raw_first = record["stream"]["first_output_s"]
            if type(raw_first) not in (int, float) or not math.isfinite(raw_first):
                raise ValueError("first-output observation lacks raw arrival evidence")
        if observation.get("got_first") is not True:
            return record, -1.0
        start, end = observation.get("observer_started_s"), observation.get(
            "first_observed_s"
        )
    else:
        start, end = observation.get("operation_started_s"), observation.get(
            "operation_ended_s"
        )
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end))
        or end < start
    ):
        raise ValueError(
            "latency cohort has missing or inverted caller observation timestamps"
        )
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
            "PASS" if record["rpc_observation"]["legacy_success"] else "FAIL",
            actual=record["rpc_observation"]["legacy_success"],
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


def validate_latency_sample(params, plan):
    if not isinstance(params, dict) or set(params) != {
        "mode",
        "input_len",
        "output_len",
    }:
        raise ValueError("latency sample requires mode and explicit request shape")
    if params["mode"] not in ("stream_ttft", "request_total"):
        raise ValueError("invalid latency sample mode")
    for key in ("input_len", "output_len"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError("latency sample shape must be positive integers")
    return copy.deepcopy(params)


def latency_sample(ctx, params, deadline):
    from ..observed import ObservedRequestBatch

    batch = ObservedRequestBatch(
        ctx,
        dict(
            count=1,
            consume="immediate",
            input_len=params["input_len"],
            output_len=params["output_len"],
            schedule_timeout_s=30,
            stream_timeout_s=60,
        ),
        params["mode"],
        deadline,
    )
    handle = ctx.register_resource("requests", batch, batch.cleanup, historical=True)
    batch.submit(deadline)
    observation = batch.snapshot_records()[0]["rpc_observation"]
    return StageOutput(
        {"requests": handle, "legacy_success": observation["legacy_success"]}
    )


HANDLERS = [
    StageHandler(
        "rpc_latency_sample",
        validate_latency_sample,
        latency_sample,
        {"requests": "requests", "legacy_success": "boolean"},
    ),
    StageHandler(
        "rpc_latency_check",
        validate,
        execute,
        {"delta_s": "number", "recovery_delta_s": "number"},
        checks=CHECKS,
    ),
]


def validate_probe(params, plan):
    allowed = {
        "input_len",
        "output_len",
        "schedule_timeout_s",
        "stream_timeout_s",
        "expected_rpc_statuses",
        "require_error_detail",
    }
    if not isinstance(params, dict) or set(params) != allowed:
        raise ValueError(
            "rpc_fault_probe requires explicit shape, budgets and expected status policy"
        )
    for key in ("input_len", "output_len"):
        if type(params[key]) is not int or not 1 <= params[key] <= 1048576:
            raise ValueError("invalid finite probe shape")
    for key in ("schedule_timeout_s", "stream_timeout_s"):
        value = params[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= 60
        ):
            raise ValueError("invalid finite probe RPC budget")
    statuses = params["expected_rpc_statuses"]
    if (
        not isinstance(statuses, list)
        or any(
            not isinstance(s, str)
            or s not in {"UNKNOWN", "INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"}
            for s in statuses
        )
        or len(statuses) != len(set(statuses))
    ):
        raise ValueError("invalid explicit request-level RPC status policy")
    if type(params["require_error_detail"]) is not bool:
        raise ValueError("require_error_detail must be boolean")
    return copy.deepcopy(params)


def probe(ctx, params, deadline):
    from ..backend import RequestBatch

    batch_params = {
        key: params[key]
        for key in ("input_len", "output_len", "schedule_timeout_s", "stream_timeout_s")
    }
    batch = RequestBatch(ctx, dict(batch_params, count=1, consume="immediate"))
    handle = ctx.register_resource("requests", batch, batch.cleanup)
    caught = None
    caught_exception = None
    try:
        batch.submit(deadline)
        batch.wait(deadline)
    except Exception as exc:
        caught = repr(exc)
        caught_exception = exc
    # The stage's own deadline is never converted into an expected request fault.
    deadline.check()
    rows = batch.snapshot_records()
    if len(rows) != 1:
        raise RuntimeError("probe did not produce its one accounted request")
    row = rows[0]
    if row["consumer_exit_s"] is None or row["transport_terminal_s"] is None:
        raise RuntimeError("probe has incomplete terminal evidence")
    if row["stream"]["started_s"] is not None and (
        row.get("consumer_completion_verified") is not True
        or row.get("consumer_done") is not True
    ):
        raise RuntimeError("probe consumer has not proved its exit")
    expected = set(params["expected_rpc_statuses"])
    observed_rpc_fault = False
    for phase in ("schedule", "stream"):
        rpc = row[phase]
        if (
            phase == "stream"
            and rpc["started_s"] is None
            and row["schedule"]["status"] != "OK"
        ):
            continue
        if rpc["ended_s"] is None or rpc["status"] not in {"OK", "REJECTED", *expected}:
            raise RuntimeError(f"unexpected or unavailable {phase} evidence: {rpc}")
        observed_rpc_fault |= rpc["status"] in expected
    if caught is not None and not observed_rpc_fault:
        raise caught_exception
    observed = (
        observed_rpc_fault
        or row["schedule"]["status"] == "REJECTED"
        or row["business_error_code"] not in (None, 0)
        or (params["require_error_detail"] and not row["business_finished"])
    )
    evidence = dict(
        complete=True,
        sample_count=1,
        min_samples=1,
        record=row,
        request_status_policy=params["expected_rpc_statuses"],
        execution_exception=caught,
    )
    path = ctx.artifact_dir / f"rpc-fault-probe-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        {"requests": handle, "fault_observed": observed},
        [
            CheckResult(
                "fault_observed",
                "PASS" if observed else "FAIL",
                actual=observed,
                expected=True,
                evidence=evidence,
            )
        ],
        [str(path)],
    )


def validate_probe_ref(params, plan):
    if not isinstance(params, dict) or set(params) != {"requests"}:
        raise ValueError("expected only a probe request handle")
    plan.reference(params["requests"], "requests")
    return copy.deepcopy(params)


def probe_cancel(ctx, params, deadline):
    from ..backend import RequestBatch

    batch = ctx.resource(params["requests"], "requests")
    if not isinstance(batch, RequestBatch) or len(batch.entries) != 1:
        raise ValueError("probe cancel requires its one-request backend cohort")
    response = batch.entries[0]["response"]
    issued = (
        batch.cancel_server(deadline)
        if response is not None and response.success
        else 0
    )
    return StageOutput({"issued": issued})


def _owner_count(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("missing or invalid Master owner count")
    return value


def _owner_counts(data):
    scheduler = _owner_count(data.get("scheduler_inflight"))
    prefill, decode = data.get("prefill_endpoints"), data.get("decode_endpoints")
    if (
        not isinstance(prefill, list)
        or not prefill
        or not isinstance(decode, list)
        or not decode
    ):
        raise ValueError("missing Master endpoint owner evidence")
    p_counts = [_owner_count(row.get("inflight_batches")) for row in prefill]
    d_counts = []
    for row in decode:
        fields = [key for key in ("inflight_requests", "total_load") if key in row]
        if not fields:
            raise ValueError("missing Decode owner load evidence")
        for key in fields:
            _owner_count(row[key])
        # The legacy helper uses OR: zero requests cannot hide positive load.
        d_counts.append(row.get("inflight_requests", 0) or row.get("total_load", 0))
    return dict(scheduler=scheduler, prefill=p_counts, decode=d_counts)


def _owners_clean(ctx, deadline, applicable):
    from .master import _master_json

    samples = []
    evidence = dict(
        applicable=applicable,
        samples=samples,
        owners=[
            "FlexLB scheduler",
            "Prefill batch ledgers",
            "Decode request/load ledgers",
        ],
        complete=False,
        reason=(
            "BATCH owner assertion"
            if applicable
            else "legacy condition excludes NON_BATCH or unsuccessful Schedule"
        ),
    )
    path = ctx.artifact_dir / f"rpc-fault-owner-{uuid.uuid4().hex}.json"
    try:
        if applicable:
            while True:
                deadline.check()
                raw = _master_json(ctx, "single", "/rtp_llm/inflight_status", deadline)
                sample = dict(time_s=ctx.clock(), raw=raw)
                samples.append(sample)
                counts = _owner_counts(raw)
                sample["owners"] = counts
                if counts["scheduler"] == 0 and not any(
                    counts["prefill"] + counts["decode"]
                ):
                    break
                deadline.sleep(0.5)
        evidence.update(
            complete=True, sample_count=len(samples), min_samples=1 if applicable else 0
        )
        return StageOutput(
            {"applicable": applicable},
            [
                CheckResult(
                    "all_owners_zero",
                    "PASS",
                    actual=samples[-1]["owners"] if samples else "not applicable",
                    expected="all owners zero" if applicable else "not applicable",
                    evidence=evidence,
                )
            ],
            [str(path)],
        )
    finally:
        path.write_text(json.dumps(evidence, indent=2))


def validate_owner_clean(params, plan):
    if params != {}:
        raise ValueError("RPC owner cleanup accepts no parameters")
    return {}


def owner_clean(ctx, params, deadline):
    return _owners_clean(ctx, deadline, True)


def probe_owner_clean(ctx, params, deadline):
    from ..backend import RequestBatch

    batch = ctx.resource(params["requests"], "requests")
    if not isinstance(batch, RequestBatch) or len(batch.entries) != 1:
        raise ValueError("probe owner check requires its one-request backend cohort")
    response = batch.entries[0]["response"]
    applicable = (
        response is not None and response.success and response.enqueued_by_master
    )
    return _owners_clean(ctx, deadline, applicable)


HANDLERS += [
    StageHandler(
        "rpc_owner_clean",
        validate_owner_clean,
        owner_clean,
        {"applicable": "boolean"},
        checks=frozenset({"all_owners_zero"}),
        requires=frozenset({"enqueue_batch"}),
    ),
    StageHandler(
        "rpc_fault_probe",
        validate_probe,
        probe,
        {"requests": "requests", "fault_observed": "boolean"},
        checks=frozenset({"fault_observed"}),
    ),
    StageHandler(
        "rpc_probe_cancel", validate_probe_ref, probe_cancel, {"issued": "integer"}
    ),
    StageHandler(
        "rpc_probe_owner_clean",
        validate_probe_ref,
        probe_owner_clean,
        {"applicable": "boolean"},
        checks=frozenset({"all_owners_zero"}),
    ),
]
