"""Explicit probe cohorts and literal NON_BATCH accounting for lifecycle cases."""

import json
import threading
import time

from ..contracts import CheckResult, StageHandler, StageOutput
from . import elastic_added_worker as added
from . import elastic_lifecycle as life


def method(p):
    if p["method"] not in ("FetchResponse", "GenerateStreamCall"):
        raise ValueError("unsupported lifecycle consumer")
    return p


def probe_validate(params, plan):
    p = life._params(
        params, plan, {"engine", "window_s", "method"}, {"engine", "window_s", "method"}
    )
    life._name(p["engine"], plan)
    if p["window_s"] not in (10, 15):
        raise ValueError("lifecycle probe issuance window must be 10 or 15 seconds")
    return method(p)


def flow_validate(params, plan):
    p = life._params(params, plan, {"flow", "method"}, {"flow", "method"})
    plan.reference(p["flow"], "flow")
    return method(p)


def flow_protocol(ctx, params, deadline):
    deadline.check()
    rows = ctx.resource(params["flow"], "flow").snapshot_records()
    admitted = [
        r for r in rows if r["schedule"]["status"] == "OK" and r["prefill_addr"]
    ]
    methods = [r["stream"]["method"] for r in admitted]
    return StageOutput(
        checks=[
            CheckResult(
                "protocol",
                (
                    "PASS"
                    if methods and all(m == params["method"] for m in methods)
                    else "FAIL"
                ),
                actual=methods,
                expected=params["method"],
            )
        ]
    )


def literal_accounting(ctx, params, deadline):
    # Preserve exactly the old Prefill field/default. Raw route counters are
    # retained in evidence but never inferred to be zero from batch counters.
    return life.accounting_window(ctx, deadline, 95, legacy_prefill_defaults=True)


def recovery_validate(params, plan):
    return method(life._params(params, plan, {"method"}, {"method"}))


def wait_consumer(done, timeout_s):
    return done.wait(timeout_s)


def recovery(ctx, params, deadline):
    from .elastic import RecordedRequests

    ready, done = threading.Event(), threading.Event()

    class RecoveryRecords(RecordedRequests):
        def _activate(self, record, call):
            super()._activate(record, call)
            if record["stream"]["method"] is not None:
                # _activate is called only after Fetch/Generate returned its
                # iterator; stream.started_s is earlier, before that call.
                evidence["open_return_s"] = ctx.clock()
                ready.set()

    records = RecoveryRecords(ctx.ops, ctx.env_epoch, ctx.clock)
    path = ctx.artifact_dir / f"elastic-cycle-recovery-{time.time_ns()}.json"
    evidence = dict(frozen=None, open_return_s=None)

    def persist():
        evidence["final_records"] = records.snapshot_records()
        path.write_text(json.dumps(evidence, indent=2))

    def cleanup(d):
        records.cancel_active("cycle_recovery_cleanup")
        try:
            if not done.wait(max(0, d.remaining())):
                raise TimeoutError("cycle recovery consumer did not exit")
        finally:
            persist()

    ctx.register_resource("requests", records, cleanup=cleanup)
    try:
        rid = ctx.ops.next_request_id()
        row = records.issue(rid, ctx.clock)

        def consume():
            try:
                records.run(
                    row,
                    dict(output_len=2, block_keys=[rid * 100 + 1]),
                    timeout_s=90,
                    schedule_timeout_s=30,
                    stream_timeout_s=60,
                )
            finally:
                done.set()
                ready.set()

        thread = threading.Thread(
            target=consume, name="elastic-cycle-recovery", daemon=True
        )
        try:
            thread.start()
        except BaseException:
            done.set()
            raise
        if not ready.wait(max(0, deadline.remaining())):
            records.cancel_active("cycle_recovery_schedule_timeout")
            raise TimeoutError("cycle recovery Schedule did not settle")
        # The legacy wire RPC timeout is 60s, but verify_recovery waits only
        # 30s for its consumer. Cancellation and cleanup retain final evidence.
        evidence["caller_observation_started_s"] = ctx.clock()
        exited = wait_consumer(done, min(30, max(0, deadline.remaining())))
        frozen = records.snapshot_records()[0]
        # StreamHandle ignores typed error frames and CANCELLED transport errors
        # in snap.error. Keep that literal predicate separate from cleanup.
        legacy_success = frozen["business_finished"] and (
            not frozen["stream"]["error"] or frozen["stream"]["status"] == "CANCELLED"
        )
        evidence["frozen"] = dict(
            time_s=ctx.clock(),
            record=frozen,
            legacy_success=legacy_success,
            consumer_exited=exited,
        )
        if not exited:
            records.cancel_active("cycle_recovery_after_legacy_observation")
        actual = frozen["stream"]["method"]
        return StageOutput(
            checks=[
                CheckResult(
                    "legacy_success",
                    "PASS" if legacy_success else "FAIL",
                    actual=legacy_success,
                    evidence=evidence["frozen"],
                ),
                CheckResult(
                    "protocol",
                    "PASS" if actual == params["method"] else "FAIL",
                    actual=actual,
                    expected=params["method"],
                ),
            ],
            artifacts=[str(path)],
        )
    finally:
        persist()


HANDLERS = [
    StageHandler(
        "elastic_lifecycle_probe",
        probe_validate,
        added.pump,
        {"accepted": "integer"},
        checks=frozenset({"received", "complete", "protocol"}),
    ),
    StageHandler(
        "elastic_lifecycle_flow_protocol",
        flow_validate,
        flow_protocol,
        {},
        checks=frozenset({"protocol"}),
    ),
    StageHandler(
        "elastic_literal_nonbatch_accounting",
        life.accounting_validate,
        literal_accounting,
        {},
        checks=frozenset({"scheduler", "prefill_batches", "decode_load"}),
    ),
    StageHandler(
        "elastic_cycle_recovery",
        recovery_validate,
        recovery,
        {},
        checks=frozenset({"legacy_success", "protocol"}),
    ),
]
