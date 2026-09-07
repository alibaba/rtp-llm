from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...harness import AssertUtils
from ...registry import case
from ...support.admission import (
    STREAM_TIMEOUT_S,
    _capacity_spec,
    _master_http,
    _prefill_names,
)


@case(
    "admission_master_capacity_reject",
    category="admission",
    profiles=["batch-window"],
    source=(
        "gap G11: master outstanding-capacity admission — typed QUEUE_FULL "
        "(8502 TooManyRequests) fast reject.  F6 verdict overturned: the "
        "reject IS typed (dedicated code + status_name + detail); the legacy "
        "assertion family ('outstanding'/'exhaust'/'resource'/'8431') "
        "matched zero tokens of the actual response payload."
    ),
)
def admission_master_capacity(ctx: CaseContext):
    """Master-side unified admission: with
    capacity.maxOutstandingRequestsGlobal=2 and PRIORITY ordering, the
    submit path (RequestRegistry.register -> tryAcquireOutstandingPermit)
    fast-rejects every request beyond the global budget with the typed
    QUEUE_FULL error — code 8502, error_message JSON
    {"status_name":"TooManyRequests","detail":"QUEUE_FULL"} — a
    synchronous, typed rejection, no queueing and no leak.  Once the
    in-flight occupants terminate, a sequential request must succeed
    again.

    F6 note (admission wave-2): the old docstring claimed
    RESOURCE_EXHAUSTED "master outstanding capacity exhausted" — that
    string family never matched the real payload (the 8431
    RESOURCE_EXHAUSTED path is the acceptance-limit / eviction gate, not
    the outstanding permit).  The master behaviour was always typed; the
    defect was the test's assertion family.

    Profile semantics (v2): the outstanding-capacity permit is
    taken on the master submit path for every delivery mode, but
    _capacity_spec pins the legacy fault axes (PRIORITY + FIXED_WINDOW +
    BATCH) via FLEXLB_CONFIG — re-running under another --profile would
    execute the identical configuration, so the declaration stays
    batch-window (label honesty + regression efficiency).
    """
    env = ctx.env_manager.ensure(_capacity_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        # Slow prefills keep the two admitted occupants inside the budget.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=4000.0)

        def run(rid: int):
            # Direct Schedule so the typed reject is asserted on the raw
            # response (code + error_message), not on run_one_request's
            # flattened text (which drops the code).
            t0 = time.monotonic()
            resp = ops.schedule(rid, input_len=512, output_len=2)
            if resp.code != 200 or not resp.success:
                return (
                    "reject",
                    resp.code,
                    str(resp.error_message),
                    (time.monotonic() - t0),
                )
            # batch-window profile: BATCH dispatch -> FetchResponse stream.
            handle = ops.start_stream(resp, rid)
            handle.wait_end(15.0)
            snap = handle.snap
            if snap.error or not snap.completed:
                return (
                    "serve_err",
                    resp.code,
                    (snap.error or "stream did not complete"),
                    (time.monotonic() - t0),
                )
            return "served", resp.code, None, time.monotonic() - t0

        rids = [ops.next_request_id(base) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, rids))
        rejected = [
            (code, msg, t) for kind, code, msg, t in results if kind == "reject"
        ]
        serve_failures = [
            (code, msg) for kind, code, msg, _ in results if kind == "serve_err"
        ]
        served = [t for kind, _, _, t in results if kind == "served"]
        reject_types = sorted({f"{code}:{msg[:60]}" for code, msg, _ in rejected})
        reject_fast = all(t < 3.0 for _, _, t in rejected)
        reject_typed = bool(rejected) and all(
            code == 8502
            and "toomanyrequests" in msg.lower()
            and "queue_full" in msg.lower()
            for code, msg, _ in rejected
        )

        for n in names:
            ops.set_perf(n, prefill_fixed_ms=100.0)
        rid5 = ops.next_request_id(base)
        _, err5 = ops.run_one_request(
            rid5, input_len=512, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )
        # follow-up fix: the recovery verdict used to swallow err5 —
        # a failed recovery had NO visible cause.  Surface the raw error
        # (resp code + message) inside the detail so the failure is
        # diagnosable from the report alone.
        err5_detail = f" err5={str(err5)[:120]!r}" if err5 else ""

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        passed = (
            1 <= len(rejected) <= 2
            and len(served) >= 2
            and not serve_failures
            and reject_fast
            and reject_typed
            and err5 is None
            and inflight_ok
        )
        return passed, (
            f"served={len(served)}, rejected={len(rejected)} "
            f"(fast={reject_fast}, typed_8502_queue_full={reject_typed}, "
            f"types={reject_types[:2]}), "
            f"serve_failures={serve_failures[:1]}, "
            f"sequential_recovery={err5 is None}{err5_detail}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
