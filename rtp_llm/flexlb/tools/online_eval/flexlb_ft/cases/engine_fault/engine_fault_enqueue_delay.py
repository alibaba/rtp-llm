from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.engine_fault import STREAM_TIMEOUT_S, _master_http, _prefill_names


@case(
    "engine_fault_enqueue_delay",
    category="engine_fault",
    requires=["enqueue_batch"],
    source="gap G6/G7: /inject type=enqueue_delay (deferred enqueue ack, BATCH dispatch)",
)
def inject_enqueue_delay(ctx: CaseContext):
    """enqueue_delay defers the whole enqueue runnable (admission + ack) by
    delay_ms, so the BATCH-dispatch schedule() — which waits for the enqueue
    ack — grows by roughly delay_ms.  delay_ms must stay well below
    dispatcher.enqueueRpcTimeoutMs (default 5000) or the RPC deadline
    fires first.

    Assertions: end-to-end latency delta >= 1.2s at delay_ms=1500, the
    request still SUCCEEDS (delay, not failure), and latency recovers once
    the injection is cleared.

    Profile semantics (v2): the deferred runnable is the
    engine's EnqueueBatch processing, which exists only under the BATCH
    dispatcher — requires=["enqueue_batch"] keeps the case to the
    BATCH-dispatch profiles (batch-window, single-batch).  Unlike the
    fault_spec cases this one runs on the shared smoke env
    (real per-profile config), so single-batch exercises the SINGLE
    decision axis on the same enqueue path.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "engine_fault")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        rid0 = ops.next_request_id(base)
        t0 = time.monotonic()
        _, err0 = ops.run_one_request(rid0, stream_timeout_s=STREAM_TIMEOUT_S)
        baseline_total = time.monotonic() - t0
        if err0:
            return False, f"baseline request failed: {err0}"

        inject_type_all(ops, names, "enqueue_delay", delay_ms=1500)
        rid1 = ops.next_request_id(base)
        t1 = time.monotonic()
        _, err1 = ops.run_one_request(rid1, stream_timeout_s=STREAM_TIMEOUT_S)
        delayed_total = time.monotonic() - t1

        clear_type_all(ops, names, "enqueue_delay")
        rid2 = ops.next_request_id(base)
        t2 = time.monotonic()
        _, err2 = ops.run_one_request(rid2, stream_timeout_s=STREAM_TIMEOUT_S)
        recovered_total = time.monotonic() - t2

        delta = delayed_total - baseline_total
        # Integration-round cascade hygiene: residue from
        # earlier cases on this shared env settles via the stale-TTL +
        # ExpirationTimer path (worst ~90s); drain it BEFORE the clean
        # assertion so another case's TTL settle cannot fail this one.
        # Best-effort on purpose — a true leak never drains and the 10s
        # assertion below still catches it.
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )

        passed = (
            err1 is None
            and err2 is None
            and delta >= 1.2
            and recovered_total <= baseline_total + 1.0
            and inflight_ok
        )
        return passed, (
            f"baseline={baseline_total:.2f}s, delayed={delayed_total:.2f}s "
            f"(delta={delta:.2f}s >= 1.2), recovered={recovered_total:.2f}s, "
            f"delayed_ok={err1 is None}, recovered_ok={err2 is None}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_delay")
