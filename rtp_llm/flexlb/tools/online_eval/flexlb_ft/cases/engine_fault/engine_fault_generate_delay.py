from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.engine_fault import _master_http, _measure_ttft, _prefill_names


@case(
    "engine_fault_generate_delay",
    category="engine_fault",
    source="gap G6/G7: /inject type=generate_delay (prefill execution inflation, all profiles)",
)
def inject_generate_delay(ctx: CaseContext):
    """generate_delay adds delay_ms to the prefill execution estimate
    (runPrefillBatch), so the first-output latency grows by roughly
    delay_ms under EVERY profile (unlike enqueue_delay, schedule() stays
    fast and only TTFT inflates).

    Assertions: TTFT delta >= 1.2s at delay_ms=1500, the request still
    SUCCEEDS, and TTFT recovers after the injection is cleared."""
    ops = ctx.ops()
    base = rid_base(ctx, "engine_fault")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        rid0 = ops.next_request_id(base)
        ttft_base, err0, enq0 = _measure_ttft(ops, rid0)
        if err0:
            return False, f"baseline request failed: {err0}"

        inject_type_all(ops, names, "generate_delay", delay_ms=1500)
        rid1 = ops.next_request_id(base)
        ttft_delayed, err1, _ = _measure_ttft(ops, rid1)

        clear_type_all(ops, names, "generate_delay")
        rid2 = ops.next_request_id(base)
        ttft_recovered, err2, _ = _measure_ttft(ops, rid2)

        delta = ttft_delayed - ttft_base
        if enq0:
            # Integration-round cascade hygiene: drain earlier
            # cases' TTL-settling residue before the clean assertion (see
            # inject_enqueue_delay) — best-effort, the 10s assertion below
            # keeps the real leak detection.
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A (non-batch path)"

        passed = (
            err1 is None
            and err2 is None
            and delta >= 1.2
            and ttft_recovered <= ttft_base + 1.0
            and inflight_ok
        )
        return passed, (
            f"ttft_baseline={ttft_base:.2f}s, ttft_delayed={ttft_delayed:.2f}s "
            f"(delta={delta:.2f}s >= 1.2), ttft_recovered={ttft_recovered:.2f}s, "
            f"delayed_ok={err1 is None}, recovered_ok={err2 is None}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "generate_delay")
