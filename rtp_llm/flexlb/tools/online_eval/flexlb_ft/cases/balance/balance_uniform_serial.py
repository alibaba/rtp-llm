from __future__ import annotations

import json
import time
from collections import Counter

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.balance import STREAM_TIMEOUT_S, _prefill_names


@case(
    "balance_uniform_serial",
    category="balance",
    source="scheduling_smoke.py S1+S6+S8 (merged)",
)
def balance_uniform_serial(ctx: CaseContext):
    """Homogeneous serial traffic spreads evenly across equivalent engines.

    Result properties (graded): P1 request-uniformity max-share + P2
    no-starvation, measured over n=20 serial requests per variant, counted
    from CLIENT landing addresses.

    Two parameterized variants:
      * plain — no injection;
      * speed_hetero — one prefill slowed via set_perf (200ms fixed).  The
        injection is a regression guard, not a routing signal: the prefill
        score is ledgerWaitMs + FORMULA estimate (a pure function of the
        request's token shape) + batcherWaitMs, so engine *speed* never
        enters the score and serial requests leave no backlog — both
        engines stay tied and the tie window is sampled uniformly per
        request.  A skewed split under this variant would mean the score
        model started leaking speed or leaving residue — a real defect the
        P1 property catches at any grade.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    n = 20
    perf_engine = None
    try:
        for variant, slow in (("plain", False), ("speed_hetero", True)):
            if slow:
                prefill_names = _prefill_names(ops)
                if len(prefill_names) >= 2:
                    ops.set_perf(prefill_names[1], prefill_fixed_ms=200.0)
                    perf_engine = prefill_names[1]
                    time.sleep(1.5)  # master perf sync

            addrs = []
            failure = None
            for _ in range(n):
                rid = ops.next_request_id(rid_base(ctx, "balance"))
                keys = [rid * 100 + j for j in range(3)]
                addr, err = ops.run_one_request(
                    rid,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failure = f"{variant}: rid={rid} failed: {err}"
                    break
                addrs.append(addr)
            if failure:
                report.invariant("P6", False, context=variant, detail=failure)
                break

            addr_map = ops.addr_to_name()
            dist = Counter(addr_map.get(a, a) for a in addrs)
            used = len(dist)
            max_share = max(dist.values()) / n
            dist_json = json.dumps(dict(dist), sort_keys=True)
            report.check(
                "P1",
                max_share,
                context=variant,
                detail=f"n={n}, dist={dist_json}",
            )
            report.invariant("P2", used >= 2, context=variant, detail=f"workers={used}")

        slow_note = (
            f", speed_injection={perf_engine} (observational: engine speed "
            f"is not a prefill-score input)"
            if perf_engine
            else ""
        )
        return report.finish(
            f"variants=plain+speed_hetero, n={n}, grades: {report.summary()}"
            f"{slow_note}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if perf_engine:
            try:
                ops.set_perf(perf_engine, prefill_fixed_ms=100.0)
            except Exception:
                pass
