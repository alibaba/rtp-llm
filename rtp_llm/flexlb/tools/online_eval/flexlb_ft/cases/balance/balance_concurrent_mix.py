from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.balance import STREAM_TIMEOUT_S


@case(
    "balance_concurrent_mix",
    category="balance",
    source="scheduling_smoke.py S7 (strengthened)",
)
def balance_concurrent_mix(ctx: CaseContext):
    """A concurrent mixed burst must not collapse onto a single engine.

    Result properties: P1 (relaxed band inheritance) + P2 no-starvation +
    P6 completeness over a 20-request / 20-way-concurrent burst, counted
    from client landing addresses.

    P1 band note (relax=1): under a concurrent burst the master may process
    the burst in several groups; within a group every request is evaluated
    against the same live ledger snapshot, so the split is a fresh uniform
    draw per group — group-splitting is CORRECT balancing behaviour, not a
    defect.  The effective bands are therefore shifted one tier right
    (strict→0.75, normal/loose→0.85): the calibrated loose floor (0.85,
    false-failure < 1% at 2 engines / 20 samples) is never widened past
    itself.

    P6 note (intake2 scheduler contract): the
    RequestScheduler/GroupPolicy architecture fails fast with retryable
    NO_PREFILL_WORKER (8402) once the per-engine inflight-batch admission
    ledger is saturated — dispatcher.maxInflightBatchesPerPrefillWorker
    (harness default 4) x 2 prefill workers = 8 concurrent batch
    reservations; each burst request is its own batch, so a 20-way burst
    exceeds the floor and the excess is rejected at select time via
    projection PROJECTION_BLOCKED_DELIVERY_CAPACITY_BATCH_ADMISSION
    (flexlb-sync BatchPrefillAdmission.reserveBatch +
    WorkerBatcher.admissionBlockUnderLock).  The old "master queues the
    entire burst" assumption no longer holds: completeness now means
    (a) no failure other than batch-admission backpressure, and (b) at
    least the 8-reservation floor admitted — the first 8 arrivals always
    find a permit (releases only raise the floor).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "balance")
    try:
        rids = [ops.next_request_id(base) for _ in range(20)]

        def run(rid: int):
            keys = [rid * 100 + j for j in range(3)]
            return ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(run, rids))
        addrs = []
        failures = []
        for rid, (addr, err) in zip(rids, results):
            if err:
                failures.append(f"rid={rid}: {err}")
            else:
                addrs.append(addr)

        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a in addrs)
        used = len(dist)
        n_ok = len(addrs)
        max_share = max(dist.values()) / n_ok if n_ok else 1.0
        dist_json = json.dumps(dict(dist), sort_keys=True)

        admission_denied = [f for f in failures if "NO_PREFILL_WORKER" in f]
        hard_failures = [f for f in failures if "NO_PREFILL_WORKER" not in f]
        report.invariant(
            "P6",
            not hard_failures and n_ok >= 8,
            detail=(
                f"completed={n_ok}/20, batch-admission-denied="
                f"{len(admission_denied)} (retryable 8402, intake2 "
                f"contract), failures={hard_failures[:2]}"
            ),
        )
        report.check(
            "P1",
            max_share,
            context="burst20",
            relax=1,
            detail=f"ok={n_ok}, dist={dist_json} (concurrent group-split "
            f"is correct behaviour — bands shifted one tier)",
        )
        report.invariant("P2", used >= 2, detail=f"workers={used}")

        return report.finish(
            f"burst=20x20-way, workers={used}, " f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
