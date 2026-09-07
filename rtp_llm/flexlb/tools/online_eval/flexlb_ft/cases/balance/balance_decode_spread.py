from __future__ import annotations

import json

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.balance import STREAM_TIMEOUT_S, _decode_names


@case(
    "balance_decode_spread",
    category="balance",
    source="scheduling_smoke.py S3+S10 (merged)",
)
def balance_decode_spread(ctx: CaseContext):
    """Decode traffic spreads across the decode fleet at both small and
    large sample sizes.

    Result properties: P2 no-starvation (min engines used) + P1 distribution
    bound (case-calibrated 4-engine bands) + P6 completeness, parameterized
    over n=10 and n=50 tiers.  Landing points are engine-completed counts
    (decode deltas via mock snapshots).

    P1 band note (case override, 4-engine caliber): the decode selector is
    KV_USAGE_WEIGHTED_RANDOM, not a uniform draw — weights track per-worker
    KV residue left by earlier cases, so the 2-engine bands do not apply.
    Calibration (batch-window, engine-completed deltas): n=10
    observed max_share 0.40 (dist 1/4/3/2), n=50 observed 0.40 (dist
    20/10/13/7 — the KV-weighted draw routinely puts ~40% on the residue-
    heaviest engine).  Bands kept at n=10: 0.60/0.70/0.80, n=50:
    0.40/0.50/0.60 — the n=50 strict tier sits ON the observed mode (a
    quality bar for weight convergence, not a statistical guarantee);
    widen from full-suite regression data if the
    residue-inheritance across predecessor cases pushes it over.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    try:
        tiers = (
            # (n, min_used, P1 case bands)
            (10, 2, {"strict": 0.60, "normal": 0.70, "loose": 0.80}),
            (50, 3, {"strict": 0.40, "normal": 0.50, "loose": 0.60}),
        )
        for n, min_used, bands in tiers:
            decode_names = _decode_names(ops)
            if len(decode_names) < 2:
                return False, "need >=2 decode workers"
            snap0 = ops.snapshot_by_name()
            baseline = {name: snap0[name].get("completed", 0) for name in decode_names}

            failures = []
            for _ in range(n):
                rid = ops.next_request_id(rid_base(ctx, "balance"))
                keys = [rid * 100 + j for j in range(3)]
                _, err = ops.run_one_request(
                    rid,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"rid={rid}: {err}")

            snap1 = ops.snapshot_by_name()
            deltas = {
                name: snap1[name].get("completed", 0) - baseline.get(name, 0)
                for name in decode_names
            }
            total = sum(deltas.values())
            used = sum(1 for v in deltas.values() if v > 0)
            max_share = max(deltas.values()) / n if n else 1.0
            deltas_json = json.dumps(deltas, sort_keys=True)

            report.invariant(
                "P6",
                not failures and total >= n,
                context=f"n{n}",
                detail=f"failures={failures[:2]}, total_delta={total}/{n}",
            )
            report.invariant(
                "P2",
                used >= min_used,
                context=f"n{n}",
                detail=f"used={used}/{len(decode_names)} (need >= {min_used})",
            )
            report.check(
                "P1",
                max_share,
                context=f"n{n}",
                bands=bands,
                detail=f"dist={deltas_json} (4-engine KV-weighted caliber)",
            )

        return report.finish(f"tiers=n10+n50, grades: {report.summary()}")
    except Exception as exc:
        return False, f"exception: {exc!r}"
