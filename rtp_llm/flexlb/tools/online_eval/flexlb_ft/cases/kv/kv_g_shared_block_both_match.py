from __future__ import annotations

import json

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    PREFIX_INPUT_LEN,
    STREAM_TIMEOUT_S,
    _engine_cache_keys,
    _fam_keys,
    _kv_spec,
    _prefill_names,
    _seed_shared_prefix,
    _wait_cache_sync,
)


@case(
    "kv_g_shared_block_both_match",
    category="kv",
    source="kv family: shared holder set -> equal-hit tie",
)
def kv_g_shared_block_both_match(ctx: CaseContext):
    """[global] Both holders of a shared block match: tie, not fight.

    Scenario: the double dispatch shares family-0 between e1 and e2
    (the master's global index maps the blocks to a holder SET).
    Behaviour: affinity with two equal max-hit candidates.  Expected
    (contract): maxHit == minHit -> NO_CACHE_LEAD — subsequent
    same-prefix requests spread across the holders (P1 max-share over
    20 serial requests, P2 both engines used) and the holder-union
    share stays 100%; a one-holder-only index would instead pin every
    request onto a single engine (max-share ~1.0).  Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)

        e1, e2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"shared seeding failed: {err}"
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after seeding"

        shared_ok = set(fam0) <= _engine_cache_keys(ops, e1) and set(
            fam0
        ) <= _engine_cache_keys(ops, e2)

        # -- 20 serial same-prefix requests: equal-hit tie spreads.
        addrs = []
        for _ in range(20):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                report.invariant("P6", False, detail=f"request failed: {err}")
                break
            addrs.append(ops.addr_to_name().get(addr, addr))
        if addrs:
            dist = {}
            for n in addrs:
                dist[n] = dist.get(n, 0) + 1
            max_share = max(dist.values()) / len(addrs)
            used = len(dist)
            union_share = (dist.get(e1, 0) + dist.get(e2, 0)) / len(addrs)
            report.check(
                "P1",
                max_share,
                context="shared_both_match",
                detail=f"dist={json.dumps(dist, sort_keys=True)}",
            )
            report.invariant(
                "P2",
                used >= 2,
                context="shared_both_match",
                detail=f"workers={used}",
            )
            report.invariant(
                "P6",
                union_share == 1.0,
                context="holder_union",
                detail=f"holder-union share={union_share:.2f} (e1+e2)",
            )
        passed, detail, rep = report.finish(
            f"holders={{{e1}, {e2}}}, grades: {report.summary()}"
        )
        return passed and shared_ok, f"shared_ok={shared_ok}, {detail}", rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
