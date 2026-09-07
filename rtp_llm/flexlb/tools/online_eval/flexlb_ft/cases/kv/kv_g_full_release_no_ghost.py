from __future__ import annotations

import json
import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    P1_WAVE_N,
    PREFIX_INPUT_LEN,
    _cache_evict,
    _drain_fired,
    _engine_cache_keys,
    _fam_keys,
    _fire_request,
    _kv_spec,
    _prefill_names,
    _seed_shared_prefix,
    _wait_cache_sync,
)


@case(
    "kv_g_full_release_no_ghost",
    category="kv",
    source="kv family: full release leaves no ghost entries",
)
def kv_g_full_release_no_ghost(ctx: CaseContext):
    """[global] Full release leaves no ghost entries.

    Scenario: family-0 shared between e1 and e2; BOTH holders evict it;
    the sync converges (>= 3.5s of cache_version quiet).  Behaviour:
    full release of a shared block.  Expected (contract): the master's
    index carries no residue — same-prefix requests behave zero-hit
    (P1 spread over the fired batch, no engine pinned); a ghost entry
    would keep the family stuck on one engine (max-share ~1.0).
    Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    fired, fired_handles = [], {}
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

        _cache_evict(ops, e1, fam0)
        _cache_evict(ops, e2, fam0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after full release"
        released_ok = all(
            not (set(fam0) & _engine_cache_keys(ops, n)) for n in (e1, e2)
        )

        # -- fired batch (two-phase, decisions inside one sync window):
        #    zero-hit tie-window spread — no ghost stickiness.  Hedge
        #    note (deliberate, not incidental): the 0.12s fire spacing
        #    keeps every decision inside the live ~2s ledger window, so
        #    later fires hedge away from earlier landings — the spread is
        #    negatively correlated and the real false-fail rate sits BELOW
        #    the independent-binomial nominal (P1_WAVE_N calibration);
        #    widening the fire spacing silently removes this protection.
        wave_names = []
        for _ in range(P1_WAVE_N):
            rid = ops.next_request_id(base)
            name, err = _fire_request(
                ops,
                rid,
                fired,
                fired_handles,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
            )
            if err:
                report.invariant("P6", False, detail=f"wave fire failed: {err}")
                break
            wave_names.append(name)
            time.sleep(0.12)
        _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}
        if wave_names:
            dist = {}
            for n in wave_names:
                dist[n] = dist.get(n, 0) + 1
            max_share = max(dist.values()) / len(wave_names)
            report.check(
                "P1",
                max_share,
                context="full_release",
                detail=(
                    f"both_evicted={{{e1}, {e2}}}, "
                    f"dist={json.dumps(dist, sort_keys=True)}"
                ),
            )
            passed, detail, rep = report.finish(f"grades: {report.summary()}")
            return (
                passed and released_ok,
                f"released_ok={released_ok}, {detail}",
                rep,
            )
        passed, detail, rep = report.finish("wave never fired")
        return False, detail, rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)
