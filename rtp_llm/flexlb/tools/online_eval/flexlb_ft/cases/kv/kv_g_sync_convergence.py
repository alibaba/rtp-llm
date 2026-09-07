from __future__ import annotations

import json
import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    P1_WAVE_N,
    PREFIX_INPUT_LEN,
    STREAM_TIMEOUT_S,
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
    "kv_g_sync_convergence",
    category="kv",
    source="kv family: mixed admit/evict stream converges to snapshot truth",
)
def kv_g_sync_convergence(ctx: CaseContext):
    """[global] Mixed admit/evict sequences converge to snapshot truth.

    Scenario: an INTERLEAVED event stream — family-0 double-dispatched
    (2 admits) then evicted from e1; family-1 admitted once (landing
    spot s1 recorded); family-0 evicted from e2 too; family-2
    double-dispatched then evicted from one side — then >= 3.5s of
    silence.  Behaviour: incremental sync under a mixed event stream
    (the out-of-order / dropped-update paths have never been
    exercised).  Expected (contract): post-silence routing matches the
    ENGINE snapshots — family-0 (no holder) spreads (P1 over the fired
    batch), family-1 (sole holder s1) sticks (P9), family-2 (sole
    holder d2) sticks (P9); a ghost or a lost update lands in the wrong
    bucket.  Prediction: UNCERTAIN — the reorder/drop path is
    untested; a failure here is a finding, not a flake to retry away.
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
        fam1 = _fam_keys(base, 1)
        fam2 = _fam_keys(base, 2)

        # -- op 1-2: admit fam0 on both engines (double dispatch).
        e1, e2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"fam0 double dispatch failed: {err}"

        # -- op 3: partial evict of fam0 from e1.
        _cache_evict(ops, e1, fam0)

        # -- op 4: admit fam1 on its natural landing spot.
        rid_f1 = ops.next_request_id(base)
        addr_s1, err = ops.run_one_request(
            rid_f1,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam1,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"fam1 admit failed: {err}"
        s1 = ops.addr_to_name().get(addr_s1, addr_s1)

        # -- op 5: full evict of fam0 (the e2 side).
        _cache_evict(ops, e2, fam0)

        # -- op 6-7: admit fam2 on both, then evict one side.
        d1, d2, err = _seed_shared_prefix(ops, base, fam2, PREFIX_INPUT_LEN)
        if err:
            return False, f"fam2 double dispatch failed: {err}"
        _cache_evict(ops, d1, fam2)

        # -- silence: >= 3.5s of cache_version quiet, then the state is
        #    what it is — routing must agree with the engine snapshots.
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after the mixed stream"
        holder_keys = {n: _engine_cache_keys(ops, n) for n in names}
        fam0_ghost = any(set(fam0) & ks for ks in holder_keys.values())
        fam1_sole = sorted(n for n, ks in holder_keys.items() if set(fam1) <= ks)
        fam2_sole = sorted(n for n, ks in holder_keys.items() if set(fam2) <= ks)
        snapshot_ok = not fam0_ghost and fam1_sole == [s1] and fam2_sole == [d2]

        # -- fam0 (no holder): fired batch spreads (no ghost stickiness).
        #    Hedge note (deliberate, not incidental): the 0.12s fire
        #    spacing keeps every decision inside the live ~2s ledger
        #    window, so later fires hedge away from earlier landings —
        #    the spread is negatively correlated and the real false-fail
        #    rate sits BELOW the independent-binomial nominal (P1_WAVE_N
        #    calibration); widening the fire spacing silently removes
        #    this protection.
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
            report.check(
                "P1",
                max(dist.values()) / len(wave_names),
                context="fam0_no_holder",
                detail=f"dist={json.dumps(dist, sort_keys=True)}",
            )

        # -- fam1/fam2 (sole holders): continuations stick (P9 x2).
        for label, keys, holder in (
            ("fam1_sole", fam1, s1),
            ("fam2_sole", fam2, d2),
        ):
            addrs = []
            for _ in range(5):
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=PREFIX_INPUT_LEN,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    report.invariant(
                        "P6", False, detail=f"{label} request failed: {err}"
                    )
                    break
                addrs.append(ops.addr_to_name().get(addr, addr))
            if addrs:
                hits = sum(1 for n in addrs if n == holder)
                report.check(
                    "P9",
                    hits / len(addrs),
                    context=label,
                    detail=f"holder={holder}, hits={hits}/{len(addrs)}",
                )
        passed, detail, rep = report.finish(
            f"final_holders: fam1={s1}, fam2={d2}, grades: {report.summary()}"
        )
        return (
            passed and snapshot_ok,
            f"snapshot_ok={snapshot_ok} (fam0_ghost={fam0_ghost}, "
            f"fam1_holders={fam1_sole}, fam2_holders={fam2_sole}), {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)
