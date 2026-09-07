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
    _wait_cache_sync,
)


@case(
    "kv_pe_evict_zero_match",
    category="kv",
    source="kv family: eviction event syncs through the master index",
)
def kv_pe_evict_zero_match(ctx: CaseContext):
    """[per-engine] A forced evict syncs through: no stale stickiness.

    Scenario: family-0 is primed on its landing engine X (capacity 16 —
    the spec's tiny-capacity spirit, sized up from 4 so the 10-block
    prefix survives the LRU and prices past the affinity line); the
    sync converges; a positive-control request sticks to X; then
    /cache_evict removes the whole family from X and the sync converges
    again (>= 3.5s of cache_version quiet).  Behaviour: master-side
    propagation of the eviction event.  Expected (contract): the
    post-evict batch carries the same prefix but must NOT stick to X —
    zero-hit tie-window spread over the fired batch (P1); a stale
    master index would keep routing the family onto X (max-share 1.0).
    Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx, "_evict", prefill_cache_blocks=16))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)

        # -- prime: wherever family-0 first lands is the holder X.
        rid_prime = ops.next_request_id(base)
        addr_x, err = ops.run_one_request(
            rid_prime,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"prime failed: {err}"
        x_name = ops.addr_to_name().get(addr_x, addr_x)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after prime"

        # -- positive control: affinity is live — the replay sticks to X.
        rid_ctl = ops.next_request_id(base)
        addr_ctl, err_ctl = ops.run_one_request(
            rid_ctl,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        ctl_name = ops.addr_to_name().get(addr_ctl, addr_ctl)
        if err_ctl:
            return False, f"positive control failed: {err_ctl}"
        if ctl_name != x_name:
            return False, (
                f"positive control landed on {ctl_name} instead of holder "
                f"{x_name} — affinity was not live before the evict"
            )

        # -- evict the whole family from X, wait for the sync to settle.
        _cache_evict(ops, x_name, fam0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after evict"
        x_keys = _engine_cache_keys(ops, x_name)
        evicted_ok = not (set(fam0) & x_keys)

        # -- negative control: a fired batch (two-phase, decisions inside
        #    one sync window) must spread — no stale stickiness to X.
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
        outcomes = _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}
        if wave_names:
            dist = {}
            for n in wave_names:
                dist[n] = dist.get(n, 0) + 1
            max_share = max(dist.values()) / len(wave_names)
            share_x = dist.get(x_name, 0) / len(wave_names)
            report.check(
                "P1",
                max_share,
                context="post_evict",
                detail=(
                    f"holder_evicted={x_name}, share_x={share_x:.2f}, "
                    f"dist={json.dumps(dist, sort_keys=True)}"
                ),
            )
            passed, detail, rep = report.finish(
                f"evicted_holder={x_name}, grades: {report.summary()}"
            )
            return (
                passed and evicted_ok,
                f"evicted_from_holder={evicted_ok}, {detail}",
                rep,
            )
        passed, detail, rep = report.finish("wave never fired")
        return False, detail, rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)
