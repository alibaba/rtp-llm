from __future__ import annotations

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    PREFIX_INPUT_LEN,
    STREAM_TIMEOUT_S,
    _cache_evict,
    _engine_cache_keys,
    _fam_keys,
    _kv_spec,
    _prefill_names,
    _seed_shared_prefix,
    _wait_cache_sync,
)


@case(
    "kv_g_partial_release_redirect",
    category="kv",
    source="kv family: partial release of a shared block",
)
def kv_g_partial_release_redirect(ctx: CaseContext):
    """[global] Partial release: one holder evicts -> redirect to the other.

    Scenario: family-0 shared between e1 and e2; /cache_evict releases
    it from e1 only (e1's own churn — the spec's small-A/big-A surface,
    expressed through the eviction endpoint because EnvSpec capacities
    are uniform across engines).  Behaviour: partial release of a
    shared block.  Expected (contract): after convergence e2 is the
    SOLE holder and the sole max-hit candidate — every subsequent
    same-prefix request redirects to e2 (P9 over 10 serial requests);
    a stale e1 entry would either pin traffic on e1 or equalize the hit
    into a spread.  Prediction: passes (blank-slate lock contract).
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

        # -- e1 releases its copy; e2 keeps the shared family.
        _cache_evict(ops, e1, fam0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after partial release"
        released_ok = not (set(fam0) & _engine_cache_keys(ops, e1)) and set(
            fam0
        ) <= _engine_cache_keys(ops, e2)

        # -- P9: every continuation redirects onto the surviving holder.
        addrs = []
        for _ in range(10):
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
            hits = sum(1 for n in addrs if n == e2)
            report.check(
                "P9",
                hits / len(addrs),
                context="partial_release",
                detail=f"survivor={e2}, hits={hits}/{len(addrs)}, " f"released={e1}",
            )
        passed, detail, rep = report.finish(
            f"survivor={e2}, released={e1}, grades: {report.summary()}"
        )
        return passed and released_ok, f"released_ok={released_ok}, {detail}", rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
