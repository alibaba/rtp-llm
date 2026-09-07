from __future__ import annotations

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    PREFIX_INPUT_LEN,
    STREAM_TIMEOUT_S,
    _cache_evict,
    _contiguous_prefix_len,
    _engine_cache_keys,
    _fam_keys,
    _kv_spec,
    _prefill_names,
    _seed_shared_prefix,
    _wait_cache_sync,
)


@case(
    "kv_pe_prefix_continuity",
    category="kv",
    source="kv family: continuous-prefix matching, gap truncates",
)
def kv_pe_prefix_continuity(ctx: CaseContext):
    """[per-engine] Continuous-prefix matching: a gap truncates the hit.

    Scenario: the ledger-separated double dispatch shares family-0
    between e1 and e2; /cache_evict then removes k2 from e1 (leaving 9
    blocks with a GAP) and k9,k10 from e2 (leaving the first 8 blocks
    CONTIGUOUS).  Behaviour: prefix-hit accounting.  Expected
    (contract): a request for [k1..k10] matches only e2's contiguous
    run — 8 blocks, hitTokens 8192, the M3 half-hit frame (a partial
    hit still concentrates on its holder); a block-COUNT caliber
    instead would rank e1 (9 blocks) over e2 (8) and route there, and a
    suffix-match caliber would equalize the two.  The 5-request batch
    must therefore land on e2 (M3).  Prediction: passes.
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

        # -- carve the two views: e1 loses k2 (gap), e2 loses k9,k10
        #    (trailing blocks — its k1..k8 run stays contiguous).
        _cache_evict(ops, e1, [fam0[1]])
        _cache_evict(ops, e2, [fam0[8], fam0[9]])
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after carving"
        e1_run = _contiguous_prefix_len(_engine_cache_keys(ops, e1), fam0)
        e2_run = _contiguous_prefix_len(_engine_cache_keys(ops, e2), fam0)
        carved_ok = e1_run == 1 and e2_run == 8

        # -- M3: the 5-request batch concentrates on the LONGEST
        #    CONTIGUOUS run holder (e2).  A count caliber would send it
        #    to e1 (9 blocks); a suffix caliber would spread it.
        addrs = []
        for _ in range(5):
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
                "M3",
                hits / len(addrs),
                context="continuity",
                detail=(
                    f"contiguous_holder={e2}(run={e2_run}), "
                    f"gapped={e1}(run={e1_run}), hits={hits}/{len(addrs)}"
                ),
            )
        passed, detail, rep = report.finish(
            f"runs: {e1}={e1_run}, {e2}={e2_run}, grades: {report.summary()}"
        )
        return passed and carved_ok, f"carved_ok={carved_ok}, {detail}", rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
