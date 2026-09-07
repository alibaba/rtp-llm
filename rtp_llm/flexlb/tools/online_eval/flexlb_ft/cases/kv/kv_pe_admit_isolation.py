from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    PREFIX_INPUT_LEN,
    STREAM_TIMEOUT_S,
    _drain_fired,
    _engine_cache_keys,
    _fam_keys,
    _fire_request,
    _kv_spec,
    _poll_engine_pending,
    _prefill_names,
    _wait_cache_sync,
)


@case(
    "kv_pe_admit_isolation",
    category="kv",
    source="kv family: per-engine cache ledger isolation",
)
def kv_pe_admit_isolation(ctx: CaseContext):
    """[per-engine] A's admissions never widen B's key set.

    Scenario: ledger-separated seeding (the kv_prefix_stickiness
    technique) pins family-0 on engine A and family-5 on engine B; with
    B slowed to 5s, families 1..4 are zero-hit and their ledger pricing
    admits them on A only.  Behaviour: per-engine admit accounting in
    the master's cache-status index.  Expected (contract): only A's
    cache_key_set grows with the four families — B's stays exactly its
    own seed family — and subsequent same-prefix continuations stick to
    A (P9); a global broadcast of A's admits would equalize the hit and
    dissolve the affinity into spread.  Prediction: passes.
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

        # -- ledger-separated seeding: fam0 -> A, fam5 -> the other engine.
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=2000.0)
        time.sleep(1.5)  # master perf sync
        fam0 = _fam_keys(base, 0)
        rid_a = ops.next_request_id(base)
        a_name, err = _fire_request(
            ops,
            rid_a,
            fired,
            fired_handles,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
        )
        if err is None and not _poll_engine_pending(ops, a_name, 1):
            err = f"seed A never appeared on {a_name}"
        _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}
        b_name, err_b = None, "seed A failed"
        if err is None:
            fam5 = _fam_keys(base, 5)
            rid_b = ops.next_request_id(base)
            addr_b, err_b = ops.run_one_request(
                rid_b,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam5,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            b_name = ops.addr_to_name().get(addr_b, addr_b)
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=100.0)
        if err or err_b:
            return False, f"seeding failed: {err or err_b}"
        if a_name == b_name:
            return False, f"family separation failed: both seeds on {a_name}"

        # -- B stays heavy: zero-hit families 1..4 divert onto A only.
        ops.set_perf(b_name, prefill_fixed_ms=5000.0)
        time.sleep(1.5)
        fams = [_fam_keys(base, i) for i in range(1, 5)]
        for keys in fams:
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=keys,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                return False, f"family admit request failed: {err}"
            if ops.addr_to_name().get(addr, addr) != a_name:
                return False, (
                    f"zero-hit family landed on {addr} instead of {a_name} "
                    f"(B-slow diversion failed)"
                )
        ops.set_perf(b_name, prefill_fixed_ms=100.0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after family admits"

        # -- contract: A carries the four families, B does not.
        expected = {k for keys in fams for k in keys}
        a_keys = _engine_cache_keys(ops, a_name)
        b_keys = _engine_cache_keys(ops, b_name)
        missing_on_a = sorted(expected - a_keys)[:4]
        leaked_to_b = sorted(expected & b_keys)[:4]
        isolation_ok = not missing_on_a and not leaked_to_b

        # -- P9: same-prefix continuations stick to the sole holder A.
        addrs = []
        for i in range(10):
            keys = fams[i % 4]
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=keys,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                report.invariant("P6", False, detail=f"continuation failed: {err}")
                break
            addrs.append(ops.addr_to_name().get(addr, addr))
        if addrs:
            hits = sum(1 for n in addrs if n == a_name)
            report.check(
                "P9",
                hits / len(addrs),
                context="fam1_4",
                detail=f"holder={a_name}, hits={hits}/{len(addrs)}, " f"other={b_name}",
            )

        # -- final mock view: B never picked up A's families (a stray
        #    continuation onto B would admit there — caught here too).
        b_keys_after = _engine_cache_keys(ops, b_name)
        leaked = sorted((expected | set(fam0)) & b_keys_after)[:4]
        mock_ok = not leaked
        passed, detail, rep = report.finish(
            f"holder={a_name}, other={b_name}, " f"grades: {report.summary()}"
        )
        return (
            passed and isolation_ok and mock_ok,
            f"isolation_ok={isolation_ok} "
            f"(A missing={len(expected - a_keys)}, B leaked="
            f"{len(expected & b_keys)}), final_B_leak={len(leaked)}"
            f"{' e.g. ' + str(leaked) if leaked else ''}, {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)
        for name in _prefill_names(ops):
            try:
                ops.set_perf(name, prefill_fixed_ms=100.0)
            except Exception:
                pass
