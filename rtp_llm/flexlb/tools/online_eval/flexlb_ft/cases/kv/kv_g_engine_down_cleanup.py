from __future__ import annotations

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
    _wait_master_alive,
)


@case(
    "kv_g_engine_down_cleanup",
    category="kv",
    source="kv family: engine removal cleans only its own entries",
)
def kv_g_engine_down_cleanup(ctx: CaseContext):
    """[global] Engine-down cleanup keeps shared blocks for the survivor.

    Scenario: 3-prefill env with dynamic file discovery; family-0 is
    double-dispatched onto h1 and h2 (the third engine holds nothing);
    remove_engine takes h1 down permanently.  Behaviour: index cleanup
    on engine removal.  Expected (contract): ONLY h1's holder entries
    are dropped — the shared family stays attributed to h2, so
    post-removal same-prefix requests keep landing on h2 (P9 over 5
    requests) and the survivor's key set is untouched; a cleanup that
    drops the whole key entry would orphan h2's cache and scatter the
    family uniformly across the remaining engines.  Prediction:
    UNCERTAIN — the removal -> index-cleanup wiring has never been
    verified; a failure is a finding (over-cleanup or leak).
    """
    env = ctx.env_manager.ensure(
        _kv_spec(ctx, "_down", n_prefill=3, discovery="discovery_file")
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    removed = False
    try:
        names = _prefill_names(ops)
        if len(names) < 3:
            return False, "need >=3 prefill workers"
        fam0 = _fam_keys(base, 0)

        # -- share fam0 across two of the three engines.
        h1, h2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"shared seeding failed: {err}"
        other = [n for n in names if n not in (h1, h2)]
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after seeding"
        shared_ok = (
            set(fam0) <= _engine_cache_keys(ops, h1)
            and set(fam0) <= _engine_cache_keys(ops, h2)
            and all(not (set(fam0) & _engine_cache_keys(ops, n)) for n in other)
        )

        # -- h1 goes down; the master's alive count must follow.
        status, body = ops.remove_engine(engine_name=h1)
        removed = status == 200
        if not removed:
            return False, f"remove_engine({h1}) failed: {status} {body}"
        if not _wait_master_alive(ops, "PREFILL", len(names) - 1):
            return False, (
                f"master did not converge to {len(names) - 1} alive prefill "
                f"engines after removal (alive="
                f"{ops.master_alive_count('PREFILL')})"
            )

        # -- P9: the family keeps routing to the surviving holder h2.
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
        survivor_kept = set(fam0) <= _engine_cache_keys(ops, h2)
        if addrs:
            hits = sum(1 for n in addrs if n == h2)
            report.check(
                "P9",
                hits / len(addrs),
                context="engine_down",
                detail=(
                    f"survivor={h2}, removed={h1}, hits={hits}/{len(addrs)}, "
                    f"other_alive={other}"
                ),
            )
        passed, detail, rep = report.finish(
            f"survivor={h2}, removed={h1}, grades: {report.summary()}"
        )
        return (
            passed and shared_ok and survivor_kept,
            f"shared_ok={shared_ok}, survivor_kept={survivor_kept}, {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if removed:
            # env hygiene: restore the prefill count for later reuse
            try:
                ops.add_engine("prefill")
            except Exception:
                pass
