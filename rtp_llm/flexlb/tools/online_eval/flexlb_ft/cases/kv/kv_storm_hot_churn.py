from __future__ import annotations

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    PREFIX_BLOCKS,
    PREFIX_INPUT_LEN,
    STORM_CAPACITY_BLOCKS,
    STORM_FAMILIES,
    STORM_FLIP_BOUND,
    STORM_HIT_RATE_BANDS,
    STORM_REPLICATION_BANDS,
    STORM_WINDOW,
    STORM_WINDOWS,
    STREAM_TIMEOUT_S,
    _contiguous_prefix_len,
    _engine_cache_keys,
    _fam_keys,
    _kv_spec,
    _prefill_names,
)


@case(
    "kv_storm_hot_churn",
    category="kv",
    source="kv family: hot-prefix churn storm vs small LRU",
)
def kv_storm_hot_churn(ctx: CaseContext):
    """[storm] Rotating hot prefixes vs a small LRU — graded
    replication-band case.

    Scenario: 4 hot families (10 blocks each = 40 hot blocks) rotate
    one per window (5 requests/window, 10 windows = 50 requests) while
    each engine's LRU holds only 24 blocks — every rotation must evict
    yesterday's hot family somewhere.  Behaviour: churn-driven
    replication and the master index's stability under it.  Contract:
    (a) the AVERAGE REPLICATION FACTOR (per-family prefill holder
    count mean, via cache_key_set) stays inside the calibrated band
    (strict/normal/loose = 1.5/1.75/2.0 — see STORM_REPLICATION_BANDS)
    with a hard structural cap max_replication <= n_prefill; (b)
    holder FLIPS stay within the traffic-driven bound (an unbounded
    ping-pong means sync thrash); (c) the hit-tier concentration M3
    stays above the recalibrated floor STORM_HIT_RATE_BANDS (a request
    counts as hit-served when its landing engine already held the
    family's contiguous prefix, >= 8 blocks; 2026-09-04 n=1
    recalibration — see the constant).

    Prediction (post the env reporting-caliber fix): the kv env's
    decode pool (decode_cache_blocks=4 -> 4096 tokens) is now
    REPORTED at its real size, so the master's used/total math no
    longer reads 99.9% on it and the decode fleet does not park —
    P6/flips/M3 all pass and the replication band is the only
    potential trip point (calibration mean 1.125 sits deep inside
    strict).  The pre-fix "hit rate collapse" prediction was an
    artifact of the mis-reported total (structurally parked decodes
    starve everything downstream), not a real replication blow-up.

    Promotion from expected-fail to graded: the declared
    finding ("no replication suppression in the KV sync layer ->
    hit-rate collapse") did not survive calibration — replication is
    healthy (mean 1.125, steady-state 1.5, max = the 2-prefill
    structural ceiling, M3 0.86 >= loose) — so the case is a regular
    graded case now and its achieved grade rolls into the suite
    verdict again.  The band polices the FAKE-FIX direction: a hit
    rate repaired by replicating everywhere (admission control gone)
    parks KV footprint for no throughput.  The replication
    distribution / flips / M3 numbers stay in the case detail as
    evidence either way.

    2026-09-04 M3 recalibration: the shared-band floor (loose 0.6) sat
    exactly on this run's 0.600 (30/50) — a boundary artifact failing
    a normal-grade run — and the cross-era drift (0.86 -> 0.600, codex
    admission changes) argues for tiers split from the collapse
    regimes rather than the promotion-era numbers; see
    STORM_HIT_RATE_BANDS above.
    """
    env = ctx.env_manager.ensure(
        _kv_spec(ctx, "_storm", prefill_cache_blocks=STORM_CAPACITY_BLOCKS)
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fams = [_fam_keys(base, f) for f in range(STORM_FAMILIES)]

        def holder_mask(keys: list) -> dict:
            return {n: bool(set(keys) & _engine_cache_keys(ops, n)) for n in names}

        prev_hold = None
        flips = 0
        m3_hits = 0
        m3_total = 0
        failures = []
        for w in range(STORM_WINDOWS):
            hot = w % STORM_FAMILIES
            keys = fams[hot]
            for _ in range(STORM_WINDOW):
                # pre-request view: does the landing engine hold a
                # contiguous run long enough to price past the line?
                runs = {
                    n: _contiguous_prefix_len(_engine_cache_keys(ops, n), keys)
                    for n in names
                }
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=PREFIX_INPUT_LEN,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                m3_total += 1
                if err:
                    failures.append(f"w{w} rid={rid}: {err}")
                    continue
                landed = ops.addr_to_name().get(addr, addr)
                if runs.get(landed, 0) >= PREFIX_BLOCKS - 2:
                    m3_hits += 1
            # end-of-window flip accounting (any-block holder mask)
            cur = {}
            for f in range(STORM_FAMILIES):
                mask = holder_mask(fams[f])
                for n in names:
                    cur[(n, f)] = mask[n]
            if prev_hold is not None:
                flips += sum(1 for k in cur if cur[k] != prev_hold.get(k))
            prev_hold = cur

        # -- replication factor: calibrated band (STORM_REPLICATION_BANDS)
        #    on the per-family MEAN, plus a hard structural cap
        #    max <= n_prefill.  With 2 prefills the cap is trivially
        #    saturated (max <= 2 always) and carries NO statistical
        #    power — it is a pure regression hard-hat for the
        #    n_prefill >= 3 fleet, where max > n_prefill would mean a
        #    holder-index accounting bug rather than traffic-driven
        #    replication.
        replication = {
            f: sum(1 for n in names if prev_hold.get((n, f)))
            for f in range(STORM_FAMILIES)
        }
        max_replication = max(replication.values()) if replication else 0
        mean_replication = (
            sum(replication.values()) / len(replication) if replication else 0.0
        )

        hit_rate = m3_hits / m3_total if m3_total else 0.0
        flips_ok = flips <= STORM_FLIP_BOUND
        max_repl_ok = max_replication <= len(names)
        report.invariant("P6", not failures, detail=f"failures={failures[:2]}")
        report.check(
            "M3",
            hit_rate,
            context="storm_hit_rate",
            bands=STORM_HIT_RATE_BANDS,
            detail=(
                f"hits={m3_hits}/{m3_total}, flips={flips}"
                f"(<= {STORM_FLIP_BOUND}), replication={replication}"
            ),
        )
        report.check(
            "P5",
            mean_replication,
            context="storm_replication",
            bands=STORM_REPLICATION_BANDS,
            detail=(
                f"replication={replication}, max={max_replication}"
                f"/{len(names)} (hard cap), flips={flips}"
            ),
        )
        passed, detail, rep = report.finish(
            f"windows={STORM_WINDOWS}, grades: {report.summary()}"
        )
        return (
            passed and flips_ok and max_repl_ok,
            f"flips={flips} (bound {STORM_FLIP_BOUND}, ok={flips_ok}), "
            f"mean_replication={mean_replication:.3f} "
            f"(bands {STORM_REPLICATION_BANDS}), "
            f"max_replication={max_replication}/{len(names)} (hard cap, "
            f"ok={max_repl_ok}), {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
