from __future__ import annotations

import json
from collections import Counter

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.balance import (
    _drain_fired,
    _fire_request,
    _master_http,
    _poll_engine_pending,
    _prefill_names,
)


@case(
    "balance_len_mixed",
    category="balance",
    source="length-heterogeneity dimension L1",
)
def balance_len_mixed(ctx: CaseContext):
    """Bimodal length mix balances TOKEN footprint, not request count.

    Result properties (graded): P3 token-weighted max-share (first
    calibrated measurement of the P3 band), P2 short-request spread (both
    engines take short work), P6 completeness.

    Construction (5 waves, each 2 long + 6 short fire-and-forget):
      * ONE formula for both sides: mock execution time and the
        master's ledger prediction share the production DSv4 fit, so a
        long request's ledger entry decays on exactly the clock the mock
        sleeps — the diversion window equals the fitted prefill time;
      * long ladder 131072..147456: the fit predicts ~1.72-2.06 s all-miss,
        wide enough to choreograph a wave well inside the window (the old
        32k ladder predicted ~342 ms — too narrow to orchestrate against,
        which the retired set_perf(3000ms) crutch used to paper over);
      * wave choreography (S4 ledger technique, symmetric — no single hot
        engine is manufactured):
        1. L_a fired on an empty ledger pair -> uniform tie-window pick (X);
        2. poll X pending (engine-side proof the ledger entry exists);
        3. L_b fired immediately after that proof: X carries the FULL
           fitted ledger (~1.72-2.06 s) while Y is still empty — a gap
           ~20x the 10%/20ms tie window, so L_b deterministically lands
           on Y with no timing assumption about when shorter requests
           register their own ledger entries (the t+200ms variant of this
           step raced the shorts' registration and split the longs 7/3);
        4. poll Y pending (both longs now provably in flight);
        5. 6 shorts fired while BOTH longs execute: X has decayed only by
           the polls' overhead (tens of ms — inside the ~190ms tie window
           of the ~1.9s ledgers), so the shorts spread across both
           engines; the exact split does not matter for P3 because the
           shorts carry ~1% of the wave's tokens;
        6. drain the wave to terminal state + master inflight clean,
           so the next wave starts from a settled (double-zero) ledger.
      * per wave: X = L_a + ~3 shorts, Y = L_b + ~3 shorts in tokens; the
        ladder is cyclic (L_a series == L_b series), so the aggregate
        token share is pinned at ~0.50 for ANY short split (see the P3
        calibration note in grade.GRADE_BANDS).

    Why not request-count uniformity (P1): the wave deliberately lands
    BOTH long requests' complements asymmetrically in flight order —
    token balance and request-count balance genuinely conflict in this
    scene, and P3 is the property that matters.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "balance")
    prefill_names: list[str] = []
    fired: list[tuple[int, object]] = []
    fired_handles: dict[int, object] = {}
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"

        # deterministic bimodal ladder 131072..147456 (reproducible reruns):
        # ~1.72-2.06 s fitted all-miss prefill = the ledger window the wave
        # choreography runs inside (same formula on mock and master sides)
        long_lens = [131072 + (i % 5) * 4096 for i in range(10)]
        landed: list[tuple[int, str, int]] = []  # (rid, engine_name, input_len)
        failure = None

        for wave in range(5):
            la, lb = long_lens[2 * wave], long_lens[2 * wave + 1]
            # 1. L_a on the empty ledger pair.
            rid = ops.next_request_id(base)
            name_a, err = _fire_request(
                ops, rid, fired, fired_handles, input_len=la, output_len=2
            )
            if err:
                failure = f"wave{wave} L_a: {err}"
                break
            landed.append((rid, name_a, la))
            # 2. engine-side proof the ledger entry is live.
            if not _poll_engine_pending(ops, name_a, 1):
                failure = f"wave{wave} L_a never appeared on {name_a}"
                break
            # 3. L_b immediately after the L_a dispatch proof: X carries the
            #    full fitted ledger (~1.72-2.06 s) while Y is still empty —
            #    ~20x the tie window, so L_b deterministically lands on Y
            #    (no dependency on the shorts' ledger registration timing).
            rid = ops.next_request_id(base)
            name_b, err = _fire_request(
                ops, rid, fired, fired_handles, input_len=lb, output_len=2
            )
            if err:
                failure = f"wave{wave} L_b: {err}"
                break
            landed.append((rid, name_b, lb))
            if not _poll_engine_pending(ops, name_b, 1):
                failure = f"wave{wave} L_b never appeared on {name_b}"
                break
            # 4. 6 shorts while BOTH longs are in flight: X has decayed only
            #    by the polls' overhead (tens of ms, inside the ~190ms tie
            #    window of the ~1.9s ledgers), so the shorts spread evenly —
            #    both engines take short work (P2) and the wave stays
            #    token-symmetric for any exact split.
            for short_idx in range(6):
                rid = ops.next_request_id(base)
                name, err = _fire_request(
                    ops, rid, fired, fired_handles, input_len=512, output_len=2
                )
                if err:
                    failure = f"wave{wave} short#{short_idx}: {err}"
                    break
                landed.append((rid, name, 512))
            if failure:
                break

            # 6. drain the wave before the next one starts clean.
            outcomes = _drain_fired(ops, fired, fired_handles)
            unfinished = [(r, n, e) for (r, n, ok, e) in outcomes if not ok]
            if unfinished:
                failure = f"wave{wave} drain incomplete: {unfinished[:2]}"
                # drop the undrained tail from landed so P3 counts only
                # completed traffic
                bad_rids = {r for r, _n, _e in unfinished}
                landed = [t for t in landed if t[0] not in bad_rids]
                break
            fired.clear()
            fired_handles.clear()
            clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
            if not clean_ok:
                failure = f"wave{wave} inflight not clean: {clean_detail}"
                break

        if failure:
            report.invariant("P6", False, detail=failure)
        else:
            report.invariant("P6", True, detail="all 40 requests drained")

        if landed:
            token_by_engine: Counter = Counter()
            short_by_engine: Counter = Counter()
            for _rid, name, ln in landed:
                token_by_engine[name] += ln
                if ln == 512:
                    short_by_engine[name] += 1
            total_tokens = sum(token_by_engine.values())
            max_share = (
                max(token_by_engine.values()) / total_tokens if total_tokens else 1.0
            )
            short_engines = len(short_by_engine)
            tokens_json = json.dumps(
                {k: token_by_engine[k] for k in sorted(token_by_engine)},
                sort_keys=True,
            )
            report.check(
                "P3",
                max_share,
                context="bimodal_5waves",
                detail=f"tokens={tokens_json}, shorts={dict(short_by_engine)}",
            )
            report.invariant(
                "P2",
                short_engines >= 2,
                context="short_spread",
                detail=f"engines taking shorts={short_engines}",
            )

        return report.finish(
            f"waves=5, landed={len(landed)}/40, " f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if fired or fired_handles:
            _drain_fired(ops, fired, fired_handles)
        try:
            # Best-effort residue drain with the TTL-aware window (task
            # #87 — same rationale as balance_overload_avoid_prefill: a
            # drain-fallback cancel that fails settles on the stale-TTL +
            # ExpirationTimer path, worst ~90s).
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        except Exception:
            pass
