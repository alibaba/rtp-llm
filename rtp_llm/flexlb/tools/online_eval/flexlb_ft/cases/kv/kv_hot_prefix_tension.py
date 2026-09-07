from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import KV_CACHE_SYNC_WAIT_S, STREAM_TIMEOUT_S, _prefill_names


@case(
    "kv_hot_prefix_tension",
    category="kv",
    source="hot-prefix tension M2",
)
def kv_hot_prefix_tension(ctx: CaseContext):
    """A 70%-traffic hot prefix family: stickiness holds AND the holder's
    concentration stays capped.

    Result properties (graded combination): P9 family stickiness (graded —
    the design's tension axis 1), M2 holder total-share cap (graded upper
    bound — tension axis 2, first calibrated measurement of the M2 band),
    P2 free-flow no-starvation (the other engine still takes free traffic),
    P6 completeness.

    Construction: family F shares a 16-block long prefix (keys 3001-3016,
    input_len=16384).  One seed request lands on engine X (uniform initial
    pick); after the master cache sync the main phase runs 40 serial
    requests in a fixed 7:3 interleave — 28 family continuations (every one
    carries the ~10.7s estimate discount on X: est = 16384 - 0.7*15360 =
    5619 vs 16384 elsewhere, a gap ~20x the tie window -> deterministic
    stickiness) and 12 unique-key free requests (no affinity on either
    engine -> uniform tie-window spread).

    On X's accumulating state: each completed family request re-admits the
    same 16 blocks (LRU-refreshed, idempotent) and adds its inputLen to the
    mock's KV accounting — the holder's cache/KV footprint keeps growing
    across the phase (observational), while the routing ledger itself
    resets between serial requests (each completes before the next fires),
    which is what keeps P9 deterministic and pins the M2 model to the free
    flow's binomial spread.

    M2 caliber: family and free requests share input_len=16384, so token
    share and request share coincide; the holder's TOTAL share counts seed,
    family continuations AND the free requests that tie-window scatter onto
    it: (29 + k)/41 with k ~ B(12, 0.5) over the free flow (29 = seed + 28
    continuations deterministically on X when stickiness is perfect), i.e.
    ~0.854 ± 0.042 (1σ) — see the M2 calibration note in grade.GRADE_BANDS
    for the false-fail derivation of the band values.

    Free-flow starvation (P2): if all 12 free requests were swallowed by
    X the other engine would idle — that is the starvation this property
    forbids (probability 0.5**12 under correct uniform spread).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "kv")
    family_keys = list(range(3001, 3017))
    input_len = 16384
    try:
        # -- seed: family F prefix lands on X (uniform initial pick).
        rid_seed = ops.next_request_id(base)
        seed_addr, seed_err = ops.run_one_request(
            rid_seed,
            input_len=input_len,
            output_len=2,
            block_keys=family_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed_err:
            report.invariant("P6", False, detail=f"seed failed: {seed_err}")
            return report.finish(f"seed request failed: {seed_err}")
        addr_map = ops.addr_to_name()
        holder = addr_map.get(seed_addr, seed_addr)
        other_names = [n for n in _prefill_names(ops) if n != holder]
        time.sleep(KV_CACHE_SYNC_WAIT_S)  # master cache sync

        # -- main phase: 40 serial, fixed 7:3 interleave (28 family + 12 free).
        cont_n, free_n = 28, 12
        cont_addrs, free_addrs, failures = [], [], []
        for i in range(cont_n + free_n):
            rid = ops.next_request_id(base)
            if i % 10 < 7:  # 7 family : 3 free per decade
                addr, err = ops.run_one_request(
                    rid,
                    input_len=input_len,
                    output_len=2,
                    block_keys=family_keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"cont rid={rid}: {err}")
                else:
                    cont_addrs.append(addr)
            else:
                keys = [rid * 100 + j for j in range(16)]
                addr, err = ops.run_one_request(
                    rid,
                    input_len=input_len,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"free rid={rid}: {err}")
                else:
                    free_addrs.append(addr)

        holder_hits = sum(1 for a in cont_addrs if addr_map.get(a, a) == holder)
        stick_share = holder_hits / len(cont_addrs) if cont_addrs else 0.0
        free_on_other = sum(1 for a in free_addrs if addr_map.get(a, a) != holder)
        # M2 caliber: the holder's TOTAL share — seed + family continuations
        # + free requests scattered onto it by the tie window (token share ==
        # request share by uniform input_len).
        free_on_holder = len(free_addrs) - free_on_other
        holder_total = holder_hits + 1 + free_on_holder  # + seed
        total = 1 + len(cont_addrs) + len(free_addrs)
        holder_share = holder_total / total if total else 1.0
        holder_token_share = (
            holder_total * input_len / (total * input_len) if total else 1.0
        )

        report.invariant(
            "P6",
            not failures and len(cont_addrs) == cont_n and len(free_addrs) == free_n,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "P9",
            stick_share,
            context="hot_family",
            detail=(
                f"holder={holder}, hits={holder_hits}/{len(cont_addrs)}, "
                f"other={other_names}"
            ),
        )
        report.check(
            "M2",
            holder_share,
            context="holder_total_share",
            detail=(
                f"holder={holder}: {holder_total}/{total} requests "
                f"(token share {holder_token_share:.3f} — equal by uniform "
                f"input_len), free_on_other={free_on_other}/{len(free_addrs)}"
            ),
        )
        report.invariant(
            "P2",
            free_on_other >= 1,
            context="free_flow",
            detail=(
                f"free requests landing off-holder={free_on_other}/"
                f"{len(free_addrs)} (other engine must not be starved)"
            ),
        )
        return report.finish(
            f"holder={holder}, stick={holder_hits}/{len(cont_addrs)}, "
            f"holder_share={holder_share:.3f}, "
            f"free_off_holder={free_on_other}/{len(free_addrs)}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
