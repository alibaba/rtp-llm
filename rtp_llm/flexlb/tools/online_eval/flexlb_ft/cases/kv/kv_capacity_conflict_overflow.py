from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    CONFLICT_BLOCKS,
    CONFLICT_INPUT_LEN,
    CONFLICT_SEED_INPUT_LEN,
    STREAM_TIMEOUT_S,
    _drain_fired,
    _fam_keys,
    _fire_request,
    _kv_spec,
    _poll_engine_pending,
    _prefill_names,
    _wait_cache_sync,
)


@case(
    "kv_capacity_conflict_overflow",
    category="kv",
    source="kv family: affinity yields to a full holder ledger",
)
def kv_capacity_conflict_overflow(ctx: CaseContext):
    """[capacity] Affinity yields when the holder's ledger is full.

    Scenario: a 40-block family-0 is primed on engine e1 (40960-token
    seqLen keeps the seed's hit share at 27.8% >= the 20% gate even
    against a 147456-token seed); both prefills are slowed to 5s; the
    seed CARRIES the family-0 prefix so affinity pins it on e1 — the
    holder IS the hot engine (the balance_overload_avoid_prefill seed
    technique); the cool engine e2 is then restored while the seed's
    ~2.06s predicted ledger keeps e1 hot, and a 5-request same-prefix
    wave fires through the live ledger (two-phase, 0.12s spacing).
    Behaviour: the affinity-vs-capacity conflict — every wave request
    is full-hit on e1, but e1's projected TTFT sits ~2s above e2
    (>> maxExtraTtftMs).  Expected (contract): OVER_CAP overflow — the
    affinity gate steps aside and the wave spills onto the
    non-matching engine (hot_share < 1, P5); short requests stay
    protected (P7 vs the unloaded baseline) and NOTHING parks on a
    queue timeout (P6 — the routing decision returns immediately).
    Prediction: UNCERTAIN — both sides of the switch point are
    verified in isolation (balance_overload_avoid_prefill, the kv
    affinity cases) but were never exercised in the same frame; a
    failure is a finding.

    Construction note (2026-09-04): the prime (40960 tokens) and seed
    (147456 tokens) seq_lens both tower over the kv-family default
    decode pool (12 blocks = 12288 tokens), and the master's
    CostBasedDecodeStrategy.rejectIfPhysicalCapacityIsTooSmall turns
    any decode seq_len above the engine-reported totalKv into a typed
    StaticCapacityExceededException reject before routing even starts —
    observed fail: "Decode request seq_len=40960 exceeds max known
    physical KV=12288" (the case died on its FIRST request; the typed
    reject itself is correct engine-side behavior, the CONSTRUCTION
    was over-limit).  The case now pins its own env with a 180-block
    decode pool (184320 tokens >= the 144-block seed plus output
    margin) so every request's decode phase fits a single engine and
    the affinity-vs-capacity conflict — a PREFILL-side property — is
    the thing under test; decode-pool size plays no role in the
    prefill assertions (see _kv_spec), so this unblocks the
    construction without touching the semantics.

    Threshold note (2026-09-05): physical fit is only the first gate —
    the pre-dispatch permit also runs DecodeEndpoint.engineDispatchCapacityFits
    with maxKvUsagePercent, whose Java default is 90 (flexlb_ft never
    sets the key), against the master's GROSS demand (min(seqLen +
    maxNewTokens, totalKv), no prefix-hit deduction — the net-demand
    cut belongs to the engine-side prepare-stage ALLOCATE; master-side
    gross demand is a deliberate conservative design choice).  The
    seed's gross demand is 147458 tokens, 2 tokens past the old
    160-block pool's 90% × 163840 = 147456 gate (park to the 30s
    client deadline); 180 blocks put the gate at 90% × 180 × 1024 =
    165888 ≥ 147458, with margin.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx, "_conflict", decode_cache_blocks=180))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    is_batch = ctx.batch_dispatch()
    caliber = "completion_duration" if is_batch else "client_ttft"
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0, CONFLICT_BLOCKS)

        # -- prime the long family on its natural landing engine.
        rid_prime = ops.next_request_id(base)
        addr_e1, err = ops.run_one_request(
            rid_prime,
            input_len=CONFLICT_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"prime failed: {err}"
        e1 = ops.addr_to_name().get(addr_e1, addr_e1)
        e2 = next(n for n in names if n != e1)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after prime"

        # -- the seed carries the family prefix: affinity pins it on e1.
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=5000.0)
        time.sleep(1.5)  # master perf sync
        seed_keys = fam0 + [base + 9000 + j for j in range(104)]
        rid_seed = ops.next_request_id(base)
        seed_name, err = _fire_request(
            ops,
            rid_seed,
            fired,
            fired_handles,
            input_len=CONFLICT_SEED_INPUT_LEN,
            output_len=2,
            block_keys=seed_keys,
        )
        if err:
            return False, f"seed failed: {err}"
        if seed_name != e1:
            return False, (
                f"seed carried the holder prefix but landed on {seed_name} "
                f"instead of holder {e1} (affinity pin failed)"
            )
        if not _poll_engine_pending(ops, e1, 1):
            return False, f"seed never appeared on {e1}"

        # -- cool engine fast again; baseline anchors the P7 denominator.
        ops.set_perf(e2, prefill_fixed_ms=100.0)
        time.sleep(0.3)

        def timed_request(rid: int, **kwargs):
            t_send = time.monotonic()
            try:
                resp = ops.schedule(rid, **kwargs)
            except Exception as exc:
                return None, None, None, repr(exc)
            if resp.code != 200 or not resp.success:
                return None, None, None, f"schedule failed: {resp.error_message}"
            name = ops.addr_to_name().get(ops.role_addr(resp, "PREFILL"), "")
            input_pb = (
                None
                if resp.enqueued_by_master
                else ops.build_generate_input(rid, **kwargs)
            )
            try:
                handle = ops.start_stream(resp, rid, input_pb=input_pb)
            except Exception as exc:
                return name, None, None, f"stream failed to open: {exc!r}"
            ended = handle.wait_end(STREAM_TIMEOUT_S)
            snap = handle.snap
            ttft = snap.first_received_s - t_send if snap.first_received_s else None
            dur = snap.terminated_s - t_send if snap.terminated_s else None
            if not ended or snap.error or not snap.completed:
                return name, ttft, dur, (snap.error or "stream did not complete")
            return name, ttft, dur, None

        rid_base_line = ops.next_request_id(base)
        base_name, base_ttft, base_dur, base_err = timed_request(
            rid_base_line, output_len=2
        )
        if base_err:
            return False, f"baseline failed: {base_err}"

        # -- wave: 5 same-prefix requests fired back-to-back through the
        #    live seed ledger (two-phase — decisions first, then collect).
        wave = []
        for i in range(5):
            rid = ops.next_request_id(base)
            name, err = _fire_request(
                ops,
                rid,
                fired,
                fired_handles,
                input_len=CONFLICT_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
            )
            wave.append((rid, name, err))
            if err:
                report.invariant("P6", False, detail=f"wave fire failed: {err}")
            if i < 4:
                time.sleep(0.12)
        outcomes = {rid: (name, err) for rid, name, err in wave}
        _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}

        wave_names = [name for _, name, err in wave if err is None]
        failures = [f"rid={rid}: {err}" for rid, _, err in wave if err]
        hot_count = sum(1 for n in wave_names if n == e1)
        hot_share = hot_count / len(wave) if wave else 1.0
        overflow_ok = hot_share < 1.0

        report.invariant("P6", not failures, detail=f"failures={failures[:2]}")
        report.check(
            "P5",
            hot_share,
            context="capacity_conflict",
            detail=(
                f"hot=holder={e1}({hot_count}/" f"{max(len(wave_names), 1)}), cool={e2}"
            ),
        )
        # P7 dual caliber (profile-dependent measurement, one band
        # table).  The wave requests were drained without per-request
        # timing capture, so the protection caliber is measured by ONE
        # timed same-prefix probe through the SAME live seed ledger:
        # the wave's routing outcome already proved the overflow, the
        # probe proves the short-request protection (a parked request
        # would blow past every tier).
        metric_base = (base_dur if is_batch else base_ttft) or 0.0
        rid_probe = ops.next_request_id(base)
        probe_name, probe_ttft, probe_dur, probe_err = timed_request(
            rid_probe,
            input_len=CONFLICT_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
        )
        if probe_err:
            report.invariant(
                "P6", False, detail=f"protection probe failed: {probe_err}"
            )
            p7_value = float("inf")
            p7_detail = f"caliber={caliber}, probe failed: {probe_err}"
        else:
            probe_metric = (probe_dur if is_batch else probe_ttft) or 0.0
            if metric_base > 0 and probe_metric > 0:
                p7_value = probe_metric / metric_base
                p7_detail = (
                    f"caliber={caliber}, base={metric_base:.3f}s, "
                    f"probe={probe_metric:.3f}s, probe_landed={probe_name}"
                )
            else:
                p7_value = float("inf")
                p7_detail = f"caliber={caliber}, missing timing"
        report.check("P7", p7_value, context=caliber, detail=p7_detail)

        passed, detail, rep = report.finish(
            f"hot=holder={e1}, cool={e2}, grades: {report.summary()}"
        )
        return (
            passed and overflow_ok,
            f"overflow_ok={overflow_ok} (hot_share={hot_share:.2f} < 1), " f"{detail}",
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
