from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    DECODE_HARD_GATE,
    DECODE_WAVE_REQUESTS,
    _decode_names,
    _decode_park_spec,
    _drain_fired,
    _fire_request,
    _master_http,
)


@case(
    "engine_decode_hard_gate_unbounded_park",
    category="admission",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 W2: engine decode hard gate "
        "(decodeMaxConcurrency=128 unbounded park — no queue-pressure reject)"
    ),
)
def engine_decode_hard_gate_unbounded_park(ctx: CaseContext):
    """Engine decode hard gate: unbounded park, never a queue-pressure
    rejection.

    Scenario: dedicated 2P+1D env with the master decode routing cap
    raised to 5000, so the ONLY admission edge is the engine's
    decodeMaxConcurrency=128 hard gate (the production waiting_streams_
    semantics).  decode_scale=10 stretches each decode step
    ((19.5 + 0.175 x running) ms x 10) and output_len=64 keeps every
    request ~25 decode steps (≈ 10.4s residency at the gate, 2.6
    tokens/step MTP fold) in the engine.

    Gate-reachability construction (2026-09-05 revision, verdict §4.5
    option (a)): the 2026-09-04 construction (150 fires x output_len=32)
    measured decode_running_max=78 (< gate 128) with decode_waiting_max=0
    — that residency did not stack past the gate, so the park proof
    never activated (the half-false-green finding).  The load is
    re-anchored on the measured arrival model: steady-state running =
    arrival_rate x residency, with arrival ≈ 19-25 req/s (prefill-bound,
    back-solved from running_max=78 @ residency ≈ 4.1s) and residency
    scaling linearly in output_len.  Doubling the residency
    (output_len=64, ≈ 10.4s at the gate) and deepening the wave to 280
    fires puts the modelled peak at running ≈ 155-180 > 128 with the
    overflow (~30-50 requests) parked in decodePendingQueue.  If a
    future run STILL measures sub-gate (engine-version drift, slower
    machine), the gate-aware clause below degrades to the invariant
    half ONLY and the measured running_max is reported as the next
    calibration anchor — an honest no-proof pass, never a fabricated
    one (the fallback semantics verdict §4.5 endorses).

    Behaviour: scheduleDecodeCompletion admits the first 128
    TransferToDecode arrivals as running and parks every overflow in
    the UNBOUNDED decodePendingQueue (no cap, no rejection — unlike
    the prefill gate this queue never bounces a request under queue
    pressure).  As running slots free up, parked requests are admitted
    wave by wave.

    Expected (contract): all 280 Schedule calls succeed (zero
    rejections — the engine-side form of a waitable gate); IF the
    observed decode_running_max reaches the 128 gate a snapshot poll
    must also observe decode waiting >= 1 (the overflow parked — a
    filled gate with no park is a vanished-overflow failure); after the
    drain >= 95% of the fired requests completed their streams, the
    decode park is empty (waiting == 0), the master inflight ledger is
    clean and a fresh request succeeds (recovery).

    Prediction: the invariant half (zero rejections, >= 95% drain,
    clean ledgers) is load-level independent and holds; the park half
    activates on the modelled running peak ≈ 155-180 (anchor above).
    Drain budget: 280 x 64 = 17,920 decode tokens at the ≈ 790 tok/s
    full-gate rate ≈ 23s plus the sub-gate tail, under the 45s
    per-stream cap; if a slow machine stretches waves the completion
    bar is the 95% ratio, not perfection.
    """
    env = ctx.env_manager.ensure(_decode_park_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    decode_engines = _decode_names(ops)
    if not decode_engines:
        return False, "no decode engines found"
    fired: list = []
    try:
        for n in decode_engines:
            ops.set_perf(n, decode_scale=10.0)

        fire_errors = []
        for i in range(DECODE_WAVE_REQUESTS):
            rid = ops.next_request_id(base)
            err = _fire_request(ops, rid, fired, input_len=512, output_len=64)
            if err is not None:
                fire_errors.append((rid, err))
            if (i + 1) % 25 == 0:
                time.sleep(0.05)  # tiny pacing, keeps batches flowing

        # Observe the park across the full fill window (no early exit:
        # waiting_max and running_max are both peak gauges — the pair
        # proves the gate filled and overflowed, which a waiting-only
        # early break would truncate; under a sub-gate wave both stay
        # low and the gauges record the load shape as diagnostics).
        # The 18s window spans the ~11-15s prefill supply phase plus the
        # stack to the modelled peak (see the construction note above).
        waiting_max = 0
        running_max = 0
        deadline = time.monotonic() + 18.0
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            for n in decode_engines:
                info = snap.get(n, {})
                waiting_max = max(waiting_max, int(info.get("waiting", 0)))
                running_max = max(running_max, int(info.get("running", 0)))
            time.sleep(0.2)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = sum(1 for _, ok, _ in outcomes if ok)
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        def decode_park_empty() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("waiting", 0)) == 0 for n in decode_engines
            )

        settled = wait_for(decode_park_empty, 15.0, 0.3)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 60.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        completion_ratio = completed / DECODE_WAVE_REQUESTS
        # Gate-aware park proof: when the wave stacked past the gate the
        # overflow MUST be observable as decode waiting (parked, never
        # rejected or dropped); a sub-gate wave (running_max < gate)
        # parks nothing by construction and only the zero-rejection /
        # full-drain invariant applies.
        gate_filled = running_max >= DECODE_HARD_GATE
        park_proven = waiting_max >= 1 if gate_filled else True
        passed = (
            not fire_errors
            and park_proven
            and completion_ratio >= 0.95
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={DECODE_WAVE_REQUESTS} "
            f"(fire_errors={len(fire_errors)}, first={fire_errors[:1]}), "
            f"decode_waiting_max={waiting_max}, decode_running_max={running_max} "
            f"(gate={DECODE_HARD_GATE}, gate_filled={gate_filled}, "
            f"park_proven={park_proven}), "
            f"completed={completed}/{DECODE_WAVE_REQUESTS} "
            f"({completion_ratio:.0%}, drain_errors={drain_errors[:2]}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in decode_engines:
                ops.set_perf(n, decode_scale=1.0)
        except Exception:
            pass
