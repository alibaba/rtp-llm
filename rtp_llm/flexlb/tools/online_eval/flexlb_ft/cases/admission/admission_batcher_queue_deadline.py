from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils
from ...registry import case
from ...support.admission import (
    BQ_DEADLINE_ADMITTED,
    BQ_DEADLINE_MS,
    BQ_DEADLINE_OVERFLOW,
    BQ_DEADLINE_REQUESTS,
    _await_tracked,
    _batcher_queue_spec,
    _decode_names,
    _master_http,
    _ParkedSampler,
    _prefill_names,
)


@case(
    "admission_batcher_queue_deadline",
    profiles=["batch-window", "single-batch"],
    requires=["enqueue_batch"],
    source="admission wave-2 A5: batcher-queue gate deadline — the same BATCH_SLO_EXPIRED (8511) terminal as admission_slo_queue_deadline, with the trigger source moved from the KV gate to the batcher queue capacity gate (same code, different source — the deadline classification must stay uniform)",
    category="admission",
)
def admission_batcher_queue_deadline(ctx: CaseContext):
    """Batcher-queue gate under an SLO deadline: park, then typed 8511.

    Scenario: the A5 env (batcher queue capacity 2, dispatcher lease
    window 4, prefill_fixed_ms=3000) with
    scheduler.queueTimeoutMs=1500.  Eight requests are fired 0.15s apart
    — fires 1-4 reach the engine through the 4-seat lease window; fires
    5-6 fill the batcher queue to its capacity-2 ceiling; fires 7-8 hit
    the capacity gate (Blocked) and park with their Schedule RPC still
    open.

    Behaviour (new-B semantics): a QUEUED entry's request deadline
    detaches at DELIVERY CONFIRMATION — RequestSlot.
    confirmDeliveryForPublication clears requestDeadline when the
    EnqueueBatch delivery is confirmed (verdict §1.5: the code's
    detach point, not "queue acceptance" as this docstring previously
    claimed), so fires 5-6 survive past the old expiry boundary and
    complete once the leases release — the queue absorbs them (the old
    contract expected them to expire in place).
    The two PARKED fires (7-8) expire while they wait: the absolute
    expiration (admissionTimeMs + queueTimeoutMs) completes the
    still-open Schedule RPC synchronously with the typed
    BATCH_SLO_EXPIRED error (8511, "request deadline exceeded") — the
    same producer admission_slo_queue_deadline exercises from the KV
    gate.  Expired waiters are removed from the waiters and the
    scheduler ledger synchronously, so nothing dangles; the six
    admitted requests finish their 3s batches unmolested.

    Expected (contract) — the recomputed new-B terminal split
    (2026-09-04 measured: 6 complete + 2 typed): fires 1-6 fire
    successfully, open their streams and complete normally (4 lease
    seats + 2 queue seats); EXACTLY the 2 overflow fires reject on
    their Schedule RPC with the deadline error family ("deadline"/
    "expired"/"exhaust"/"8400"/"8511"/"8431" — the same assertion
    family as the KV-gate deadline case, asserting the classification
    uniformity), each within 1.0-5.0s of its fire, fast and typed;
    zero fire errors (a typed deadline reject is an expected terminal,
    not a fire failure); after the wave the master inflight ledger is
    clean and a fresh request on the relieved gate succeeds
    (recovery).  The master-side parked count is a HARD assertion
    aligned with the expiry window (verdict §3): a background sampler
    running DURING the fire must observe parked >= 1 inside the
    overflow fires' live window [first fire 7-8 call, last 8511
    reject] — fires 5-8 ride the master ledger while only the 4 lease
    holders show on the engine, so a sample inside that window proves
    the park+expiry sequence shares one ledger view.  The 2026-09-04
    run's parked_max=-1/0 readings were the post-fire observation-loop
    defect, not a dead discriminator.

    Prediction: measured contract — the 6+2 split is structural (the
    queue accepts fires 5-6 during the 1.2s fire window while the
    first engine terminal is >= 3s away, so the boundary is
    ordinal-stable); the parked fires 7-8 expire at ~2.4s, ~0.6s
    before the first lease release — no wake-up race; the in-flight
    sampler observes parked 2-4 across [~0.9s, ~3.0s], straddling the
    expiry window.
    """
    env = ctx.env_manager.ensure(
        _batcher_queue_spec(ctx, queue_timeout_ms=BQ_DEADLINE_MS)
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    decode_names = _decode_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    rpc_rejects: list = []  # form (1): deadline typed on the Schedule RPC
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        # Fire loop: fires 1-6 are admitted (4 lease seats + 2 queue
        # seats — fire + stream opened); fires 7-8 hit the capacity
        # gate and their Schedule RPC stays open until the deadline
        # expires it with the typed reject (recorded in rpc_rejects, an
        # expected terminal).  fire_errors records only real failures
        # (RPC exception / stream-open failure), never a typed deadline
        # reject.  The parked window lives DURING this loop (fires 5-8
        # ride the master ledger while fires 7-8 hold their RPCs open
        # up to 1.5s each), so the background sampler starts BEFORE the
        # first fire (verdict §3 — a post-fire observation loop
        # structurally misses the park window).
        sampler = _ParkedSampler(ops, names, decode_names).start()
        fire_errors = []
        try:
            for _ in range(BQ_DEADLINE_REQUESTS):
                rid = ops.next_request_id(base)
                t_call = time.monotonic()
                try:
                    resp = ops.schedule(rid, input_len=512, output_len=2)
                except Exception as exc:
                    fire_errors.append((rid, repr(exc)))
                    time.sleep(0.15)
                    continue
                if resp.code != 200 or not resp.success:
                    rpc_rejects.append(
                        (
                            rid,
                            t_call,
                            time.monotonic(),
                            resp.code,
                            str(resp.error_message),
                        )
                    )
                else:
                    try:
                        handle = ops.start_stream(resp, rid)
                    except Exception as exc:
                        fire_errors.append((rid, repr(exc)))
                        time.sleep(0.15)
                        continue
                    fired.append((rid, handle, t_call))
                time.sleep(0.15)  # parks the whole wave before any expiry
        finally:
            sampler.stop()

        outcomes = _await_tracked(fired, wait_s=30.0)
        # New-B terminal split: the 6 admitted fires (4 lease seats +
        # 2 queue seats) ALL complete — the queued entries' deadline
        # detached at delivery confirmation
        # (confirmDeliveryForPublication, verdict §1.5), so they drain
        # through the released leases instead of expiring.
        delivered_ok = len(outcomes) == BQ_DEADLINE_ADMITTED and all(
            ok and err is None for _, _, _, ok, err in outcomes
        )

        def _deadline_typed(text: str) -> bool:
            lowered = text.lower()
            return any(
                kw in lowered
                for kw in (
                    "deadline",
                    "expired",
                    "exhaust",
                    "8400",
                    "8511",
                    "8431",
                )
            )

        # The overflow fires reject ON THE SCHEDULE RPC (the Blocked
        # placement keeps the RPC open until the deadline expires it);
        # the old stream-terminal form belonged to in-queue expiry,
        # which no longer occurs under the new-B split.
        wave_ok = []
        wave_details = []
        for rid, t_call, t_end, code, msg in rpc_rejects:
            typed = code == 8511 or _deadline_typed(msg)
            in_window = 1.0 <= (t_end - t_call) <= 5.0
            wave_ok.append(typed and in_window)
            wave_details.append(f"rpc:{code}:{msg[:50]}@{t_end - t_call:.2f}s")
        all_deadline = (
            len(wave_ok) == BQ_DEADLINE_OVERFLOW
            and all(wave_ok)
            and len(outcomes) + len(rpc_rejects) == BQ_DEADLINE_REQUESTS
        )

        # Deadline death removes the queue/waiter/ledger entries; the
        # six admitted fires drain normally.  Relieve the gate and
        # verify a fresh request succeeds.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=100.0)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        # Park evidence (hard, verdict §3): at least one in-flight
        # sample observed parked >= 1 INSIDE the overflow fires' live
        # window — from the first fire 7-8 Schedule call to the last
        # 8511 reject — proving the park and its typed expiry share one
        # ledger view (fires 5-8 on the master ledger while only the 4
        # lease holders show on the engine).
        parked_max = sampler.max_parked
        if rpc_rejects:
            ov_lo = min(t_call for _, t_call, _, _, _ in rpc_rejects)
            ov_hi = max(t_end for _, _, t_end, _, _ in rpc_rejects)
            park_aligned = sampler.parked_in_window(ov_lo, ov_hi)
        else:
            ov_lo = ov_hi = 0.0
            park_aligned = False
        overflow_window_s = ov_hi - ov_lo if rpc_rejects else 0.0

        passed = (
            not fire_errors
            and delivered_ok
            and all_deadline
            and park_aligned
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={BQ_DEADLINE_REQUESTS} (fire_errors={fire_errors[:1]}), "
            f"terminal_split=completed:{len(outcomes)}"
            f"/deadline_rpc_reject:{len(rpc_rejects)} "
            f"(expect {BQ_DEADLINE_ADMITTED}+{BQ_DEADLINE_OVERFLOW}), "
            f"master_side_parked_max={parked_max} "
            f"({sampler.max_detail}, samples={len(sampler.samples)}, "
            f"http_failures={sampler.http_failures}, in-flight sampled "
            f"inside the {overflow_window_s:.2f}s overflow live window — "
            f"hard), "
            f"delivered_completed={delivered_ok}, "
            f"deadline_typed={all_deadline} (details={wave_details}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
