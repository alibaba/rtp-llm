from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    LACKMEM_KEYS_PER_REQUEST,
    LACKMEM_POOL_BLOCKS,
    STREAM_TIMEOUT_S,
    _await_tracked,
    _decode_names,
    _fire_tracked,
    _lack_mem_spec,
    _lease_keys,
    _master_http,
    _prefill_names,
)


@case(
    "admission_engine_kv_lack_mem_fast_reject",
    profiles=["batch-window", "single-batch"],
    requires=["enqueue_batch"],
    source="admission wave-3 B2: engine prefill KV block-pool gate (KV v2 BlockLease admission — 602 LACK_MEM synchronous fast reject, the non-waitable engine-side complement of the master KV squeeze in admission_slo_queue_deadline)",
    category="admission",
)
def admission_engine_kv_lack_mem_fast_reject(ctx: CaseContext):
    """Engine prefill KV block-pool gate: 602 LACK_MEM fast reject.

    Scenario: dedicated 1P+2D env with a 17-block prefill KV pool
    (EnvSpec prefill_cache_blocks=17; reserve = ceil(5% x 17) = 1 block
    — the KV v2 TOTAL_AND_AVAILABLE gate).  Every request carries 8
    per-request block_cache_keys with input_len=512 — the engine-side
    need caliber is the KEY COUNT (8 blocks), while the master-side KV
    gate compares the request's seqLen (512 tokens) against the
    engine-reported available tokens (>= 1 block = 1024 even at peak
    occupancy), so the master gate never intercepts: the ENGINE gate
    is the only admission edge in play.  prefill_fixed_ms=3000 holds
    batch #1 running while batch #2 queues; both leases are
    provisioned at enqueue (the KV v2 admission-lease semantics), so
    after two fires held_blocks = 16 of 17 — a snapshot poll proves
    the pool-full state BEFORE the probe fires.

    Behaviour: the third 8-block request fails acquireBlockLease at
    EnqueueBatch Phase-1.5 — the ack carries the per-request error
    code 602 (MALLOC_FAILED, never the master's 8431) with "LACK_MEM:
    insufficient KV cache blocks (need=8, avail=1, spb=1024)";
    DefaultBatchDispatcher wraps it as EngineRejectedException
    ("EnqueueBatch rejected request N: LACK_MEM: ...").  New-B
    channel: the master surfaces the reject SYNCHRONOUSLY ON THE
    Schedule RPC (code 8510, "Delivery failed: EnqueueBatch rejected
    request N error_code=602: LACK_MEM: ..." — the same synchronous
    RPC surface admission_engine_waiting_batch_cap_reject accepts);
    the legacy stream-terminal path is kept as the accepted alternate.
    Synchronous, typed, no park, no queue (the non-waitable complement
    of admission_slo_queue_deadline, where the SAME master KV surface
    parks because that squeeze is a WAIT condition).  The rejected
    request leaves no residue (lease acquisition rolled back,
    requestStates -> "rejected").

    Expected (contract): the probe terminates FAST (< 3s from fire,
    no park residence) with the LACK_MEM family in its rejection
    ("lack_mem" + "insufficient kv cache" + the "enqueuebatch
    rejected" wrapper), read from EITHER surface — the Schedule RPC
    reject (8510-wrapped, primary under new-B) or the stream terminal
    (legacy); the two occupants complete normally and their leases
    hand back to the LRU on completion (pool recovery — pure-LRU
    blocks count as available again); a fresh 8-block request on the
    recovered pool succeeds; the master inflight and engine ledgers
    drain clean and recovery holds.

    Prediction: expected to pass — the 602 ack surface is
    BlockPoolCapacityTest's master-visible contract (11 hash-channel
    blocks vs a 10-block pool) and the EngineRejectedException
    terminal path is the enqueue-ack fault family's.  The 17-block
    sizing makes the arithmetic exact: 8+8 admitted with the reserve
    to spare, the third 8-block need denied outright by the available
    check (8 > 1) — no borderline rounding.  Risk: the two occupancy
    fires coalescing into ONE batch — both leases still provision at
    enqueue, so the pool still saturates (only the "third request"
    ordinal shifts); the 0.4s inter-fire gap keeps them separate
    anyway.
    """
    env = ctx.env_manager.ensure(_lack_mem_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        # Occupants: batch #1 running (lease taken at enqueue), batch #2
        # queued (lease ALSO taken at enqueue — KV v2 admission leases).
        fire_errors = []
        for _ in range(2):
            rid = ops.next_request_id(base)
            err = _fire_tracked(
                ops,
                rid,
                fired,
                input_len=512,
                output_len=2,
                block_keys=_lease_keys(rid),
            )
            if err is not None:
                fire_errors.append((rid, err))
            time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire
        if fire_errors:
            return False, f"occupant fire failed: {fire_errors[:1]}"

        # Pool-full proof BEFORE the probe: held = 16 of 17 blocks.
        held_seen = 0

        def pool_full() -> bool:
            snap = ops.snapshot_by_name()
            nonlocal held_seen
            held_seen = max(
                held_seen,
                max(int(snap.get(n, {}).get("held_blocks", 0)) for n in names),
            )
            return all(
                int(snap.get(n, {}).get("held_blocks", 0))
                >= 2 * LACKMEM_KEYS_PER_REQUEST
                for n in names
            )

        pool_observed = wait_for(pool_full, 8.0, 0.1)
        if not pool_observed:
            return False, (
                f"occupancy leases never saturated the pool "
                f"(held_max={held_seen}/{LACKMEM_POOL_BLOCKS})"
            )

        # Probe: 602 LACK_MEM synchronous fast reject (no park, no
        # queue).  New-B channel: the master surfaces the EnqueueBatch
        # reject SYNCHRONOUSLY ON THE Schedule RPC (code 8510 wrapping
        # the 602 ack error) — accept EITHER surface, the RPC-level
        # typed reject (primary) or a successful fire whose stream then
        # terminates with the LACK_MEM family (legacy), the same
        # dual-surface probe admission_engine_waiting_batch_cap_reject
        # uses.
        rid3 = ops.next_request_id(base)
        r3_t0 = time.monotonic()
        r3_err = ""
        try:
            resp3 = ops.schedule(
                rid3, input_len=512, output_len=2, block_keys=_lease_keys(rid3)
            )
            if resp3.code != 200 or not resp3.success:
                r3_err = f"schedule failed ({resp3.code}): {resp3.error_message}"
            else:
                handle3 = ops.start_stream(resp3, rid3)
                ended3 = handle3.wait_end(10.0)
                r3_err = str(handle3.snap.error or "") if ended3 else "no terminal"
        except Exception as exc:
            r3_err = repr(exc)
        reject_latency = time.monotonic() - r3_t0
        err_low = r3_err.lower()
        rejected = (
            "lack_mem" in err_low
            and "insufficient kv cache" in err_low
            and "enqueuebatch rejected" in err_low
            and reject_latency < 3.0
        )

        # Occupants complete; their leases hand back to the LRU on
        # completion — the pool recovers (pure-LRU counts as available).
        outcomes = _await_tracked(fired, wait_s=45.0)
        occupant_ok = len(outcomes) >= 2 and all(
            ok and err is None for _, _, _, ok, err in outcomes[:2]
        )

        avail_seen = 0

        def pool_recovered() -> bool:
            snap = ops.snapshot_by_name()
            nonlocal avail_seen
            avail_seen = max(
                avail_seen,
                max(int(snap.get(n, {}).get("available_blocks", 0)) for n in names),
            )
            return all(
                int(snap.get(n, {}).get("available_blocks", 0))
                >= LACKMEM_KEYS_PER_REQUEST
                for n in names
            )

        recovered_seen = wait_for(pool_recovered, 10.0, 0.2)

        # Post-release probe: a fresh 8-block lease on the recovered pool.
        fired4: list = []
        rid4 = ops.next_request_id(base)
        fire_err4 = _fire_tracked(
            ops,
            rid4,
            fired4,
            input_len=512,
            output_len=2,
            block_keys=_lease_keys(rid4),
        )
        outcomes4 = _await_tracked(fired4, wait_s=STREAM_TIMEOUT_S)
        lease4_ok = (
            fire_err4 is None
            and len(outcomes4) == 1
            and outcomes4[0][3]
            and outcomes4[0][4] is None
        )

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, names + _decode_names(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            pool_observed
            and rejected
            and occupant_ok
            and recovered_seen
            and lease4_ok
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"pool_observed={pool_observed} "
            f"(held={held_seen}/{LACKMEM_POOL_BLOCKS}), "
            f"probe_rejected={rejected} "
            f"(latency={reject_latency:.2f}s, err={r3_err[:80]}), "
            f"occupants_completed={occupant_ok}, "
            f"pool_recovers={recovered_seen} "
            f"(avail={avail_seen}/{LACKMEM_POOL_BLOCKS}), "
            f"fresh_lease_succeeds={lease4_ok} (fire_err4={fire_err4}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
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
