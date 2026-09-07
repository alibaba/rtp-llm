from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, EnvSpec, default_perf, wait_for
from ...registry import case
from ...support.kv import (
    DSAT_DECODE_POOL_BLOCKS,
    DSAT_INPUT_LEN,
    DSAT_PROBE_BOUND_S,
    DSAT_RECOVER_INPUT_LEN,
    STREAM_TIMEOUT_S,
    _fam_keys,
    _master_http,
    _prefill_names,
)


@case(
    "kv_decode_pool_exhaustion_terminal",
    category="kv",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
    source="KV v2 decode-exhaustion: D-pool reservation 602 + clean recovery",
)
def kv_decode_pool_exhaustion_terminal(ctx: CaseContext):
    """D-pool exhaustion: the P-enqueue decode reservation answers with
    a synchronous decode-side 602, the P lease releases without residue,
    and one block of headroom restores service.

    Env: 2P + 1D — a single decode engine pins the exhaustion target
    (with more D engines the KV-weighted router could detour around the
    exhausted one, which would make this a routing caliber, not an
    exhaustion one); the D pool is 3 blocks (reserve = ceil(3 x 0.05) =
    1).  A probe with net decode demand ceil(2560/1024) = 3 blocks hits
    the TOTAL_AND_AVAILABLE gate at avail - need = 3 - 3 = 0 < 1 and is
    rejected SYNCHRONOUSLY at the prefill's EnqueueBatch Phase 1.6 — the
    mock counterpart of production's prepare-stage ALLOCATE RPC.  The
    2560 input is the FALLBACK caliber (empirically switched, remote run
    20260903_130256): the primary tier (D pool 2, input 2048) parked
    master-side — the route-time soft reservation prices input+output
    TOKENS against a 90% door and never let the probe reach the engine
    (30s DEADLINE, engine enqueue_rpcs == 0); 2560 + output headroom
    passes that token door while the block-caliber demand still
    saturates the pool.

    PERMANENT classification (wave-3 v1 adaptation): need + reserve =
    3 + 1 > 3 — no pool state can ever admit this request, so the
    reject lands in the permanent family: the engine counts it into
    lack_mem_rejects, not kv_admission_fails.  The RETRYABLE
    occupant-based tier tried first (D pool 4, a stretched-prefill
    request holding 2 blocks) never reached the engine at all — the
    occupant occupied the single decode worker's concurrency slot
    (decode concurrency is a production-locked one per engine on this
    line), so the probe was rejected at ROUTE time with
    NO_DECODE_WORKER(8403) before any admission arithmetic ran.  With
    in-flight occupancy structurally unavailable, the case pins the
    deterministic, stack-independent contract instead: a request the
    pool cannot structurally fit counts into lack_mem_rejects
    (upstream carries the same bug, same fix).

    DOCUMENTED DIVERGENCE (flexlb_ft README:437-440): production retries
    the decode KV allocation decode_retry_times times before failing;
    the mock answers terminal FINAL on the first exhaustion — this case
    asserts the mock's documented single-shot terminal, not the
    production retry ladder.

    Assertions:
      * the probe fails FAST (< 3s — no park: the QUEUE scheduler's
        wait-condition path applies to ROUTE-time delivery capacity,
        not to an engine-side reservation reject) with a typed error
        carrying "EnqueueBatch rejected" + "LACK_MEM" + "decode-side"
        (the master wraps the engine ack error verbatim);
      * counter split: the target D engine's lack_mem_rejects grew by
        EXACTLY 1 (permanent-family accounting for the structurally
        unfittable request) while its kv_admission_fails stayed flat
        (the retryable family never saw it); every P engine's
        lack_mem_rejects stayed at its pre-probe value (the P pool was
        never the constraint) and the P held blocks fell back to the
        pre-probe watermark (the rejection branch releases the P lease
        — no residue);
      * recovery: one block of headroom (net demand 1, avail 3 -
        reserve 1 = 2) completes a fresh request and both D counters
        stop moving;
      * the master ledger and engine inflight tables drain clean.
    """
    env = ctx.env_manager.ensure(
        EnvSpec(
            label=f"kv_dsat_{ctx.profile}",
            n_prefill=2,
            n_decode=1,
            perf=default_perf(),
            master_profile=ctx.profile,
            decode_cache_blocks=DSAT_DECODE_POOL_BLOCKS,
        )
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    try:
        snap0 = ops.snapshot_by_name()
        decode_names = sorted(n for n, e in snap0.items() if e.get("role") == "decode")
        if not decode_names:
            return False, "no decode workers found"
        dname = decode_names[0]
        pnames = _prefill_names(ops)
        kv_fails_base = int(snap0.get(dname, {}).get("kv_admission_fails", 0))
        d_lack_base = int(snap0.get(dname, {}).get("lack_mem_rejects", 0))
        lack_base = {
            p: int(snap0.get(p, {}).get("lack_mem_rejects", 0)) for p in pnames
        }
        held_base = {p: int(snap0.get(p, {}).get("held_blocks", 0)) for p in pnames}

        # -- the probe: fast typed decode-side 602, no park.
        rid = ops.next_request_id(base)
        t0 = time.monotonic()
        _, err = ops.run_one_request(
            rid,
            input_len=DSAT_INPUT_LEN,
            output_len=2,
            block_keys=_fam_keys(base, 900, 2),
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        probe_dur = time.monotonic() - t0
        probe_text = err or ""
        # Per-dispatcher probe typing: "EnqueueBatch rejected" is the
        # master BATCH dispatcher's wrapper; under NON_BATCH the same
        # engine reject surfaces as a stream onError (snap.error) — only
        # the substring family is dispatcher-shared.
        probe_typed = (
            err is not None
            and "lack_mem" in probe_text.lower()
            and "decode-side" in probe_text.lower()
            and (
                not ctx.batch_dispatch()
                or "enqueuebatch rejected" in probe_text.lower()
            )
        )
        probe_fast = err is not None and probe_dur < DSAT_PROBE_BOUND_S

        # -- counter split + P-lease release (poll: the reject branch
        #    releases asynchronously with the ack).  PERMANENT family:
        #    the structurally unfittable request counts into
        #    lack_mem_rejects EXACTLY once, while the retryable-family
        #    counter kv_admission_fails never sees it.
        d_lack_grew = wait_for(
            lambda: int(
                ops.snapshot_by_name().get(dname, {}).get("lack_mem_rejects", 0)
            )
            == d_lack_base + 1,
            10.0,
            0.5,
        )
        time.sleep(0.5)
        snap1 = ops.snapshot_by_name()
        kv_fails_flat = (
            int(snap1.get(dname, {}).get("kv_admission_fails", 0)) == kv_fails_base
        )
        lack_clean = all(
            int(snap1.get(p, {}).get("lack_mem_rejects", 0)) == lack_base[p]
            for p in pnames
        )
        held_released = wait_for(
            lambda: all(
                int(ops.snapshot_by_name().get(p, {}).get("held_blocks", 0))
                == held_base[p]
                for p in pnames
            ),
            10.0,
            0.5,
        )

        # -- recovery: one block of headroom, both D counters stop.
        d_lack_after = int(
            ops.snapshot_by_name().get(dname, {}).get("lack_mem_rejects", 0)
        )
        rid_rec = ops.next_request_id(base)
        _, rec_err = ops.run_one_request(
            rid_rec,
            input_len=DSAT_RECOVER_INPUT_LEN,
            output_len=2,
            block_keys=_fam_keys(base, 910, 2),
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        time.sleep(0.5)
        counters_stable = (
            int(ops.snapshot_by_name().get(dname, {}).get("lack_mem_rejects", 0))
            == d_lack_after
            and int(ops.snapshot_by_name().get(dname, {}).get("kv_admission_fails", 0))
            == kv_fails_base
        )

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, sorted(ops.snapshot_by_name().keys()), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            probe_typed
            and probe_fast
            and d_lack_grew
            and kv_fails_flat
            and lack_clean
            and held_released
            and rec_err is None
            and counters_stable
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"probe: typed={probe_typed}, fast={probe_fast}"
            f"({probe_dur:.2f}s), err={str(probe_text)[:90]}, "
            f"d_lack_mem_rejects_grew={d_lack_grew}(+1), "
            f"d_kv_admission_fails_flat={kv_fails_flat}, "
            f"p_lack_mem_clean={lack_clean}, "
            f"p_held_released={held_released}, "
            f"recovered={rec_err is None}, "
            f"counters_stable={counters_stable}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
