from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, EnvSpec, default_perf, wait_for
from ...registry import case
from ...support.kv import (
    SAT_BURST_N,
    SAT_DECODE_POOL_BLOCKS,
    SAT_FIRE_SPACING_S,
    SAT_INPUT_LEN,
    SAT_POOL_BLOCKS,
    SAT_PREFILL_MS,
    SAT_PROBE_BOUND_S,
    SAT_REQUEST_KEYS,
    SAT_SAMPLE_S,
    STREAM_TIMEOUT_S,
    _drain_fired,
    _engine_cache_keys,
    _fam_keys,
    _fire_request,
    _master_http,
    _pool_state,
    _prefill_names,
)


@case(
    "kv_pool_saturation_evict_reject_recover",
    category="kv",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
    source="KV v2 saturation: evict wave + typed 602 + bounded failures + recovery",
)
def kv_pool_saturation_evict_reject_recover(ctx: CaseContext):
    """P-pool saturation: capacity eviction, synchronous 602 rejection,
    bounded failure share and full recovery — one pool, four calibers.

    Env: 1P (one pool = unambiguous attribution) + 2D; the P pool is 27
    blocks (reserve = ceil(27 x 0.05) = 2) — sized so the 3-occupant wave
    fits WITH reserve headroom (3 x 8 + 2 <= 27), the degenerate 24-block
    tier 602'd its own third occupant.  input_len stays at 2048 (2 blocks
    of tokens): the P-side demand is priced by the 8 block_keys, while
    the master's ROUTE-time decode soft reservation is input-caliber and
    cumulative — at input 8192 the three occupants' token reservations
    (8194 x 3) sat within a hair of the widened D pool's 90% door and
    the probe parked master-side instead of ever hitting the P gate
    (remote evidence run 20260903_130256).  The decode pool is widened
    to 32 blocks so the wave's Phase-1.6 D reservations (2 blocks x 3
    + growth) never 602 decode-side before the prefill pool saturates.

    A1 eviction wave: 10 serial disjoint-key requests (8 keys each).
    From the 4th onward the accumulated key set (32 > 27) forces LRU-tail
    capacity eviction inside admit — eviction is the release valve, never
    a rejection: all 10 requests must complete, the evictions counter
    must grow, and the key set must stay capped at 27.

    A2 three-state conservation: every snapshot sample satisfies
    held + referenced + available == cache_blocks.  The case NEVER calls
    /cache_evict, so every eviction is attributable to capacity pressure
    alone.

    A3 saturation: prefill slowed to 3s, 3 disjoint-key occupants fired
    at 0.4s spacing hold all 24 leased P blocks (of the 27-block pool)
    inside one occupancy window.
    The 4th probe must be rejected SYNCHRONOUSLY by the Phase-1.5 gate —
    a fast (< 3s, no park) typed 602 whose client-visible error carries
    "EnqueueBatch rejected" + "LACK_MEM" + "insufficient KV cache blocks"
    (the master wraps the engine ack error verbatim).  Snapshot proof
    through the window: held peaked at >= 3 x 8 blocks (all three
    occupants' leases) and available dipped to <= reserve + 1.

    A4 bounded failure share: a follow-up burst fired inside the same
    window fails with the same typed fast 602 — the failure count is
    bounded below by the probe (the pool is full: >= 1 failure is
    constructed) and above by probe + burst size; no failure may be a
    park/timeout instead of the typed reject.  First-version caliber:
    invariant + explicit bounds, no grade band.

    A5 recovery: once the occupants drain, release != delete — the freed
    key set makes the pool available again (>= 8 blocks), a fresh request
    completes, and both the master ledger and the engine inflight tables
    drain clean.
    """
    env = ctx.env_manager.ensure(
        EnvSpec(
            label=f"kv_sat_{ctx.profile}",
            n_prefill=1,
            n_decode=2,
            perf=default_perf(),
            master_profile=ctx.profile,
            prefill_cache_blocks=SAT_POOL_BLOCKS,
            decode_cache_blocks=SAT_DECODE_POOL_BLOCKS,
        )
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if not names:
            return False, "no prefill workers found"
        pname = names[0]
        snap0 = ops.snapshot_by_name()
        evict_base = int(snap0.get(pname, {}).get("cache_evictions", 0))
        lack_base = int(snap0.get(pname, {}).get("lack_mem_rejects", 0))

        # -- A1 + A2: serial eviction wave with conservation sampling.
        conservation_ok = True
        conservation_bad = ""
        wave_ok = True
        wave_err = None
        for fam in range(10):
            rid = ops.next_request_id(base)
            _, err = ops.run_one_request(
                rid,
                input_len=SAT_INPUT_LEN,
                output_len=2,
                block_keys=_fam_keys(base, fam, SAT_REQUEST_KEYS),
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                wave_ok = False
                wave_err = f"fam{fam}: {err}"
                break
            blocks, held, ref, avail = _pool_state(ops, pname)
            if held + ref + avail != blocks:
                conservation_ok = False
                conservation_bad = f"fam{fam}: {held}+{ref}+{avail} != {blocks}"
        snap1 = ops.snapshot_by_name()
        evictions = int(snap1.get(pname, {}).get("cache_evictions", 0)) - evict_base
        key_cap_ok = len(_engine_cache_keys(ops, pname)) <= SAT_POOL_BLOCKS
        evict_ok = evictions >= 1

        # -- A3: saturation window — 3 occupants x 8 blocks pinned by a
        #    3s prefill, then the probe (and burst) hit the full pool
        #    BEFORE the first admit can free anything.
        ops.set_perf(pname, prefill_fixed_ms=SAT_PREFILL_MS)
        time.sleep(1.5)  # master perf sync
        occupant_base = base + 100_000
        occupant_errs = []
        for i in range(3):
            rid = ops.next_request_id(base)
            _, err = _fire_request(
                ops,
                rid,
                fired,
                fired_handles,
                input_len=SAT_INPUT_LEN,
                output_len=2,
                block_keys=_fam_keys(occupant_base, i, SAT_REQUEST_KEYS),
            )
            occupant_errs.append(err)
            if err is None:
                time.sleep(SAT_FIRE_SPACING_S)
        time.sleep(0.3)  # all three enqueues (and their leases) landed

        # Pin the saturation peak the moment the leases land: the probe
        # and burst round trips eat into the 3s prefill window, and once
        # the first occupant finishes prefill the peak is gone (remote
        # evidence run 20260903_130256: a post-burst sampling loop saw
        # held_peak=0 with all three occupants completing fine).
        reserve_blocks = 2  # ceil(27 x 0.05)
        held_peak = 0
        saw_floor = False
        wait_for(
            lambda: _pool_state(ops, pname)[1] >= 3 * SAT_REQUEST_KEYS,
            2.0,
            0.05,
        )
        blocks, held, ref, avail = _pool_state(ops, pname)
        held_peak = max(held_peak, held)
        if avail <= reserve_blocks + 1:
            saw_floor = True
        if held + ref + avail != blocks:
            conservation_ok = False
            conservation_bad = f"saturation-lead: {held}+{ref}+{avail} != {blocks}"

        # The probe: fast, typed, synchronous — no park, no retry.
        probe_rid = ops.next_request_id(base)
        probe_t0 = time.monotonic()
        _, probe_err = ops.run_one_request(
            probe_rid,
            input_len=SAT_INPUT_LEN,
            output_len=2,
            block_keys=_fam_keys(occupant_base, 90, SAT_REQUEST_KEYS),
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        probe_dur = time.monotonic() - probe_t0
        probe_text = probe_err or ""
        # "EnqueueBatch rejected" is the master BATCH dispatcher's error
        # wrapper; under NON_BATCH the same engine reject surfaces as a
        # stream onError (snap.error) — the substring family is
        # dispatcher-shared, only the prefix is batch-specific.
        probe_typed = (
            probe_err is not None
            and "lack_mem" in probe_text.lower()
            and "insufficient kv cache" in probe_text.lower()
            and (
                not ctx.batch_dispatch()
                or "enqueuebatch rejected" in probe_text.lower()
            )
        )
        probe_fast = probe_err is not None and probe_dur < SAT_PROBE_BOUND_S

        # -- A4: bounded failure share inside the same window.
        burst_failures = 0
        burst_typed_fast = True
        for i in range(SAT_BURST_N):
            rid = ops.next_request_id(base)
            t0 = time.monotonic()
            _, err = ops.run_one_request(
                rid,
                input_len=SAT_INPUT_LEN,
                output_len=2,
                block_keys=_fam_keys(occupant_base, 80 + i, SAT_REQUEST_KEYS),
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            dur = time.monotonic() - t0
            if err is not None:
                burst_failures += 1
                if dur >= SAT_PROBE_BOUND_S or "lack_mem" not in err.lower():
                    burst_typed_fast = False
        failures_total = (1 if probe_err else 0) + burst_failures
        share_ok = 1 <= failures_total <= 1 + SAT_BURST_N
        share = failures_total / (1 + SAT_BURST_N)

        # Saturation snapshot proof continues (0.1s sampling through
        # the window): the leading sample above pinned the peak; this
        # loop adds floor / conservation coverage through the drain.
        sat_deadline = time.monotonic() + SAT_PREFILL_MS / 1000.0
        while time.monotonic() < sat_deadline:
            blocks, held, ref, avail = _pool_state(ops, pname)
            held_peak = max(held_peak, held)
            if avail <= reserve_blocks + 1:
                saw_floor = True
            if held + ref + avail != blocks:
                conservation_ok = False
                conservation_bad = f"saturation: {held}+{ref}+{avail} != {blocks}"
            time.sleep(SAT_SAMPLE_S)
        # Structural: all three occupants' 8-block leases held at once
        # (24 of 27) — a pool-relative margin would drift with pool size.
        held_saturated = held_peak >= 3 * SAT_REQUEST_KEYS
        snap2 = ops.snapshot_by_name()
        lack_delta = int(snap2.get(pname, {}).get("lack_mem_rejects", 0)) - lack_base

        # -- A5: drain, recover, verify.
        outcomes = _drain_fired(ops, fired, fired_handles, wait_s=30.0)
        fired, fired_handles = [], {}
        occupants_ok = all(c for _, _, c, _ in outcomes) and not any(
            e for e in occupant_errs
        )
        avail_recovered = wait_for(
            lambda: _pool_state(ops, pname)[3] >= SAT_REQUEST_KEYS, 10.0, 0.5
        )
        rid_rec = ops.next_request_id(base)
        _, rec_err = ops.run_one_request(
            rid_rec,
            input_len=SAT_INPUT_LEN,
            output_len=2,
            block_keys=_fam_keys(base, 500, SAT_REQUEST_KEYS),
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, sorted(ops.snapshot_by_name().keys()), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            wave_ok
            and evict_ok
            and key_cap_ok
            and conservation_ok
            and probe_typed
            and probe_fast
            and held_saturated
            and saw_floor
            and lack_delta >= 1
            and share_ok
            and burst_typed_fast
            and occupants_ok
            and avail_recovered
            and rec_err is None
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"A1 wave_ok={wave_ok}({wave_err}), evictions={evictions}"
            f"(>=1), key_cap={key_cap_ok}(<= {SAT_POOL_BLOCKS}), "
            f"A2 conservation={conservation_ok}"
            f"({conservation_bad or 'identity held'}), "
            f"A3 probe_typed={probe_typed}, probe_fast={probe_fast}"
            f"({probe_dur:.2f}s, err={str(probe_err)[:120]!r}), "
            f"held_peak={held_peak}"
            f"(>= {3 * SAT_REQUEST_KEYS}), "
            f"available_floor={saw_floor}(<= {reserve_blocks + 1}), "
            f"lack_mem_rejects_delta={lack_delta}(>=1), "
            f"A4 failures={failures_total}/{1 + SAT_BURST_N}"
            f"(share={share:.0%}), burst_typed_fast={burst_typed_fast}, "
            f"A5 occupants_ok={occupants_ok}, "
            f"avail_recovered={avail_recovered}, "
            f"fresh_request_ok={rec_err is None}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg}"
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
