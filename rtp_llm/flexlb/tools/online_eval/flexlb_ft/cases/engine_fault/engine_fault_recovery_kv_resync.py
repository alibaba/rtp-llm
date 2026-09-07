from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _cleanup_dynamic,
    _wait_master_alive,
    wait_for,
)
from ...registry import case
from ...support.engine_fault import (
    RECOVERY_EVICT_S,
    RECOVERY_KV_SYNC_S,
    RECOVERY_SETTLE_S,
    STREAM_TIMEOUT_S,
    _created_generation_count,
    _engine_ip_port,
    _ensure_started,
    _master_http,
    _master_log_offset,
    _prefill_names,
    _recovery_cache_evict,
    _recovery_cache_keys,
    _recovery_env,
    _retire_count,
)


@case(
    "engine_fault_recovery_kv_resync",
    category="engine_fault",
    profiles=["batch-window", "single-batch", "window-nonbatch"],
    # routing-shape bars calibrated on bw; sb/wn smoke-verified as-is.
    # single-nonbatch DEFERRED: 6 sampled sn runs swung between <=3/5,
    # 4/5 and 5/5 holder-stick (3 FAIL / 3 PASS, holder switching sides)
    # — the spread bar cannot separate the routing variance from the
    # 5/5 stale-baseline FINDING fingerprint on the SINGLE decision axis,
    # so the sn registration is parked until that variance is understood.
    source="E2: recovery must rebuild the cache view from a full snapshot",
)
def recovery_kv_resync(ctx: CaseContext):
    """E2 — expected behaviour: on engine recovery the master must rebuild
    the engine's cache view from a FULL snapshot, never from the old
    generation's incremental baseline, in BOTH memory regimes:

      * A (memory intact): the engine keeps its LRU through the outage —
        after recovery the holder relationship must survive, so same-prefix
        requests keep landing on the recovered engine (>= 4/5);
      * B (memory lost): the engine's key set is wiped across the restart
        (cache_evict models the reboot) — the rebuilt view must reflect the
        EMPTY key set, so same-prefix requests spread instead of sticking
        to the old holder (<= 3/5).

    Regime A also pins the generation bump: the retire
    (WorkerGenerationRetirement) clears the address-keyed cache index
    (removeEngineBlockCache) and the new generation re-pulls the full key
    set from version -1 — a master that instead kept the old version
    baseline would reject the (unchanged) engine version and LOSE the
    holder relationship (spread), failing the regime-A gate.

    FINDING if it fails: incremental-baseline resync across a generation
    boundary (stale holder or lost holder).
    """
    # Own env (see _recovery_spec): the regime-B spread bar is a routing
    # shape, and a shared env's earlier retire storms bias it.
    env, ops = _recovery_env(ctx, "_e2")
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill engines"

        # A 10-block prefix family (kv.py caliber: 10 x 1024 tokens keeps a
        # full-hit continuation past the affinity line, 9216 >= 8192).
        fam = [base + 900_000 + j for j in range(10)]

        # Seed: one request admits the family onto its landing engine X.
        rid = ops.next_request_id(base)
        addr, err = ops.run_one_request(
            rid,
            input_len=10_240,
            output_len=2,
            block_keys=fam,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"seed request failed: {err}"
        holder = ops.addr_to_name().get(addr, addr)
        other = [n for n in names if n != holder][0]

        # Holder must actually own the family before the outage.
        owner_ok = wait_for(
            lambda: set(fam) <= _recovery_cache_keys(ops, holder),
            8.0,
            0.5,
        )
        if not owner_ok:
            return False, (
                f"seed never admitted the family onto {holder} "
                f"(keys={sorted(_recovery_cache_keys(ops, holder))[:5]}...)"
            )
        # Master-side convergence (>= 3.5s quiet — kv.py caliber).
        time.sleep(RECOVERY_KV_SYNC_S)

        ip = _engine_ip_port(ops, holder)
        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Outage + recovery (mock keeps the LRU: memory-intact regime).
        ops.stop_engine(holder)
        retired = wait_for(
            lambda: _retire_count(env, ip, log_offset) > 0, RECOVERY_EVICT_S, 0.2
        )
        ops.start_engine(holder)
        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)
        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # Regime A: after the full rebuild, the holder relationship must
        # survive — 5 same-prefix requests, >= 4 must land on the holder.
        landings_a = []
        for _ in range(5):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=10_240,
                output_len=2,
                block_keys=fam,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                landings_a.append(f"ERR:{str(err)[:40]}")
            else:
                landings_a.append(ops.addr_to_name().get(addr, addr))
        hits_a = sum(1 for x in landings_a if x == holder)
        regime_a_ok = hits_a >= 4

        # Regime B: wipe the engine's key set (reboot semantics) and let
        # the master's view converge — same-prefix requests must now
        # spread; sticking to the wiped holder means the master kept a
        # stale view.
        _recovery_cache_evict(ops, holder, fam)
        owner_gone = wait_for(
            lambda: not (set(fam) & _recovery_cache_keys(ops, holder)),
            8.0,
            0.5,
        )
        time.sleep(RECOVERY_KV_SYNC_S)
        landings_b = []
        for _ in range(5):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=10_240,
                output_len=2,
                block_keys=fam,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                landings_b.append(f"ERR:{str(err)[:40]}")
            else:
                landings_b.append(ops.addr_to_name().get(addr, addr))
        hits_b = sum(1 for x in landings_b if x == holder)
        regime_b_ok = hits_b <= 3

        passed = (
            retired
            and alive_back
            and generation_bumped
            and owner_gone
            and regime_a_ok
            and regime_b_ok
        )
        return passed, (
            f"holder={holder}(other={other}), ip={ip}, "
            f"created_generations={created_before}->{created_after}, "
            f"retired={retired}, alive_restored={alive_back}, "
            f"regime_A_stick={hits_a}/5 (need >=4), "
            f"regime_B_spread={hits_b}/5 (need <=3, wipe_ok={owner_gone}), "
            f"landings_A={landings_a}, landings_B={landings_b}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _ensure_started(ops, ["prefill-0", "prefill-1"])
