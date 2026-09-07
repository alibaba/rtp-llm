from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _cleanup_dynamic,
    _pump_until_accepted,
    _wait_master_alive,
    wait_for,
)
from ...registry import case
from ...support.engine_fault import (
    RECOVERY_EVICT_S,
    RECOVERY_KV_SYNC_S,
    RECOVERY_SETTLE_S,
    _created_generation_count,
    _engine_ip_port,
    _ensure_started,
    _master_http,
    _master_log_offset,
    _recovery_cache_evict,
    _recovery_cache_keys,
    _recovery_env,
    _retire_count,
)


@case(
    "engine_fault_recovery_kv_usage_reset",
    category="engine_fault",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E6: KV usage must restart from zero after a full restart",
)
def recovery_kv_usage_reset(ctx: CaseContext):
    """E6 — expected behaviour: when an engine restarts and its KV memory
    is lost, its self-reported KV usage must restart from ZERO — and the
    point of the case is what the MASTER then does with the fresh
    self-report (master linkage as verdict, engine reset as construction
    gate):

      * construction (gate, detail-only): the engine's self-report
        restarts from zero (kv_tokens_used == 0 with an empty LRU — a
        resumed old reading is a mock restart-fidelity defect, recorded
        in the detail without failing the master leg);
      * verdict, first-wave typed-clean: the first post-recovery
        requests must succeed WITHOUT a LACK_MEM rejection — the
        master's capacity view of the new generation is rebuilt from
        the zeroed self-report, so a lack_mem reject (or any error) on
        the first wave pins the master resuming the STALE pre-outage
        occupancy (master-side defect);
      * verdict, no blacklist: the recovered engine keeps receiving
        traffic (the master does not blacklist it behind the stale
        occupancy — the accepted counter grows);
      * the generation actually turned over (fresh full-resync signal).

    FINDING if it fails: KV capacity not reset across a restart — the old
    generation's occupancy leaks into the new one (master-side), or the
    mock's restart does not model memory loss (mock-side fidelity).
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        name = "prefill-0"
        ip = _engine_ip_port(ops, name)

        # Fresh LRU baseline: wipe whatever earlier cases left on this
        # engine so kv_tokens_used reads exactly the injected pressure.
        _recovery_cache_evict(ops, name, sorted(_recovery_cache_keys(ops, name)))
        time.sleep(RECOVERY_KV_SYNC_S)

        used_before = int(
            ops.snapshot_by_name().get(name, {}).get("kv_tokens_used", -1)
        )
        if used_before != 0:
            return False, (
                f"baseline not clean after evict (kv_tokens_used={used_before})"
            )

        # Inject a large occupancy (engine self-report channel).
        pressure = 4_000_000
        ops.set_kv_pressure(name, pressure)
        pressurized = wait_for(
            lambda: int(ops.snapshot_by_name().get(name, {}).get("kv_tokens_used", -1))
            >= pressure,
            8.0,
            0.5,
        )
        if not pressurized:
            return False, "set_kv_pressure never surfaced in /snapshot"

        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Full-restart outage: stop → retire → start (memory lost).
        ops.stop_engine(name)
        retired = wait_for(
            lambda: _retire_count(env, ip, log_offset) > 0, RECOVERY_EVICT_S, 0.2
        )
        ops.start_engine(name)
        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)
        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # The engine's self-report must restart from zero (memory lost)
        # — CONSTRUCTION GATE only (detail), read BEFORE the first wave
        # occupies fresh KV blocks.
        used_after = int(ops.snapshot_by_name().get(name, {}).get("kv_tokens_used", -1))
        reset_ok = used_after == 0

        # Master verdict leg — first-wave typed-clean: the first
        # post-recovery requests must succeed with ZERO LACK_MEM
        # rejections.  The fresh generation's capacity view is rebuilt
        # from the zeroed self-report; a lack_mem on the first wave (or
        # any request error) pins the master resuming the STALE
        # pre-outage occupancy instead.
        def _lack_mem_sum() -> int:
            snap = ops.snapshot_by_name()
            return sum(
                int(info.get("lack_mem_rejects", 0))
                for n, info in snap.items()
                if "prefill" in n
            )

        lack_base = _lack_mem_sum()
        wave_errs = []
        for _ in range(3):
            rid = ops.next_request_id(base)
            _, err = ops.run_one_request(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=10.0,
            )
            if err is not None:
                wave_errs.append(str(err)[:60])
        lack_delta = _lack_mem_sum() - lack_base
        first_wave_clean = not wave_errs and lack_delta == 0

        # Traffic-restoration verdict: the master must not keep the
        # engine blacklisted behind the stale occupancy — fresh traffic
        # still lands on it.
        pumps_ok = _pump_until_accepted(ops, name, base, 15.0)

        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            retired
            and alive_back
            and generation_bumped
            and first_wave_clean
            and pumps_ok
            and recovery_ok
        )
        return passed, (
            f"ip={ip}, created_generations={created_before}->{created_after}, "
            f"retired={retired}, alive_restored={alive_back}, "
            f"kv_tokens_used={pressure}(injected)->{used_after} "
            f"(reset gate: {reset_ok}, need 0), "
            f"first_wave_typed_clean={first_wave_clean} "
            f"(errs={wave_errs[:1]}, lack_mem_delta={lack_delta}), "
            f"post_recovery_traffic={pumps_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _ensure_started(ops, ["prefill-0", "prefill-1"])
        try:
            ops.set_kv_pressure("prefill-0", 0)
        except Exception:
            pass
