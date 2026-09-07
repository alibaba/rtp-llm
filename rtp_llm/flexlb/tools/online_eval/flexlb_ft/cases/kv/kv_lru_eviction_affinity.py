from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...registry import case
from ...support.kv import KV_CACHE_SYNC_WAIT_S, STREAM_TIMEOUT_S, _lru_spec


@case(
    "kv_lru_eviction_affinity",
    category="kv",
    source="gap G10: LRU prefix reuse + capacity eviction + affinity routing end-to-end",
)
def kv_lru(ctx: CaseContext):
    """Drive the mock's per-engine MockLruBlockCache end to end:

    1. R1 primes [k1,k2] on its landing engine X — snapshot proves
       cache_keys >= 2 with zero evictions.
    2. R2 replays the SAME keys: master-side cache-status sync must route
       it back to X (S2-style affinity, one retry for sync lag).
    3. R3 replays the prefix [k1,k2] plus three fresh keys: five keys
       admitted into the capacity-4 LRU evict exactly the eldest block —
       snapshot proves evictions >= 1 and cache_keys capped at 4, and the
       prefix hit keeps R3 on X as well.
    """
    env = ctx.env_manager.ensure(_lru_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")

    def run(rid, keys, input_len):
        return ops.run_one_request(
            rid,
            input_len=input_len,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )

    try:
        keys_a = [base + 1, base + 2]
        rid1 = ops.next_request_id(base)
        addr1, err1 = run(rid1, keys_a, 2048)
        if err1:
            return False, f"R1 (prime) failed: {err1}"
        time.sleep(KV_CACHE_SYNC_WAIT_S)

        # Affinity: same keys must return to the priming engine.
        rid2 = ops.next_request_id(base)
        addr2, err2 = run(rid2, keys_a, 2048)
        if err2:
            return False, f"R2 (replay) failed: {err2}"
        affinity = addr1 == addr2
        if not affinity:
            # S2-style retry: the cache-status sync may lag one poll.
            time.sleep(KV_CACHE_SYNC_WAIT_S)
            rid2b = ops.next_request_id(base)
            addr2b, err2b = run(rid2b, keys_a, 2048)
            if err2b:
                return False, f"R2 retry failed: {err2b}"
            affinity = addr1 == addr2b

        addr_map = ops.addr_to_name()
        engine_x = addr_map.get(addr1, "?")
        snap = ops.snapshot_by_name()
        keys_after_prime = snap.get(engine_x, {}).get("cache_keys", 0)
        evictions_after_prime = snap.get(engine_x, {}).get("cache_evictions", 0)

        # Capacity pressure: prefix [k1,k2] + 3 fresh keys -> 5 admits into
        # a capacity-4 LRU -> exactly the eldest block evicted.
        keys_ext = keys_a + [base + 3, base + 4, base + 5]
        rid3 = ops.next_request_id(base)
        addr3, err3 = run(rid3, keys_ext, 4096)
        if err3:
            return False, f"R3 (pressure) failed: {err3}"
        time.sleep(0.5)  # admit lands at prefill completion
        snap = ops.snapshot_by_name()
        engine_z = addr_map.get(addr3, "?")
        keys_after_pressure = snap.get(engine_z, {}).get("cache_keys", 0)
        evictions_after_pressure = snap.get(engine_z, {}).get("cache_evictions", 0)
        prefix_affinity = addr3 == addr1

        prime_ok = keys_after_prime >= 2 and evictions_after_prime == 0
        eviction_ok = evictions_after_pressure >= 1 and keys_after_pressure <= 4
        passed = affinity and prime_ok and eviction_ok and prefix_affinity
        return passed, (
            f"engine_x={engine_x}, affinity_r2={affinity}, "
            f"after_prime: keys={keys_after_prime}, evictions={evictions_after_prime}, "
            f"pressure_landed_on={engine_z}, prefix_affinity_r3={prefix_affinity}, "
            f"after_pressure: keys={keys_after_pressure} (<=4), "
            f"evictions={evictions_after_pressure} (>=1)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
