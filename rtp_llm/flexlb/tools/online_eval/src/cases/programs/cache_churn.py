"""Bounded LRU replay retry, affinity and capacity eviction."""

from cases.config import output


def lru_affinity(case):
    # Capacity 4 includes one reserved block. Three requested blocks fit;
    # sharing only the first key forces eviction of a cold non-prefix block.
    case.step("setup", "setup", timeout_s=case.value("lru_affinity.setup_timeout_s"))
    case.step(
        "prime",
        "request",
        params=case.value("lru_affinity.prime"),
    )
    case.step(
        "prime_done",
        "wait",
        timeout_s=case.value("lru_affinity.prime_done_timeout_s"),
        params={"requests": output("prime", "requests")},
    )
    case.step(
        "prime_holder", "kv_landing", params={"requests": output("prime", "requests")}
    )
    case.step(
        "prime_sync", "balance_pause", params=case.value("lru_affinity.prime_sync")
    )
    case.step(
        "replay",
        "request",
        params=case.value("lru_affinity.replay"),
    )
    case.step(
        "replay_done",
        "wait",
        timeout_s=case.value("lru_affinity.replay_done_timeout_s"),
        params={"requests": output("replay", "requests")},
    )
    case.step(
        "retry",
        "kv_retry_misdirected",
        timeout_s=case.value("lru_affinity.retry_timeout_s"),
        params=case.params(
            "lru_affinity.retry",
            {
                "anchor": output("prime", "requests"),
                "candidate": output("replay", "requests"),
            },
        ),
    )
    case.step(
        "replay_holder", "kv_landing", params={"requests": output("retry", "requests")}
    )
    case.step(
        "replay_affinity",
        "kv_same",
        params={
            "first": output("prime_holder", "engine"),
            "second": output("replay_holder", "engine"),
        },
    )
    case.step(
        "prime_snapshot",
        "kv_snapshot",
        timeout_s=case.value("lru_affinity.prime_snapshot_timeout_s"),
        params=case.value("lru_affinity.prime_snapshot"),
    )
    case.step(
        "prime_counters",
        "kv_cache_statistics",
        params={
            "snapshot": output("prime_snapshot", "snapshot"),
            "engine": output("prime_holder", "engine"),
        },
    )
    case.step(
        "prime_keys",
        "check",
        params=case.params(
            "lru_affinity.prime_keys", {"actual": output("prime_counters", "keys")}
        ),
    )
    case.step(
        "prime_evictions",
        "check",
        params=case.params(
            "lru_affinity.prime_evictions",
            {"actual": output("prime_counters", "evictions")},
        ),
    )
    case.step(
        "cold_blocks",
        "kv_capacity_observe",
        params=case.params(
            "lru_affinity.cold_blocks", {"targets": [output("prime_holder", "engine")]}
        ),
    )
    for field in ("referenced_blocks", "held_blocks", "key_count"):
        expected = case.value(f"lru_affinity.expected_cold.{field}")
        case.step(
            "cold_" + field,
            "kv_capacity_counter",
            params=case.params(
                "lru_affinity.step_26",
                {
                    "observations": [output("cold_blocks", "observation")],
                    "field": field,
                    "expected": expected,
                },
            ),
        )
    case.step(
        "pressure",
        "request",
        params=case.value("lru_affinity.pressure"),
    )
    case.step(
        "pressure_done",
        "wait",
        timeout_s=case.value("lru_affinity.pressure_done_timeout_s"),
        params={"requests": output("pressure", "requests")},
    )
    case.step(
        "pressure_settle",
        "balance_pause",
        params=case.value("lru_affinity.pressure_settle"),
    )
    case.step(
        "pressure_holder",
        "kv_landing",
        params={"requests": output("pressure", "requests")},
    )
    case.step(
        "pressure_snapshot",
        "kv_snapshot",
        timeout_s=case.value("lru_affinity.pressure_snapshot_timeout_s"),
        params=case.value("lru_affinity.pressure_snapshot"),
    )
    case.step(
        "pressure_counters",
        "kv_cache_statistics",
        params={
            "snapshot": output("pressure_snapshot", "snapshot"),
            "engine": output("pressure_holder", "engine"),
        },
    )
    case.step(
        "pressure_keys",
        "check",
        params=case.params(
            "lru_affinity.pressure_keys",
            {"actual": output("pressure_counters", "keys")},
        ),
    )
    case.step(
        "pressure_evictions",
        "check",
        params=case.params(
            "lru_affinity.pressure_evictions",
            {"actual": output("pressure_counters", "evictions")},
        ),
    )
    case.step(
        "prefix_affinity",
        "kv_same",
        params={
            "first": output("prime_holder", "engine"),
            "second": output("pressure_holder", "engine"),
        },
    )
    case.step("cleanup", "teardown")
