"""Ledger-separated shared seeding, explicit per-engine gap/tail eviction, contiguous-prefix checks and five serial routing continuations."""

from ..case_config import output

METADATA = {
    "id": "cache_local_index",
    "description": "Ledger-separated shared seeding, explicit per-engine gap/tail eviction, "
    "contiguous-prefix checks and five serial routing continuations.",
    "category": "kv",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def continuity_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 2000},
        },
    )
    case.step("perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "seed_first",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 30,
            "consume": "deferred",
        },
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params={"requests": output("seed_first", "requests"), "phase": "scheduled"},
    )
    case.step(
        "seed_second",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("seed_second", "requests")},
    )
    case.step(
        "second_holder",
        "kv_landing",
        params={"requests": output("seed_second", "requests")},
    )
    case.step(
        "two_holders",
        "kv_distinct",
        params={
            "first": output("first_holder", "engine"),
            "second": output("second_holder", "engine"),
        },
    )
    case.step(
        "first_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "carve_gap",
        "kv_evict",
        params={"engine": output("first_holder", "engine"), "keys": [810001]},
    )
    case.step(
        "carve_tail",
        "kv_evict",
        params={"engine": output("second_holder", "engine"), "keys": [810008, 810009]},
    )
    case.step(
        "carve_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "gap_prefix",
        "kv_prefix_check",
        params={
            "snapshot": output("carve_quiet", "snapshot"),
            "engine": output("first_holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "expected": 1,
        },
    )
    case.step(
        "tail_prefix",
        "kv_prefix_check",
        params={
            "snapshot": output("carve_quiet", "snapshot"),
            "engine": output("second_holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "expected": 8,
        },
    )
    case.step(
        "continuation_0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "affinity",
        "kv_affinity_check",
        params={
            "requests": [
                output("continuation_0", "requests"),
                output("continuation_1", "requests"),
                output("continuation_2", "requests"),
                output("continuation_3", "requests"),
                output("continuation_4", "requests"),
            ],
            "holder": output("second_holder", "engine"),
            "min_samples": 5,
            "bands": {"strict": 0.8, "normal": 0.7, "loose": 0.6},
        },
    )
    case.step("cleanup", "teardown")


def continuity_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 2000},
        },
    )
    case.step("perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "seed_first",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 30,
            "consume": "immediate",
        },
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params={"requests": output("seed_first", "requests"), "phase": "scheduled"},
    )
    case.step(
        "seed_second",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("seed_second", "requests")},
    )
    case.step(
        "second_holder",
        "kv_landing",
        params={"requests": output("seed_second", "requests")},
    )
    case.step(
        "two_holders",
        "kv_distinct",
        params={
            "first": output("first_holder", "engine"),
            "second": output("second_holder", "engine"),
        },
    )
    case.step(
        "first_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "carve_gap",
        "kv_evict",
        params={"engine": output("first_holder", "engine"), "keys": [810001]},
    )
    case.step(
        "carve_tail",
        "kv_evict",
        params={"engine": output("second_holder", "engine"), "keys": [810008, 810009]},
    )
    case.step(
        "carve_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "gap_prefix",
        "kv_prefix_check",
        params={
            "snapshot": output("carve_quiet", "snapshot"),
            "engine": output("first_holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "expected": 1,
        },
    )
    case.step(
        "tail_prefix",
        "kv_prefix_check",
        params={
            "snapshot": output("carve_quiet", "snapshot"),
            "engine": output("second_holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "expected": 8,
        },
    )
    case.step(
        "continuation_0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "affinity",
        "kv_affinity_check",
        params={
            "requests": [
                output("continuation_0", "requests"),
                output("continuation_1", "requests"),
                output("continuation_2", "requests"),
                output("continuation_3", "requests"),
                output("continuation_4", "requests"),
            ],
            "holder": output("second_holder", "engine"),
            "min_samples": 5,
            "bands": {"strict": 0.8, "normal": 0.7, "loose": 0.6},
        },
    )
    case.step("cleanup", "teardown")


def evict_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prime",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "prime_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("prime", "requests")},
    )
    case.step(
        "holder",
        "kv_landing",
        params={"requests": output("prime", "requests"), "phase": "terminal"},
    )
    case.step(
        "prime_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "positive",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "positive_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("positive", "requests")},
    )
    case.step(
        "positive_holder",
        "kv_landing",
        params={"requests": output("positive", "requests"), "phase": "terminal"},
    )
    case.step(
        "positive_affinity",
        "kv_same",
        params={
            "first": output("holder", "engine"),
            "second": output("positive_holder", "engine"),
        },
    )
    case.step(
        "evict",
        "kv_evict",
        params={
            "engine": output("holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
        },
    )
    case.step(
        "eviction_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "eviction_membership",
        "kv_membership_check",
        params={
            "snapshot": output("eviction_quiet", "snapshot"),
            "engine": output("holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "relation": "none",
        },
    )
    case.step(
        "wave",
        "request",
        params={
            "count": 20,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "deferred",
            "stream_timeout_s": 30,
            "post_issue_delay_s": 0.12,
        },
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params={
            "requests": [output("wave", "requests")],
            "min_samples": 20,
            "bands": {"strict": 0.65, "normal": 0.75, "loose": 0.85},
        },
    )
    case.step("cleanup", "teardown")


def isolation_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 2000},
        },
    )
    case.step("perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "seed_a",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "a_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("seed_a", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "a_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("seed_a", "requests")},
    )
    case.step(
        "a_holder",
        "kv_landing",
        params={"requests": output("seed_a", "requests"), "phase": "terminal"},
    )
    case.step(
        "seed_b",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                815000,
                815001,
                815002,
                815003,
                815004,
                815005,
                815006,
                815007,
                815008,
                815009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "b_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("seed_b", "requests")},
    )
    case.step(
        "b_holder",
        "kv_landing",
        params={"requests": output("seed_b", "requests"), "phase": "terminal"},
    )
    case.step(
        "two_holders",
        "kv_distinct",
        params={
            "first": output("a_holder", "engine"),
            "second": output("b_holder", "engine"),
        },
    )
    case.step(
        "restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "slow_b",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("b_holder", "engine")],
            "perf": {"prefill_fixed_ms": 5000},
        },
    )
    case.step("b_perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "admit_0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_0", "requests")},
    )
    case.step(
        "admit_0_holder",
        "kv_landing",
        params={"requests": output("admit_0", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_0_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_0_holder", "engine"),
        },
    )
    case.step(
        "admit_1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_1", "requests")},
    )
    case.step(
        "admit_1_holder",
        "kv_landing",
        params={"requests": output("admit_1", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_1_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_1_holder", "engine"),
        },
    )
    case.step(
        "admit_2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_2", "requests")},
    )
    case.step(
        "admit_2_holder",
        "kv_landing",
        params={"requests": output("admit_2", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_2_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_2_holder", "engine"),
        },
    )
    case.step(
        "admit_3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_3", "requests")},
    )
    case.step(
        "admit_3_holder",
        "kv_landing",
        params={"requests": output("admit_3", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_3_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_3_holder", "engine"),
        },
    )
    case.step(
        "restore_b",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("b_holder", "engine")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "admission_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "a_membership",
        "kv_membership_check",
        params={
            "snapshot": output("admission_quiet", "snapshot"),
            "engine": output("a_holder", "engine"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "relation": "all",
        },
    )
    case.step(
        "b_isolation",
        "kv_membership_check",
        params={
            "snapshot": output("admission_quiet", "snapshot"),
            "engine": output("b_holder", "engine"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "relation": "none",
        },
    )
    case.step(
        "continuation_0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "fidelity",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_0", "requests"),
                output("continuation_1", "requests"),
                output("continuation_2", "requests"),
                output("continuation_3", "requests"),
                output("continuation_4", "requests"),
                output("continuation_5", "requests"),
                output("continuation_6", "requests"),
                output("continuation_7", "requests"),
                output("continuation_8", "requests"),
                output("continuation_9", "requests"),
            ],
            "holder": output("a_holder", "engine"),
            "min_samples": 10,
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step(
        "final_cache", "kv_snapshot", params={"targets": [output("b_holder", "engine")]}
    )
    case.step(
        "final_b_isolation",
        "kv_membership_check",
        params={
            "snapshot": output("final_cache", "snapshot"),
            "engine": output("b_holder", "engine"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "relation": "none",
        },
    )
    case.step("cleanup", "teardown")


def evict_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prime",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "prime_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("prime", "requests")},
    )
    case.step(
        "holder",
        "kv_landing",
        params={"requests": output("prime", "requests"), "phase": "terminal"},
    )
    case.step(
        "prime_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "positive",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "positive_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("positive", "requests")},
    )
    case.step(
        "positive_holder",
        "kv_landing",
        params={"requests": output("positive", "requests"), "phase": "terminal"},
    )
    case.step(
        "positive_affinity",
        "kv_same",
        params={
            "first": output("holder", "engine"),
            "second": output("positive_holder", "engine"),
        },
    )
    case.step(
        "evict",
        "kv_evict",
        params={
            "engine": output("holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
        },
    )
    case.step(
        "eviction_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "eviction_membership",
        "kv_membership_check",
        params={
            "snapshot": output("eviction_quiet", "snapshot"),
            "engine": output("holder", "engine"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "relation": "none",
        },
    )
    case.step(
        "wave",
        "request",
        params={
            "count": 20,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 30,
            "post_issue_delay_s": 0.12,
        },
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params={
            "requests": [output("wave", "requests")],
            "min_samples": 20,
            "bands": {"strict": 0.65, "normal": 0.75, "loose": 0.85},
        },
    )
    case.step("cleanup", "teardown")


def isolation_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 2000},
        },
    )
    case.step("perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "seed_a",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "a_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("seed_a", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "a_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("seed_a", "requests")},
    )
    case.step(
        "a_holder",
        "kv_landing",
        params={"requests": output("seed_a", "requests"), "phase": "terminal"},
    )
    case.step(
        "seed_b",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                815000,
                815001,
                815002,
                815003,
                815004,
                815005,
                815006,
                815007,
                815008,
                815009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "b_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("seed_b", "requests")},
    )
    case.step(
        "b_holder",
        "kv_landing",
        params={"requests": output("seed_b", "requests"), "phase": "terminal"},
    )
    case.step(
        "two_holders",
        "kv_distinct",
        params={
            "first": output("a_holder", "engine"),
            "second": output("b_holder", "engine"),
        },
    )
    case.step(
        "restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "slow_b",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("b_holder", "engine")],
            "perf": {"prefill_fixed_ms": 5000},
        },
    )
    case.step("b_perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "admit_0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_0", "requests")},
    )
    case.step(
        "admit_0_holder",
        "kv_landing",
        params={"requests": output("admit_0", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_0_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_0_holder", "engine"),
        },
    )
    case.step(
        "admit_1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_1", "requests")},
    )
    case.step(
        "admit_1_holder",
        "kv_landing",
        params={"requests": output("admit_1", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_1_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_1_holder", "engine"),
        },
    )
    case.step(
        "admit_2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_2", "requests")},
    )
    case.step(
        "admit_2_holder",
        "kv_landing",
        params={"requests": output("admit_2", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_2_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_2_holder", "engine"),
        },
    )
    case.step(
        "admit_3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "admit_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("admit_3", "requests")},
    )
    case.step(
        "admit_3_holder",
        "kv_landing",
        params={"requests": output("admit_3", "requests"), "phase": "terminal"},
    )
    case.step(
        "admit_3_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_3_holder", "engine"),
        },
    )
    case.step(
        "restore_b",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("b_holder", "engine")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "admission_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "a_membership",
        "kv_membership_check",
        params={
            "snapshot": output("admission_quiet", "snapshot"),
            "engine": output("a_holder", "engine"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "relation": "all",
        },
    )
    case.step(
        "b_isolation",
        "kv_membership_check",
        params={
            "snapshot": output("admission_quiet", "snapshot"),
            "engine": output("b_holder", "engine"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "relation": "none",
        },
    )
    case.step(
        "continuation_0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "consume": "immediate",
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "fidelity",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_0", "requests"),
                output("continuation_1", "requests"),
                output("continuation_2", "requests"),
                output("continuation_3", "requests"),
                output("continuation_4", "requests"),
                output("continuation_5", "requests"),
                output("continuation_6", "requests"),
                output("continuation_7", "requests"),
                output("continuation_8", "requests"),
                output("continuation_9", "requests"),
            ],
            "holder": output("a_holder", "engine"),
            "min_samples": 10,
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step(
        "final_cache", "kv_snapshot", params={"targets": [output("b_holder", "engine")]}
    )
    case.step(
        "final_b_isolation",
        "kv_membership_check",
        params={
            "snapshot": output("final_cache", "snapshot"),
            "engine": output("b_holder", "engine"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
                814000,
                814001,
                814002,
                814003,
                814004,
                814005,
                814006,
                814007,
                814008,
                814009,
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "relation": "none",
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "continuity_batch": {
        "build": continuity_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "continuity_nonbatch": {
        "build": continuity_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "evict_batch": {
        "build": evict_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "isolation_batch": {
        "build": isolation_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "evict_nonbatch": {
        "build": evict_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {"requires": []},
    },
    "isolation_nonbatch": {
        "build": isolation_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {"requires": []},
    },
}
