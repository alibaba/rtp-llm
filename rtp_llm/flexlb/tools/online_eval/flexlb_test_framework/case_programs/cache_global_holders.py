"""Shared holder sets, partial and full eviction, graceful holder removal, and interleaved admit/evict routing evidence."""

from ..case_config import output

METADATA = {
    "id": "cache_global_holders",
    "description": "Shared holder sets, partial and full eviction, graceful holder removal, and "
    "interleaved admit/evict routing evidence.",
    "category": "kv",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def shared_batch(case):
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
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
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
        "continuation_5",
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
        "continuation_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "continuation_10",
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
        "continuation_10_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_10", "requests")},
    )
    case.step(
        "continuation_11",
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
        "continuation_11_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_11", "requests")},
    )
    case.step(
        "continuation_12",
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
        "continuation_12_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_12", "requests")},
    )
    case.step(
        "continuation_13",
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
        "continuation_13_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_13", "requests")},
    )
    case.step(
        "continuation_14",
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
        "continuation_14_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_14", "requests")},
    )
    case.step(
        "continuation_15",
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
        "continuation_15_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_15", "requests")},
    )
    case.step(
        "continuation_16",
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
        "continuation_16_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_16", "requests")},
    )
    case.step(
        "continuation_17",
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
        "continuation_17_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_17", "requests")},
    )
    case.step(
        "continuation_18",
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
        "continuation_18_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_18", "requests")},
    )
    case.step(
        "continuation_19",
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
        "continuation_19_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_19", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
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
                output("continuation_10", "requests"),
                output("continuation_11", "requests"),
                output("continuation_12", "requests"),
                output("continuation_13", "requests"),
                output("continuation_14", "requests"),
                output("continuation_15", "requests"),
                output("continuation_16", "requests"),
                output("continuation_17", "requests"),
                output("continuation_18", "requests"),
                output("continuation_19", "requests"),
            ],
            "min_samples": 20,
            "bands": {"strict": 0.65, "normal": 0.75, "loose": 0.85},
        },
    )
    case.step(
        "holder_union",
        "kv_union_check",
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
                output("continuation_10", "requests"),
                output("continuation_11", "requests"),
                output("continuation_12", "requests"),
                output("continuation_13", "requests"),
                output("continuation_14", "requests"),
                output("continuation_15", "requests"),
                output("continuation_16", "requests"),
                output("continuation_17", "requests"),
                output("continuation_18", "requests"),
                output("continuation_19", "requests"),
            ],
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "min_samples": 20,
            "min_used": 2,
        },
    )
    case.step("cleanup", "teardown")


def shared_nonbatch(case):
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
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
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
        "continuation_5",
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
        "continuation_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "continuation_10",
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
        "continuation_10_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_10", "requests")},
    )
    case.step(
        "continuation_11",
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
        "continuation_11_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_11", "requests")},
    )
    case.step(
        "continuation_12",
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
        "continuation_12_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_12", "requests")},
    )
    case.step(
        "continuation_13",
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
        "continuation_13_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_13", "requests")},
    )
    case.step(
        "continuation_14",
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
        "continuation_14_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_14", "requests")},
    )
    case.step(
        "continuation_15",
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
        "continuation_15_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_15", "requests")},
    )
    case.step(
        "continuation_16",
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
        "continuation_16_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_16", "requests")},
    )
    case.step(
        "continuation_17",
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
        "continuation_17_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_17", "requests")},
    )
    case.step(
        "continuation_18",
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
        "continuation_18_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_18", "requests")},
    )
    case.step(
        "continuation_19",
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
        "continuation_19_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_19", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
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
                output("continuation_10", "requests"),
                output("continuation_11", "requests"),
                output("continuation_12", "requests"),
                output("continuation_13", "requests"),
                output("continuation_14", "requests"),
                output("continuation_15", "requests"),
                output("continuation_16", "requests"),
                output("continuation_17", "requests"),
                output("continuation_18", "requests"),
                output("continuation_19", "requests"),
            ],
            "min_samples": 20,
            "bands": {"strict": 0.65, "normal": 0.75, "loose": 0.85},
        },
    )
    case.step(
        "holder_union",
        "kv_union_check",
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
                output("continuation_10", "requests"),
                output("continuation_11", "requests"),
                output("continuation_12", "requests"),
                output("continuation_13", "requests"),
                output("continuation_14", "requests"),
                output("continuation_15", "requests"),
                output("continuation_16", "requests"),
                output("continuation_17", "requests"),
                output("continuation_18", "requests"),
                output("continuation_19", "requests"),
            ],
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "min_samples": 20,
            "min_used": 2,
        },
    )
    case.step("cleanup", "teardown")


def release_batch(case):
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
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
        },
    )
    case.step(
        "release_first",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "release_second",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "no_ghost",
        "kv_holders_check",
        params={
            "snapshot": output("released_quiet", "snapshot"),
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
            "holders": [],
            "match": "any_key",
        },
    )
    case.step(
        "wave",
        "request",
        timeout_s=90,
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
            "stream_timeout_s": 30,
            "consume": "deferred",
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


def release_nonbatch(case):
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
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
        },
    )
    case.step(
        "release_first",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "release_second",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "no_ghost",
        "kv_holders_check",
        params={
            "snapshot": output("released_quiet", "snapshot"),
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
            "holders": [],
            "match": "any_key",
        },
    )
    case.step(
        "wave",
        "request",
        timeout_s=90,
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
            "stream_timeout_s": 30,
            "consume": "immediate",
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


def redirect_batch(case):
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
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
        },
    )
    case.step(
        "release_first",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "first_empty",
        "kv_membership_check",
        params={
            "snapshot": output("released_quiet", "snapshot"),
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
            "relation": "none",
        },
    )
    case.step(
        "sole_holder",
        "kv_holders_check",
        params={
            "snapshot": output("released_quiet", "snapshot"),
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
            "holders": [output("second_holder", "engine")],
            "match": "full_family",
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
        "continuation_5",
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
        "continuation_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "redirect",
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
            "min_samples": 10,
            "holder": output("second_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step("cleanup", "teardown")


def redirect_nonbatch(case):
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
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
        },
    )
    case.step(
        "release_first",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "first_empty",
        "kv_membership_check",
        params={
            "snapshot": output("released_quiet", "snapshot"),
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
            "relation": "none",
        },
    )
    case.step(
        "sole_holder",
        "kv_holders_check",
        params={
            "snapshot": output("released_quiet", "snapshot"),
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
            "holders": [output("second_holder", "engine")],
            "match": "full_family",
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
        "continuation_5",
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
        "continuation_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "redirect",
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
            "min_samples": 10,
            "holder": output("second_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step("cleanup", "teardown")


def down_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
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
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1", "prefill-2"], "quiet_s": 3.5},
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
        },
    )
    case.step(
        "third_worker",
        "kv_snapshot",
        timeout_s=10,
        params={
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
            "quiet_s": 0,
            "exclude": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
        },
    )
    case.step(
        "third_empty",
        "kv_holders_check",
        params={
            "snapshot": output("third_worker", "snapshot"),
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
            "holders": [],
            "match": "any_key",
        },
    )
    case.step(
        "remove_holder",
        "elastic_remove",
        timeout_s=70,
        params={"engine": output("first_holder", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "master_converged",
        "kv_master_alive",
        timeout_s=30,
        params={"role": "PREFILL", "count": 2},
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
        "survivor_snapshot",
        "kv_snapshot",
        timeout_s=10,
        params={
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
            "quiet_s": 0,
            "exclude": [output("first_holder", "engine")],
        },
    )
    case.step(
        "survivor_kept",
        "kv_membership_check",
        params={
            "snapshot": output("survivor_snapshot", "snapshot"),
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
            "relation": "all",
        },
    )
    case.step(
        "survivor_fidelity",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_0", "requests"),
                output("continuation_1", "requests"),
                output("continuation_2", "requests"),
                output("continuation_3", "requests"),
                output("continuation_4", "requests"),
            ],
            "min_samples": 5,
            "holder": output("second_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step("cleanup", "teardown")


def down_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
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
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1", "prefill-2"], "quiet_s": 3.5},
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params={
            "snapshot": output("seed_quiet", "snapshot"),
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
            "holders": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
            "match": "full_family",
        },
    )
    case.step(
        "third_worker",
        "kv_snapshot",
        timeout_s=10,
        params={
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
            "quiet_s": 0,
            "exclude": [
                output("first_holder", "engine"),
                output("second_holder", "engine"),
            ],
        },
    )
    case.step(
        "third_empty",
        "kv_holders_check",
        params={
            "snapshot": output("third_worker", "snapshot"),
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
            "holders": [],
            "match": "any_key",
        },
    )
    case.step(
        "remove_holder",
        "elastic_remove",
        timeout_s=70,
        params={"engine": output("first_holder", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "master_converged",
        "kv_master_alive",
        timeout_s=30,
        params={"role": "PREFILL", "count": 2},
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
        "survivor_snapshot",
        "kv_snapshot",
        timeout_s=10,
        params={
            "targets": ["prefill-0", "prefill-1", "prefill-2"],
            "quiet_s": 0,
            "exclude": [output("first_holder", "engine")],
        },
    )
    case.step(
        "survivor_kept",
        "kv_membership_check",
        params={
            "snapshot": output("survivor_snapshot", "snapshot"),
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
            "relation": "all",
        },
    )
    case.step(
        "survivor_fidelity",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_0", "requests"),
                output("continuation_1", "requests"),
                output("continuation_2", "requests"),
                output("continuation_3", "requests"),
                output("continuation_4", "requests"),
            ],
            "min_samples": 5,
            "holder": output("second_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step("cleanup", "teardown")


def mixed_batch(case):
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
        "first_release",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "family_one_0",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "family_one_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("family_one_0", "requests")},
    )
    case.step(
        "family_one_holder",
        "kv_landing",
        params={"requests": output("family_one_0", "requests")},
    )
    case.step(
        "second_release",
        "kv_evict",
        params={
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
        },
    )
    case.step("family_two_fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "family_two_slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 2000},
        },
    )
    case.step("family_two_perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "family_two_seed_first",
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
            "stream_timeout_s": 30,
            "consume": "deferred",
        },
    )
    case.step(
        "family_two_first_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("family_two_seed_first", "requests"),
            "fleet": output("family_two_fleet", "snapshot"),
        },
    )
    case.step(
        "family_two_first_holder",
        "kv_landing",
        params={
            "requests": output("family_two_seed_first", "requests"),
            "phase": "scheduled",
        },
    )
    case.step(
        "family_two_seed_second",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "family_two_second_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("family_two_seed_second", "requests")},
    )
    case.step(
        "family_two_second_holder",
        "kv_landing",
        params={"requests": output("family_two_seed_second", "requests")},
    )
    case.step(
        "family_two_two_holders",
        "kv_distinct",
        params={
            "first": output("family_two_first_holder", "engine"),
            "second": output("family_two_second_holder", "engine"),
        },
    )
    case.step(
        "family_two_first_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("family_two_seed_first", "requests")},
    )
    case.step(
        "family_two_restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "family_two_release",
        "kv_evict",
        params={
            "engine": output("family_two_first_holder", "engine"),
            "keys": [
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
        },
    )
    case.step(
        "mixed_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "family_zero_empty",
        "kv_holders_check",
        params={
            "snapshot": output("mixed_quiet", "snapshot"),
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
            "holders": [],
            "match": "any_key",
        },
    )
    case.step(
        "family_one_sole",
        "kv_holders_check",
        params={
            "snapshot": output("mixed_quiet", "snapshot"),
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
            ],
            "holders": [output("family_one_holder", "engine")],
            "match": "full_family",
        },
    )
    case.step(
        "family_two_sole",
        "kv_holders_check",
        params={
            "snapshot": output("mixed_quiet", "snapshot"),
            "keys": [
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
            "holders": [output("family_two_second_holder", "engine")],
            "match": "full_family",
        },
    )
    case.step(
        "wave",
        "request",
        timeout_s=90,
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
            "stream_timeout_s": 30,
            "consume": "deferred",
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
    case.step(
        "continuation_f1_0",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_0", "requests")},
    )
    case.step(
        "continuation_f1_1",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_1", "requests")},
    )
    case.step(
        "continuation_f1_2",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_2", "requests")},
    )
    case.step(
        "continuation_f1_3",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_3", "requests")},
    )
    case.step(
        "continuation_f1_4",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_4", "requests")},
    )
    case.step(
        "fidelity_f1",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_f1_0", "requests"),
                output("continuation_f1_1", "requests"),
                output("continuation_f1_2", "requests"),
                output("continuation_f1_3", "requests"),
                output("continuation_f1_4", "requests"),
            ],
            "min_samples": 5,
            "holder": output("family_one_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step(
        "continuation_f2_0",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_0", "requests")},
    )
    case.step(
        "continuation_f2_1",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_1", "requests")},
    )
    case.step(
        "continuation_f2_2",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_2", "requests")},
    )
    case.step(
        "continuation_f2_3",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_3", "requests")},
    )
    case.step(
        "continuation_f2_4",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_4", "requests")},
    )
    case.step(
        "fidelity_f2",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_f2_0", "requests"),
                output("continuation_f2_1", "requests"),
                output("continuation_f2_2", "requests"),
                output("continuation_f2_3", "requests"),
                output("continuation_f2_4", "requests"),
            ],
            "min_samples": 5,
            "holder": output("family_two_second_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step("cleanup", "teardown")


def mixed_nonbatch(case):
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
        "first_release",
        "kv_evict",
        params={
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
        },
    )
    case.step(
        "family_one_0",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "family_one_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("family_one_0", "requests")},
    )
    case.step(
        "family_one_holder",
        "kv_landing",
        params={"requests": output("family_one_0", "requests")},
    )
    case.step(
        "second_release",
        "kv_evict",
        params={
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
        },
    )
    case.step("family_two_fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "family_two_slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 2000},
        },
    )
    case.step("family_two_perf_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "family_two_seed_first",
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
            "stream_timeout_s": 30,
            "consume": "immediate",
        },
    )
    case.step(
        "family_two_first_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("family_two_seed_first", "requests"),
            "fleet": output("family_two_fleet", "snapshot"),
        },
    )
    case.step(
        "family_two_first_holder",
        "kv_landing",
        params={
            "requests": output("family_two_seed_first", "requests"),
            "phase": "scheduled",
        },
    )
    case.step(
        "family_two_seed_second",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "family_two_second_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("family_two_seed_second", "requests")},
    )
    case.step(
        "family_two_second_holder",
        "kv_landing",
        params={"requests": output("family_two_seed_second", "requests")},
    )
    case.step(
        "family_two_two_holders",
        "kv_distinct",
        params={
            "first": output("family_two_first_holder", "engine"),
            "second": output("family_two_second_holder", "engine"),
        },
    )
    case.step(
        "family_two_first_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("family_two_seed_first", "requests")},
    )
    case.step(
        "family_two_restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "family_two_release",
        "kv_evict",
        params={
            "engine": output("family_two_first_holder", "engine"),
            "keys": [
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
        },
    )
    case.step(
        "mixed_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "family_zero_empty",
        "kv_holders_check",
        params={
            "snapshot": output("mixed_quiet", "snapshot"),
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
            "holders": [],
            "match": "any_key",
        },
    )
    case.step(
        "family_one_sole",
        "kv_holders_check",
        params={
            "snapshot": output("mixed_quiet", "snapshot"),
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
            ],
            "holders": [output("family_one_holder", "engine")],
            "match": "full_family",
        },
    )
    case.step(
        "family_two_sole",
        "kv_holders_check",
        params={
            "snapshot": output("mixed_quiet", "snapshot"),
            "keys": [
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
            "holders": [output("family_two_second_holder", "engine")],
            "match": "full_family",
        },
    )
    case.step(
        "wave",
        "request",
        timeout_s=90,
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
            "stream_timeout_s": 30,
            "consume": "immediate",
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
    case.step(
        "continuation_f1_0",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_0", "requests")},
    )
    case.step(
        "continuation_f1_1",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_1", "requests")},
    )
    case.step(
        "continuation_f1_2",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_2", "requests")},
    )
    case.step(
        "continuation_f1_3",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_3", "requests")},
    )
    case.step(
        "continuation_f1_4",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f1_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f1_4", "requests")},
    )
    case.step(
        "fidelity_f1",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_f1_0", "requests"),
                output("continuation_f1_1", "requests"),
                output("continuation_f1_2", "requests"),
                output("continuation_f1_3", "requests"),
                output("continuation_f1_4", "requests"),
            ],
            "min_samples": 5,
            "holder": output("family_one_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step(
        "continuation_f2_0",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_0", "requests")},
    )
    case.step(
        "continuation_f2_1",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_1", "requests")},
    )
    case.step(
        "continuation_f2_2",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_2", "requests")},
    )
    case.step(
        "continuation_f2_3",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_3", "requests")},
    )
    case.step(
        "continuation_f2_4",
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
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "continuation_f2_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("continuation_f2_4", "requests")},
    )
    case.step(
        "fidelity_f2",
        "kv_fidelity_check",
        params={
            "requests": [
                output("continuation_f2_0", "requests"),
                output("continuation_f2_1", "requests"),
                output("continuation_f2_2", "requests"),
                output("continuation_f2_3", "requests"),
                output("continuation_f2_4", "requests"),
            ],
            "min_samples": 5,
            "holder": output("family_two_second_holder", "engine"),
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "shared_batch": {
        "build": shared_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "shared_nonbatch": {
        "build": shared_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "release_batch": {
        "build": release_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "release_nonbatch": {
        "build": release_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "redirect_batch": {
        "build": redirect_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "redirect_nonbatch": {
        "build": redirect_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "down_batch": {
        "build": down_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "down_nonbatch": {
        "build": down_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "mixed_batch": {
        "build": mixed_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "mixed_nonbatch": {
        "build": mixed_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
}
