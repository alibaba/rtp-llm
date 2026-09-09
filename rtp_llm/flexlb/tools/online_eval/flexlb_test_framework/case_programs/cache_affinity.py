"""Prefix fidelity and free-flow diversity, hot holder total-share tension, and full/half/zero-hit tier contrast."""

from ..case_config import output

METADATA = {
    "id": "cache_affinity",
    "description": "Prefix fidelity and free-flow diversity, hot holder total-share tension, and "
    "full/half/zero-hit tier contrast.",
    "category": "kv",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def prefix_batch(case):
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
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
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
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008],
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
    case.step("cache_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "two_holders",
        "kv_distinct",
        params={
            "first": output("first_holder", "engine"),
            "second": output("second_holder", "engine"),
        },
    )
    case.step(
        "mixed_0",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_0", "requests")},
    )
    case.step(
        "mixed_1",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_1", "requests")},
    )
    case.step(
        "mixed_2",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_2", "requests")},
    )
    case.step(
        "mixed_3",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000300,
                97000301,
                97000302,
                97000303,
                97000304,
                97000305,
                97000306,
                97000307,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_3", "requests")},
    )
    case.step(
        "mixed_4",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000400,
                97000401,
                97000402,
                97000403,
                97000404,
                97000405,
                97000406,
                97000407,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_4", "requests")},
    )
    case.step(
        "mixed_5",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_5", "requests")},
    )
    case.step(
        "mixed_6",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_6", "requests")},
    )
    case.step(
        "mixed_7",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_7", "requests")},
    )
    case.step(
        "mixed_8",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000800,
                97000801,
                97000802,
                97000803,
                97000804,
                97000805,
                97000806,
                97000807,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_8", "requests")},
    )
    case.step(
        "mixed_9",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000900,
                97000901,
                97000902,
                97000903,
                97000904,
                97000905,
                97000906,
                97000907,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_9", "requests")},
    )
    case.step(
        "mixed_10",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_10_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_10", "requests")},
    )
    case.step(
        "mixed_11",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_11_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_11", "requests")},
    )
    case.step(
        "mixed_12",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_12_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_12", "requests")},
    )
    case.step(
        "mixed_13",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001300,
                97001301,
                97001302,
                97001303,
                97001304,
                97001305,
                97001306,
                97001307,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_13_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_13", "requests")},
    )
    case.step(
        "mixed_14",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001400,
                97001401,
                97001402,
                97001403,
                97001404,
                97001405,
                97001406,
                97001407,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_14_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_14", "requests")},
    )
    case.step(
        "mixed_15",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_15_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_15", "requests")},
    )
    case.step(
        "mixed_16",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_16_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_16", "requests")},
    )
    case.step(
        "mixed_17",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_17_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_17", "requests")},
    )
    case.step(
        "mixed_18",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001800,
                97001801,
                97001802,
                97001803,
                97001804,
                97001805,
                97001806,
                97001807,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_18_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_18", "requests")},
    )
    case.step(
        "mixed_19",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001900,
                97001901,
                97001902,
                97001903,
                97001904,
                97001905,
                97001906,
                97001907,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_19_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_19", "requests")},
    )
    case.step(
        "mixed_20",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_20_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_20", "requests")},
    )
    case.step(
        "mixed_21",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_21_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_21", "requests")},
    )
    case.step(
        "mixed_22",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_22_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_22", "requests")},
    )
    case.step(
        "mixed_23",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002300,
                97002301,
                97002302,
                97002303,
                97002304,
                97002305,
                97002306,
                97002307,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_23_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_23", "requests")},
    )
    case.step(
        "mixed_24",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002400,
                97002401,
                97002402,
                97002403,
                97002404,
                97002405,
                97002406,
                97002407,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_24_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_24", "requests")},
    )
    case.step(
        "mixed_25",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_25_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_25", "requests")},
    )
    case.step(
        "mixed_26",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_26_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_26", "requests")},
    )
    case.step(
        "mixed_27",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_27_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_27", "requests")},
    )
    case.step(
        "mixed_28",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002800,
                97002801,
                97002802,
                97002803,
                97002804,
                97002805,
                97002806,
                97002807,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_28_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_28", "requests")},
    )
    case.step(
        "mixed_29",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002900,
                97002901,
                97002902,
                97002903,
                97002904,
                97002905,
                97002906,
                97002907,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_29_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_29", "requests")},
    )
    case.step(
        "family_fidelity",
        "kv_fidelity_check",
        params={
            "requests": [
                output("mixed_0", "requests"),
                output("mixed_1", "requests"),
                output("mixed_2", "requests"),
                output("mixed_5", "requests"),
                output("mixed_6", "requests"),
                output("mixed_7", "requests"),
                output("mixed_10", "requests"),
                output("mixed_11", "requests"),
                output("mixed_12", "requests"),
                output("mixed_15", "requests"),
                output("mixed_16", "requests"),
                output("mixed_17", "requests"),
                output("mixed_20", "requests"),
                output("mixed_21", "requests"),
                output("mixed_22", "requests"),
                output("mixed_25", "requests"),
                output("mixed_26", "requests"),
                output("mixed_27", "requests"),
            ],
            "holder": output("first_holder", "engine"),
            "min_samples": 18,
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step(
        "free_spread",
        "kv_union_check",
        params={
            "requests": [
                output("mixed_3", "requests"),
                output("mixed_4", "requests"),
                output("mixed_8", "requests"),
                output("mixed_9", "requests"),
                output("mixed_13", "requests"),
                output("mixed_14", "requests"),
                output("mixed_18", "requests"),
                output("mixed_19", "requests"),
                output("mixed_23", "requests"),
                output("mixed_24", "requests"),
                output("mixed_28", "requests"),
                output("mixed_29", "requests"),
            ],
            "holders": ["prefill-0", "prefill-1"],
            "min_samples": 12,
            "min_used": 2,
        },
    )
    case.step("cleanup", "teardown")


def prefix_nonbatch(case):
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
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
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
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008],
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
    case.step("cache_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "two_holders",
        "kv_distinct",
        params={
            "first": output("first_holder", "engine"),
            "second": output("second_holder", "engine"),
        },
    )
    case.step(
        "mixed_0",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_0", "requests")},
    )
    case.step(
        "mixed_1",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_1", "requests")},
    )
    case.step(
        "mixed_2",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_2", "requests")},
    )
    case.step(
        "mixed_3",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000300,
                97000301,
                97000302,
                97000303,
                97000304,
                97000305,
                97000306,
                97000307,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_3", "requests")},
    )
    case.step(
        "mixed_4",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000400,
                97000401,
                97000402,
                97000403,
                97000404,
                97000405,
                97000406,
                97000407,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_4", "requests")},
    )
    case.step(
        "mixed_5",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_5", "requests")},
    )
    case.step(
        "mixed_6",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_6", "requests")},
    )
    case.step(
        "mixed_7",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_7", "requests")},
    )
    case.step(
        "mixed_8",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000800,
                97000801,
                97000802,
                97000803,
                97000804,
                97000805,
                97000806,
                97000807,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_8", "requests")},
    )
    case.step(
        "mixed_9",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97000900,
                97000901,
                97000902,
                97000903,
                97000904,
                97000905,
                97000906,
                97000907,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_9", "requests")},
    )
    case.step(
        "mixed_10",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_10_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_10", "requests")},
    )
    case.step(
        "mixed_11",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_11_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_11", "requests")},
    )
    case.step(
        "mixed_12",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_12_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_12", "requests")},
    )
    case.step(
        "mixed_13",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001300,
                97001301,
                97001302,
                97001303,
                97001304,
                97001305,
                97001306,
                97001307,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_13_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_13", "requests")},
    )
    case.step(
        "mixed_14",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001400,
                97001401,
                97001402,
                97001403,
                97001404,
                97001405,
                97001406,
                97001407,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_14_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_14", "requests")},
    )
    case.step(
        "mixed_15",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_15_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_15", "requests")},
    )
    case.step(
        "mixed_16",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_16_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_16", "requests")},
    )
    case.step(
        "mixed_17",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_17_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_17", "requests")},
    )
    case.step(
        "mixed_18",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001800,
                97001801,
                97001802,
                97001803,
                97001804,
                97001805,
                97001806,
                97001807,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_18_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_18", "requests")},
    )
    case.step(
        "mixed_19",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97001900,
                97001901,
                97001902,
                97001903,
                97001904,
                97001905,
                97001906,
                97001907,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_19_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_19", "requests")},
    )
    case.step(
        "mixed_20",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_20_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_20", "requests")},
    )
    case.step(
        "mixed_21",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_21_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_21", "requests")},
    )
    case.step(
        "mixed_22",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_22_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_22", "requests")},
    )
    case.step(
        "mixed_23",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002300,
                97002301,
                97002302,
                97002303,
                97002304,
                97002305,
                97002306,
                97002307,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_23_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_23", "requests")},
    )
    case.step(
        "mixed_24",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002400,
                97002401,
                97002402,
                97002403,
                97002404,
                97002405,
                97002406,
                97002407,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_24_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_24", "requests")},
    )
    case.step(
        "mixed_25",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_25_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_25", "requests")},
    )
    case.step(
        "mixed_26",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_26_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_26", "requests")},
    )
    case.step(
        "mixed_27",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_27_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_27", "requests")},
    )
    case.step(
        "mixed_28",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002800,
                97002801,
                97002802,
                97002803,
                97002804,
                97002805,
                97002806,
                97002807,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_28_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_28", "requests")},
    )
    case.step(
        "mixed_29",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                97002900,
                97002901,
                97002902,
                97002903,
                97002904,
                97002905,
                97002906,
                97002907,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_29_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_29", "requests")},
    )
    case.step(
        "family_fidelity",
        "kv_fidelity_check",
        params={
            "requests": [
                output("mixed_0", "requests"),
                output("mixed_1", "requests"),
                output("mixed_2", "requests"),
                output("mixed_5", "requests"),
                output("mixed_6", "requests"),
                output("mixed_7", "requests"),
                output("mixed_10", "requests"),
                output("mixed_11", "requests"),
                output("mixed_12", "requests"),
                output("mixed_15", "requests"),
                output("mixed_16", "requests"),
                output("mixed_17", "requests"),
                output("mixed_20", "requests"),
                output("mixed_21", "requests"),
                output("mixed_22", "requests"),
                output("mixed_25", "requests"),
                output("mixed_26", "requests"),
                output("mixed_27", "requests"),
            ],
            "holder": output("first_holder", "engine"),
            "min_samples": 18,
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step(
        "free_spread",
        "kv_union_check",
        params={
            "requests": [
                output("mixed_3", "requests"),
                output("mixed_4", "requests"),
                output("mixed_8", "requests"),
                output("mixed_9", "requests"),
                output("mixed_13", "requests"),
                output("mixed_14", "requests"),
                output("mixed_18", "requests"),
                output("mixed_19", "requests"),
                output("mixed_23", "requests"),
                output("mixed_24", "requests"),
                output("mixed_28", "requests"),
                output("mixed_29", "requests"),
            ],
            "holders": ["prefill-0", "prefill-1"],
            "min_samples": 12,
            "min_used": 2,
        },
    )
    case.step("cleanup", "teardown")


def hot_tension(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "seed",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "seed_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("seed", "requests")},
    )
    case.step("holder", "kv_landing", params={"requests": output("seed", "requests")})
    case.step("cache_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "mixed_0",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_0", "requests")},
    )
    case.step(
        "mixed_1",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_1", "requests")},
    )
    case.step(
        "mixed_2",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_2", "requests")},
    )
    case.step(
        "mixed_3",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_3", "requests")},
    )
    case.step(
        "mixed_4",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_4", "requests")},
    )
    case.step(
        "mixed_5",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_5", "requests")},
    )
    case.step(
        "mixed_6",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_6", "requests")},
    )
    case.step(
        "mixed_7",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98000700,
                98000701,
                98000702,
                98000703,
                98000704,
                98000705,
                98000706,
                98000707,
                98000708,
                98000709,
                98000710,
                98000711,
                98000712,
                98000713,
                98000714,
                98000715,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_7", "requests")},
    )
    case.step(
        "mixed_8",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98000800,
                98000801,
                98000802,
                98000803,
                98000804,
                98000805,
                98000806,
                98000807,
                98000808,
                98000809,
                98000810,
                98000811,
                98000812,
                98000813,
                98000814,
                98000815,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_8", "requests")},
    )
    case.step(
        "mixed_9",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98000900,
                98000901,
                98000902,
                98000903,
                98000904,
                98000905,
                98000906,
                98000907,
                98000908,
                98000909,
                98000910,
                98000911,
                98000912,
                98000913,
                98000914,
                98000915,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_9", "requests")},
    )
    case.step(
        "mixed_10",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_10_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_10", "requests")},
    )
    case.step(
        "mixed_11",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_11_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_11", "requests")},
    )
    case.step(
        "mixed_12",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_12_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_12", "requests")},
    )
    case.step(
        "mixed_13",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_13_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_13", "requests")},
    )
    case.step(
        "mixed_14",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_14_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_14", "requests")},
    )
    case.step(
        "mixed_15",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_15_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_15", "requests")},
    )
    case.step(
        "mixed_16",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_16_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_16", "requests")},
    )
    case.step(
        "mixed_17",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98001700,
                98001701,
                98001702,
                98001703,
                98001704,
                98001705,
                98001706,
                98001707,
                98001708,
                98001709,
                98001710,
                98001711,
                98001712,
                98001713,
                98001714,
                98001715,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_17_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_17", "requests")},
    )
    case.step(
        "mixed_18",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98001800,
                98001801,
                98001802,
                98001803,
                98001804,
                98001805,
                98001806,
                98001807,
                98001808,
                98001809,
                98001810,
                98001811,
                98001812,
                98001813,
                98001814,
                98001815,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_18_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_18", "requests")},
    )
    case.step(
        "mixed_19",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98001900,
                98001901,
                98001902,
                98001903,
                98001904,
                98001905,
                98001906,
                98001907,
                98001908,
                98001909,
                98001910,
                98001911,
                98001912,
                98001913,
                98001914,
                98001915,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_19_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_19", "requests")},
    )
    case.step(
        "mixed_20",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_20_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_20", "requests")},
    )
    case.step(
        "mixed_21",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_21_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_21", "requests")},
    )
    case.step(
        "mixed_22",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_22_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_22", "requests")},
    )
    case.step(
        "mixed_23",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_23_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_23", "requests")},
    )
    case.step(
        "mixed_24",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_24_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_24", "requests")},
    )
    case.step(
        "mixed_25",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_25_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_25", "requests")},
    )
    case.step(
        "mixed_26",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_26_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_26", "requests")},
    )
    case.step(
        "mixed_27",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98002700,
                98002701,
                98002702,
                98002703,
                98002704,
                98002705,
                98002706,
                98002707,
                98002708,
                98002709,
                98002710,
                98002711,
                98002712,
                98002713,
                98002714,
                98002715,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_27_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_27", "requests")},
    )
    case.step(
        "mixed_28",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98002800,
                98002801,
                98002802,
                98002803,
                98002804,
                98002805,
                98002806,
                98002807,
                98002808,
                98002809,
                98002810,
                98002811,
                98002812,
                98002813,
                98002814,
                98002815,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_28_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_28", "requests")},
    )
    case.step(
        "mixed_29",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98002900,
                98002901,
                98002902,
                98002903,
                98002904,
                98002905,
                98002906,
                98002907,
                98002908,
                98002909,
                98002910,
                98002911,
                98002912,
                98002913,
                98002914,
                98002915,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_29_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_29", "requests")},
    )
    case.step(
        "mixed_30",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_30_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_30", "requests")},
    )
    case.step(
        "mixed_31",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_31_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_31", "requests")},
    )
    case.step(
        "mixed_32",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_32_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_32", "requests")},
    )
    case.step(
        "mixed_33",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_33_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_33", "requests")},
    )
    case.step(
        "mixed_34",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_34_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_34", "requests")},
    )
    case.step(
        "mixed_35",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_35_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_35", "requests")},
    )
    case.step(
        "mixed_36",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                3001,
                3002,
                3003,
                3004,
                3005,
                3006,
                3007,
                3008,
                3009,
                3010,
                3011,
                3012,
                3013,
                3014,
                3015,
                3016,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_36_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_36", "requests")},
    )
    case.step(
        "mixed_37",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98003700,
                98003701,
                98003702,
                98003703,
                98003704,
                98003705,
                98003706,
                98003707,
                98003708,
                98003709,
                98003710,
                98003711,
                98003712,
                98003713,
                98003714,
                98003715,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_37_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_37", "requests")},
    )
    case.step(
        "mixed_38",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98003800,
                98003801,
                98003802,
                98003803,
                98003804,
                98003805,
                98003806,
                98003807,
                98003808,
                98003809,
                98003810,
                98003811,
                98003812,
                98003813,
                98003814,
                98003815,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_38_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_38", "requests")},
    )
    case.step(
        "mixed_39",
        "request",
        params={
            "count": 1,
            "input_len": 16384,
            "output_len": 2,
            "block_keys": [
                98003900,
                98003901,
                98003902,
                98003903,
                98003904,
                98003905,
                98003906,
                98003907,
                98003908,
                98003909,
                98003910,
                98003911,
                98003912,
                98003913,
                98003914,
                98003915,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "mixed_39_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("mixed_39", "requests")},
    )
    case.step(
        "family_fidelity",
        "kv_fidelity_check",
        params={
            "requests": [
                output("mixed_0", "requests"),
                output("mixed_1", "requests"),
                output("mixed_2", "requests"),
                output("mixed_3", "requests"),
                output("mixed_4", "requests"),
                output("mixed_5", "requests"),
                output("mixed_6", "requests"),
                output("mixed_10", "requests"),
                output("mixed_11", "requests"),
                output("mixed_12", "requests"),
                output("mixed_13", "requests"),
                output("mixed_14", "requests"),
                output("mixed_15", "requests"),
                output("mixed_16", "requests"),
                output("mixed_20", "requests"),
                output("mixed_21", "requests"),
                output("mixed_22", "requests"),
                output("mixed_23", "requests"),
                output("mixed_24", "requests"),
                output("mixed_25", "requests"),
                output("mixed_26", "requests"),
                output("mixed_30", "requests"),
                output("mixed_31", "requests"),
                output("mixed_32", "requests"),
                output("mixed_33", "requests"),
                output("mixed_34", "requests"),
                output("mixed_35", "requests"),
                output("mixed_36", "requests"),
            ],
            "holder": output("holder", "engine"),
            "min_samples": 28,
            "bands": {"strict": 0.95, "normal": 0.9, "loose": 0.8},
        },
    )
    case.step(
        "holder_total",
        "kv_holder_share_check",
        params={
            "requests": [
                output("seed", "requests"),
                output("mixed_0", "requests"),
                output("mixed_1", "requests"),
                output("mixed_2", "requests"),
                output("mixed_3", "requests"),
                output("mixed_4", "requests"),
                output("mixed_5", "requests"),
                output("mixed_6", "requests"),
                output("mixed_7", "requests"),
                output("mixed_8", "requests"),
                output("mixed_9", "requests"),
                output("mixed_10", "requests"),
                output("mixed_11", "requests"),
                output("mixed_12", "requests"),
                output("mixed_13", "requests"),
                output("mixed_14", "requests"),
                output("mixed_15", "requests"),
                output("mixed_16", "requests"),
                output("mixed_17", "requests"),
                output("mixed_18", "requests"),
                output("mixed_19", "requests"),
                output("mixed_20", "requests"),
                output("mixed_21", "requests"),
                output("mixed_22", "requests"),
                output("mixed_23", "requests"),
                output("mixed_24", "requests"),
                output("mixed_25", "requests"),
                output("mixed_26", "requests"),
                output("mixed_27", "requests"),
                output("mixed_28", "requests"),
                output("mixed_29", "requests"),
                output("mixed_30", "requests"),
                output("mixed_31", "requests"),
                output("mixed_32", "requests"),
                output("mixed_33", "requests"),
                output("mixed_34", "requests"),
                output("mixed_35", "requests"),
                output("mixed_36", "requests"),
                output("mixed_37", "requests"),
                output("mixed_38", "requests"),
                output("mixed_39", "requests"),
            ],
            "holder": output("holder", "engine"),
            "min_samples": 41,
            "bands": {"strict": 0.88, "normal": 0.93, "loose": 0.96},
        },
    )
    case.step(
        "free_not_starved",
        "kv_off_holder_check",
        params={
            "requests": [
                output("mixed_7", "requests"),
                output("mixed_8", "requests"),
                output("mixed_9", "requests"),
                output("mixed_17", "requests"),
                output("mixed_18", "requests"),
                output("mixed_19", "requests"),
                output("mixed_27", "requests"),
                output("mixed_28", "requests"),
                output("mixed_29", "requests"),
                output("mixed_37", "requests"),
                output("mixed_38", "requests"),
                output("mixed_39", "requests"),
            ],
            "holder": output("holder", "engine"),
            "min_samples": 12,
        },
    )
    case.step("cleanup", "teardown")


def mixed_tiers(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "full_seed",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_seed_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_seed", "requests")},
    )
    case.step(
        "full_holder",
        "kv_landing",
        params={"requests": output("full_seed", "requests")},
    )
    case.step("full_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "full_0",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_0", "requests")},
    )
    case.step(
        "full_1",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_1", "requests")},
    )
    case.step(
        "full_2",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_2", "requests")},
    )
    case.step(
        "full_3",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_3", "requests")},
    )
    case.step(
        "full_4",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_4", "requests")},
    )
    case.step(
        "full_5",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_5", "requests")},
    )
    case.step(
        "full_6",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_6", "requests")},
    )
    case.step(
        "full_7",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_7", "requests")},
    )
    case.step(
        "full_8",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_8", "requests")},
    )
    case.step(
        "full_9",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "full_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("full_9", "requests")},
    )
    case.step(
        "half_seed",
        "request",
        params={
            "count": 1,
            "input_len": 4096,
            "output_len": 2,
            "block_keys": [5001, 5002, 5003, 5004],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_seed_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_seed", "requests")},
    )
    case.step(
        "half_holder",
        "kv_landing",
        params={"requests": output("half_seed", "requests")},
    )
    case.step("half_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "half_0",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000000,
                99000001,
                99000002,
                99000003,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_0", "requests")},
    )
    case.step(
        "half_1",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000100,
                99000101,
                99000102,
                99000103,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_1", "requests")},
    )
    case.step(
        "half_2",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000200,
                99000201,
                99000202,
                99000203,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_2", "requests")},
    )
    case.step(
        "half_3",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000300,
                99000301,
                99000302,
                99000303,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_3", "requests")},
    )
    case.step(
        "half_4",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000400,
                99000401,
                99000402,
                99000403,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_4", "requests")},
    )
    case.step(
        "half_5",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000500,
                99000501,
                99000502,
                99000503,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_5", "requests")},
    )
    case.step(
        "half_6",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000600,
                99000601,
                99000602,
                99000603,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_6", "requests")},
    )
    case.step(
        "half_7",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000700,
                99000701,
                99000702,
                99000703,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_7", "requests")},
    )
    case.step(
        "half_8",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000800,
                99000801,
                99000802,
                99000803,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_8", "requests")},
    )
    case.step(
        "half_9",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                5001,
                5002,
                5003,
                5004,
                99000900,
                99000901,
                99000902,
                99000903,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "half_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("half_9", "requests")},
    )
    case.step(
        "zero_0",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500000,
                99500001,
                99500002,
                99500003,
                99500004,
                99500005,
                99500006,
                99500007,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_0_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_0", "requests")},
    )
    case.step(
        "zero_1",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500100,
                99500101,
                99500102,
                99500103,
                99500104,
                99500105,
                99500106,
                99500107,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_1_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_1", "requests")},
    )
    case.step(
        "zero_2",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500200,
                99500201,
                99500202,
                99500203,
                99500204,
                99500205,
                99500206,
                99500207,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_2_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_2", "requests")},
    )
    case.step(
        "zero_3",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500300,
                99500301,
                99500302,
                99500303,
                99500304,
                99500305,
                99500306,
                99500307,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_3_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_3", "requests")},
    )
    case.step(
        "zero_4",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500400,
                99500401,
                99500402,
                99500403,
                99500404,
                99500405,
                99500406,
                99500407,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_4_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_4", "requests")},
    )
    case.step(
        "zero_5",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500500,
                99500501,
                99500502,
                99500503,
                99500504,
                99500505,
                99500506,
                99500507,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_5_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_5", "requests")},
    )
    case.step(
        "zero_6",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500600,
                99500601,
                99500602,
                99500603,
                99500604,
                99500605,
                99500606,
                99500607,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_6_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_6", "requests")},
    )
    case.step(
        "zero_7",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500700,
                99500701,
                99500702,
                99500703,
                99500704,
                99500705,
                99500706,
                99500707,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_7_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_7", "requests")},
    )
    case.step(
        "zero_8",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500800,
                99500801,
                99500802,
                99500803,
                99500804,
                99500805,
                99500806,
                99500807,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_8_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_8", "requests")},
    )
    case.step(
        "zero_9",
        "request",
        params={
            "count": 1,
            "input_len": 8192,
            "output_len": 2,
            "block_keys": [
                99500900,
                99500901,
                99500902,
                99500903,
                99500904,
                99500905,
                99500906,
                99500907,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "zero_9_terminal",
        "wait",
        timeout_s=15,
        params={"requests": output("zero_9", "requests")},
    )
    case.step(
        "full_concentration",
        "kv_affinity_check",
        params={
            "requests": [
                output("full_0", "requests"),
                output("full_1", "requests"),
                output("full_2", "requests"),
                output("full_3", "requests"),
                output("full_4", "requests"),
                output("full_5", "requests"),
                output("full_6", "requests"),
                output("full_7", "requests"),
                output("full_8", "requests"),
                output("full_9", "requests"),
            ],
            "holder": output("full_holder", "engine"),
            "min_samples": 10,
            "bands": {"strict": 0.8, "normal": 0.7, "loose": 0.6},
        },
    )
    case.step(
        "half_concentration",
        "kv_affinity_check",
        params={
            "requests": [
                output("half_0", "requests"),
                output("half_1", "requests"),
                output("half_2", "requests"),
                output("half_3", "requests"),
                output("half_4", "requests"),
                output("half_5", "requests"),
                output("half_6", "requests"),
                output("half_7", "requests"),
                output("half_8", "requests"),
                output("half_9", "requests"),
            ],
            "holder": output("half_holder", "engine"),
            "min_samples": 10,
            "bands": {"strict": 0.8, "normal": 0.7, "loose": 0.6},
        },
    )
    case.step(
        "zero_spread",
        "kv_union_check",
        params={
            "requests": [
                output("zero_0", "requests"),
                output("zero_1", "requests"),
                output("zero_2", "requests"),
                output("zero_3", "requests"),
                output("zero_4", "requests"),
                output("zero_5", "requests"),
                output("zero_6", "requests"),
                output("zero_7", "requests"),
                output("zero_8", "requests"),
                output("zero_9", "requests"),
            ],
            "holders": ["prefill-0", "prefill-1"],
            "min_samples": 10,
            "min_used": 2,
        },
    )
    case.step("cleanup", "teardown")


def leader_spill_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_second",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-1"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("steer_a_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "steer_a",
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
        "steer_a_done",
        "wait",
        timeout_s=15,
        params={"requests": output("steer_a", "requests")},
    )
    case.step(
        "steer_a_holder",
        "kv_landing",
        params={"requests": output("steer_a", "requests")},
    )
    case.step(
        "steer_a_placement",
        "kv_same",
        params={"first": output("steer_a_holder", "engine"), "second": "prefill-0"},
    )
    case.step(
        "steer_a_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "restore_second",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "slow_first",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("steer_b_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "steer_b",
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
        "steer_b_done",
        "wait",
        timeout_s=15,
        params={"requests": output("steer_b", "requests")},
    )
    case.step(
        "steer_b_holder",
        "kv_landing",
        params={"requests": output("steer_b", "requests")},
    )
    case.step(
        "steer_b_placement",
        "kv_same",
        params={"first": output("steer_b_holder", "engine"), "second": "prefill-1"},
    )
    case.step(
        "restore_first",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("steer_restore_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "steer_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "steer_family_0",
        "kv_membership_check",
        params={
            "snapshot": output("steer_quiet", "snapshot"),
            "engine": "prefill-0",
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
        "steer_family_1",
        "kv_membership_check",
        params={
            "snapshot": output("steer_quiet", "snapshot"),
            "engine": "prefill-1",
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
            "relation": "all",
        },
    )
    case.step(
        "baseline_w0_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r0",
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
        "baseline_w0_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r0", "requests")},
    )
    case.step(
        "baseline_w0_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r0_before", "snapshot"),
            "requests": output("baseline_w0_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r1",
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
        "baseline_w0_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r1", "requests")},
    )
    case.step(
        "baseline_w0_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r1_before", "snapshot"),
            "requests": output("baseline_w0_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r2",
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
        "baseline_w0_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r2", "requests")},
    )
    case.step(
        "baseline_w0_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r2_before", "snapshot"),
            "requests": output("baseline_w0_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r3",
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
        "baseline_w0_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r3", "requests")},
    )
    case.step(
        "baseline_w0_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r3_before", "snapshot"),
            "requests": output("baseline_w0_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("baseline_w0_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "baseline_w1_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r0",
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
        "baseline_w1_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r0", "requests")},
    )
    case.step(
        "baseline_w1_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r0_before", "snapshot"),
            "requests": output("baseline_w1_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r1",
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
        "baseline_w1_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r1", "requests")},
    )
    case.step(
        "baseline_w1_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r1_before", "snapshot"),
            "requests": output("baseline_w1_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r2",
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
        "baseline_w1_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r2", "requests")},
    )
    case.step(
        "baseline_w1_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r2_before", "snapshot"),
            "requests": output("baseline_w1_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r3",
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
        "baseline_w1_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r3", "requests")},
    )
    case.step(
        "baseline_w1_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r3_before", "snapshot"),
            "requests": output("baseline_w1_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("baseline_w1_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "baseline_w2_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r0",
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
        "baseline_w2_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r0", "requests")},
    )
    case.step(
        "baseline_w2_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r0_before", "snapshot"),
            "requests": output("baseline_w2_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r1",
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
        "baseline_w2_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r1", "requests")},
    )
    case.step(
        "baseline_w2_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r1_before", "snapshot"),
            "requests": output("baseline_w2_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r2",
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
        "baseline_w2_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r2", "requests")},
    )
    case.step(
        "baseline_w2_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r2_before", "snapshot"),
            "requests": output("baseline_w2_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r3",
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
        "baseline_w2_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r3", "requests")},
    )
    case.step(
        "baseline_w2_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r3_before", "snapshot"),
            "requests": output("baseline_w2_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("baseline_w2_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "baseline_digest",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturate_leader",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("saturation_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "saturation_w0_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r0",
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
    case.step("saturation_w0_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w0_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r1",
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
    case.step("saturation_w0_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w0_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r2",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w0_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w0_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r3",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w0_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r0", "requests")},
    )
    case.step(
        "saturation_w0_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r0_before", "snapshot"),
            "requests": output("saturation_w0_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r1", "requests")},
    )
    case.step(
        "saturation_w0_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r1_before", "snapshot"),
            "requests": output("saturation_w0_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r2", "requests")},
    )
    case.step(
        "saturation_w0_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r2_before", "snapshot"),
            "requests": output("saturation_w0_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r3", "requests")},
    )
    case.step(
        "saturation_w0_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r3_before", "snapshot"),
            "requests": output("saturation_w0_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r0",
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
    case.step("saturation_w1_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w1_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r1",
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
    case.step("saturation_w1_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w1_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r2",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w1_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w1_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r3",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w1_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r0", "requests")},
    )
    case.step(
        "saturation_w1_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r0_before", "snapshot"),
            "requests": output("saturation_w1_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r1", "requests")},
    )
    case.step(
        "saturation_w1_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r1_before", "snapshot"),
            "requests": output("saturation_w1_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r2", "requests")},
    )
    case.step(
        "saturation_w1_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r2_before", "snapshot"),
            "requests": output("saturation_w1_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r3", "requests")},
    )
    case.step(
        "saturation_w1_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r3_before", "snapshot"),
            "requests": output("saturation_w1_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r0",
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
    case.step("saturation_w2_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w2_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r1",
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
    case.step("saturation_w2_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w2_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r2",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w2_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w2_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r3",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w2_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r0", "requests")},
    )
    case.step(
        "saturation_w2_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r0_before", "snapshot"),
            "requests": output("saturation_w2_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r1", "requests")},
    )
    case.step(
        "saturation_w2_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r1_before", "snapshot"),
            "requests": output("saturation_w2_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r2", "requests")},
    )
    case.step(
        "saturation_w2_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r2_before", "snapshot"),
            "requests": output("saturation_w2_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r3", "requests")},
    )
    case.step(
        "saturation_w2_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r3_before", "snapshot"),
            "requests": output("saturation_w2_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r0",
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
    case.step("saturation_w3_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w3_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r1",
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
    case.step("saturation_w3_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w3_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r2",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w3_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w3_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r3",
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
            "consume": "deferred",
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w3_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r0", "requests")},
    )
    case.step(
        "saturation_w3_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r0_before", "snapshot"),
            "requests": output("saturation_w3_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r1", "requests")},
    )
    case.step(
        "saturation_w3_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r1_before", "snapshot"),
            "requests": output("saturation_w3_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r2", "requests")},
    )
    case.step(
        "saturation_w3_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r2_before", "snapshot"),
            "requests": output("saturation_w3_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r3", "requests")},
    )
    case.step(
        "saturation_w3_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r3_before", "snapshot"),
            "requests": output("saturation_w3_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recover_leader",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("recovery_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "saturation_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "recovery_w0_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r0",
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
        "recovery_w0_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r0", "requests")},
    )
    case.step(
        "recovery_w0_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r0_before", "snapshot"),
            "requests": output("recovery_w0_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r1",
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
        "recovery_w0_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r1", "requests")},
    )
    case.step(
        "recovery_w0_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r1_before", "snapshot"),
            "requests": output("recovery_w0_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r2",
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
        "recovery_w0_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r2", "requests")},
    )
    case.step(
        "recovery_w0_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r2_before", "snapshot"),
            "requests": output("recovery_w0_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r3",
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
        "recovery_w0_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r3", "requests")},
    )
    case.step(
        "recovery_w0_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r3_before", "snapshot"),
            "requests": output("recovery_w0_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w0_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w1_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r0",
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
        "recovery_w1_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r0", "requests")},
    )
    case.step(
        "recovery_w1_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r0_before", "snapshot"),
            "requests": output("recovery_w1_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r1",
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
        "recovery_w1_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r1", "requests")},
    )
    case.step(
        "recovery_w1_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r1_before", "snapshot"),
            "requests": output("recovery_w1_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r2",
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
        "recovery_w1_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r2", "requests")},
    )
    case.step(
        "recovery_w1_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r2_before", "snapshot"),
            "requests": output("recovery_w1_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r3",
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
        "recovery_w1_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r3", "requests")},
    )
    case.step(
        "recovery_w1_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r3_before", "snapshot"),
            "requests": output("recovery_w1_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w1_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w2_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r0",
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
        "recovery_w2_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r0", "requests")},
    )
    case.step(
        "recovery_w2_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r0_before", "snapshot"),
            "requests": output("recovery_w2_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r1",
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
        "recovery_w2_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r1", "requests")},
    )
    case.step(
        "recovery_w2_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r1_before", "snapshot"),
            "requests": output("recovery_w2_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r2",
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
        "recovery_w2_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r2", "requests")},
    )
    case.step(
        "recovery_w2_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r2_before", "snapshot"),
            "requests": output("recovery_w2_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r3",
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
        "recovery_w2_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r3", "requests")},
    )
    case.step(
        "recovery_w2_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r3_before", "snapshot"),
            "requests": output("recovery_w2_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w2_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w3_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r0",
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
        "recovery_w3_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r0", "requests")},
    )
    case.step(
        "recovery_w3_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r0_before", "snapshot"),
            "requests": output("recovery_w3_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r1",
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
        "recovery_w3_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r1", "requests")},
    )
    case.step(
        "recovery_w3_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r1_before", "snapshot"),
            "requests": output("recovery_w3_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r2",
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
        "recovery_w3_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r2", "requests")},
    )
    case.step(
        "recovery_w3_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r2_before", "snapshot"),
            "requests": output("recovery_w3_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r3",
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
        "recovery_w3_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r3", "requests")},
    )
    case.step(
        "recovery_w3_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r3_before", "snapshot"),
            "requests": output("recovery_w3_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w3_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w4_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r0",
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
        "recovery_w4_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r0", "requests")},
    )
    case.step(
        "recovery_w4_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r0_before", "snapshot"),
            "requests": output("recovery_w4_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r1",
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
        "recovery_w4_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r1", "requests")},
    )
    case.step(
        "recovery_w4_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r1_before", "snapshot"),
            "requests": output("recovery_w4_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r2",
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
        "recovery_w4_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r2", "requests")},
    )
    case.step(
        "recovery_w4_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r2_before", "snapshot"),
            "requests": output("recovery_w4_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r3",
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
        "recovery_w4_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r3", "requests")},
    )
    case.step(
        "recovery_w4_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r3_before", "snapshot"),
            "requests": output("recovery_w4_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w4_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w5_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r0",
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
        "recovery_w5_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r0", "requests")},
    )
    case.step(
        "recovery_w5_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r0_before", "snapshot"),
            "requests": output("recovery_w5_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r1",
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
        "recovery_w5_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r1", "requests")},
    )
    case.step(
        "recovery_w5_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r1_before", "snapshot"),
            "requests": output("recovery_w5_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r2",
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
        "recovery_w5_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r2", "requests")},
    )
    case.step(
        "recovery_w5_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r2_before", "snapshot"),
            "requests": output("recovery_w5_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r3",
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
        "recovery_w5_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r3", "requests")},
    )
    case.step(
        "recovery_w5_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r3_before", "snapshot"),
            "requests": output("recovery_w5_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w5_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_digest",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "phase_observations",
        "kv_phase_observation",
        params={
            "saturation": [
                output("saturation_w0_r0_hit", "sample"),
                output("saturation_w0_r1_hit", "sample"),
                output("saturation_w0_r2_hit", "sample"),
                output("saturation_w0_r3_hit", "sample"),
                output("saturation_w1_r0_hit", "sample"),
                output("saturation_w1_r1_hit", "sample"),
                output("saturation_w1_r2_hit", "sample"),
                output("saturation_w1_r3_hit", "sample"),
                output("saturation_w2_r0_hit", "sample"),
                output("saturation_w2_r1_hit", "sample"),
                output("saturation_w2_r2_hit", "sample"),
                output("saturation_w2_r3_hit", "sample"),
                output("saturation_w3_r0_hit", "sample"),
                output("saturation_w3_r1_hit", "sample"),
                output("saturation_w3_r2_hit", "sample"),
                output("saturation_w3_r3_hit", "sample"),
            ],
            "recovery_windows": [
                [
                    output("recovery_w0_r0_hit", "sample"),
                    output("recovery_w0_r1_hit", "sample"),
                    output("recovery_w0_r2_hit", "sample"),
                    output("recovery_w0_r3_hit", "sample"),
                ],
                [
                    output("recovery_w1_r0_hit", "sample"),
                    output("recovery_w1_r1_hit", "sample"),
                    output("recovery_w1_r2_hit", "sample"),
                    output("recovery_w1_r3_hit", "sample"),
                ],
                [
                    output("recovery_w2_r0_hit", "sample"),
                    output("recovery_w2_r1_hit", "sample"),
                    output("recovery_w2_r2_hit", "sample"),
                    output("recovery_w2_r3_hit", "sample"),
                ],
                [
                    output("recovery_w3_r0_hit", "sample"),
                    output("recovery_w3_r1_hit", "sample"),
                    output("recovery_w3_r2_hit", "sample"),
                    output("recovery_w3_r3_hit", "sample"),
                ],
                [
                    output("recovery_w4_r0_hit", "sample"),
                    output("recovery_w4_r1_hit", "sample"),
                    output("recovery_w4_r2_hit", "sample"),
                    output("recovery_w4_r3_hit", "sample"),
                ],
                [
                    output("recovery_w5_r0_hit", "sample"),
                    output("recovery_w5_r1_hit", "sample"),
                    output("recovery_w5_r2_hit", "sample"),
                    output("recovery_w5_r3_hit", "sample"),
                ],
            ],
            "family_keys": [
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
            "leader": "prefill-0",
            "ready_rate": 0.75,
            "phase_snapshots": {
                "steer": output("steer_quiet", "snapshot"),
                "baseline": output("baseline_digest", "snapshot"),
                "saturation": output("saturation_quiet", "snapshot"),
                "recovery": output("recovery_digest", "snapshot"),
            },
            "families": [
                [
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
                [
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
            ],
        },
    )
    case.step(
        "holder_flips",
        "kv_window_transitions",
        params={
            "snapshots": [
                output("baseline_w0_end", "snapshot"),
                output("baseline_w1_end", "snapshot"),
                output("baseline_w2_end", "snapshot"),
                output("saturation_w0_end", "snapshot"),
                output("saturation_w1_end", "snapshot"),
                output("saturation_w2_end", "snapshot"),
                output("saturation_w3_end", "snapshot"),
                output("recovery_w0_end", "snapshot"),
                output("recovery_w1_end", "snapshot"),
                output("recovery_w2_end", "snapshot"),
                output("recovery_w3_end", "snapshot"),
                output("recovery_w4_end", "snapshot"),
                output("recovery_w5_end", "snapshot"),
            ],
            "families": [
                [
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
                [
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
            ],
        },
    )
    case.step(
        "all_phase_requests",
        "kv_hit_completeness",
        params={
            "samples": [
                output("baseline_w0_r0_hit", "sample"),
                output("baseline_w0_r1_hit", "sample"),
                output("baseline_w0_r2_hit", "sample"),
                output("baseline_w0_r3_hit", "sample"),
                output("baseline_w1_r0_hit", "sample"),
                output("baseline_w1_r1_hit", "sample"),
                output("baseline_w1_r2_hit", "sample"),
                output("baseline_w1_r3_hit", "sample"),
                output("baseline_w2_r0_hit", "sample"),
                output("baseline_w2_r1_hit", "sample"),
                output("baseline_w2_r2_hit", "sample"),
                output("baseline_w2_r3_hit", "sample"),
                output("saturation_w0_r0_hit", "sample"),
                output("saturation_w0_r1_hit", "sample"),
                output("saturation_w0_r2_hit", "sample"),
                output("saturation_w0_r3_hit", "sample"),
                output("saturation_w1_r0_hit", "sample"),
                output("saturation_w1_r1_hit", "sample"),
                output("saturation_w1_r2_hit", "sample"),
                output("saturation_w1_r3_hit", "sample"),
                output("saturation_w2_r0_hit", "sample"),
                output("saturation_w2_r1_hit", "sample"),
                output("saturation_w2_r2_hit", "sample"),
                output("saturation_w2_r3_hit", "sample"),
                output("saturation_w3_r0_hit", "sample"),
                output("saturation_w3_r1_hit", "sample"),
                output("saturation_w3_r2_hit", "sample"),
                output("saturation_w3_r3_hit", "sample"),
                output("recovery_w0_r0_hit", "sample"),
                output("recovery_w0_r1_hit", "sample"),
                output("recovery_w0_r2_hit", "sample"),
                output("recovery_w0_r3_hit", "sample"),
                output("recovery_w1_r0_hit", "sample"),
                output("recovery_w1_r1_hit", "sample"),
                output("recovery_w1_r2_hit", "sample"),
                output("recovery_w1_r3_hit", "sample"),
                output("recovery_w2_r0_hit", "sample"),
                output("recovery_w2_r1_hit", "sample"),
                output("recovery_w2_r2_hit", "sample"),
                output("recovery_w2_r3_hit", "sample"),
                output("recovery_w3_r0_hit", "sample"),
                output("recovery_w3_r1_hit", "sample"),
                output("recovery_w3_r2_hit", "sample"),
                output("recovery_w3_r3_hit", "sample"),
                output("recovery_w4_r0_hit", "sample"),
                output("recovery_w4_r1_hit", "sample"),
                output("recovery_w4_r2_hit", "sample"),
                output("recovery_w4_r3_hit", "sample"),
                output("recovery_w5_r0_hit", "sample"),
                output("recovery_w5_r1_hit", "sample"),
                output("recovery_w5_r2_hit", "sample"),
                output("recovery_w5_r3_hit", "sample"),
            ],
            "min_samples": 52,
        },
    )
    case.step(
        "baseline_hit",
        "kv_hit_rate_check",
        params={
            "samples": [
                output("baseline_w0_r0_hit", "sample"),
                output("baseline_w0_r1_hit", "sample"),
                output("baseline_w0_r2_hit", "sample"),
                output("baseline_w0_r3_hit", "sample"),
                output("baseline_w1_r0_hit", "sample"),
                output("baseline_w1_r1_hit", "sample"),
                output("baseline_w1_r2_hit", "sample"),
                output("baseline_w1_r3_hit", "sample"),
                output("baseline_w2_r0_hit", "sample"),
                output("baseline_w2_r1_hit", "sample"),
                output("baseline_w2_r2_hit", "sample"),
                output("baseline_w2_r3_hit", "sample"),
            ],
            "min_samples": 12,
            "bands": {"strict": 0.9, "normal": 0.85, "loose": 0.8},
        },
    )
    case.step(
        "saturation_hit",
        "kv_hit_rate_check",
        params={
            "samples": [
                output("saturation_w0_r0_hit", "sample"),
                output("saturation_w0_r1_hit", "sample"),
                output("saturation_w0_r2_hit", "sample"),
                output("saturation_w0_r3_hit", "sample"),
                output("saturation_w1_r0_hit", "sample"),
                output("saturation_w1_r1_hit", "sample"),
                output("saturation_w1_r2_hit", "sample"),
                output("saturation_w1_r3_hit", "sample"),
                output("saturation_w2_r0_hit", "sample"),
                output("saturation_w2_r1_hit", "sample"),
                output("saturation_w2_r2_hit", "sample"),
                output("saturation_w2_r3_hit", "sample"),
                output("saturation_w3_r0_hit", "sample"),
                output("saturation_w3_r1_hit", "sample"),
                output("saturation_w3_r2_hit", "sample"),
                output("saturation_w3_r3_hit", "sample"),
            ],
            "min_samples": 16,
            "bands": {"strict": 0.5, "normal": 0.4, "loose": 0.3},
        },
    )
    case.step(
        "recovery_steady_hit",
        "kv_hit_rate_check",
        params={
            "samples": [
                output("recovery_w4_r0_hit", "sample"),
                output("recovery_w4_r1_hit", "sample"),
                output("recovery_w4_r2_hit", "sample"),
                output("recovery_w4_r3_hit", "sample"),
                output("recovery_w5_r0_hit", "sample"),
                output("recovery_w5_r1_hit", "sample"),
                output("recovery_w5_r2_hit", "sample"),
                output("recovery_w5_r3_hit", "sample"),
            ],
            "min_samples": 8,
            "bands": {"strict": 0.85, "normal": 0.8, "loose": 0.75},
        },
    )
    case.step(
        "replication",
        "kv_replication_check",
        params={
            "snapshot": output("recovery_w5_end", "snapshot"),
            "families": [
                [
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
                [
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
            ],
            "bands": {"strict": 1.5, "normal": 1.75, "loose": 2},
            "max_holders": 2,
        },
    )
    case.step("cleanup", "teardown")


def leader_spill_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_second",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-1"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("steer_a_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "steer_a",
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
        "steer_a_done",
        "wait",
        timeout_s=15,
        params={"requests": output("steer_a", "requests")},
    )
    case.step(
        "steer_a_holder",
        "kv_landing",
        params={"requests": output("steer_a", "requests")},
    )
    case.step(
        "steer_a_placement",
        "kv_same",
        params={"first": output("steer_a_holder", "engine"), "second": "prefill-0"},
    )
    case.step(
        "steer_a_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "restore_second",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "slow_first",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("steer_b_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "steer_b",
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
        "steer_b_done",
        "wait",
        timeout_s=15,
        params={"requests": output("steer_b", "requests")},
    )
    case.step(
        "steer_b_holder",
        "kv_landing",
        params={"requests": output("steer_b", "requests")},
    )
    case.step(
        "steer_b_placement",
        "kv_same",
        params={"first": output("steer_b_holder", "engine"), "second": "prefill-1"},
    )
    case.step(
        "restore_first",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("steer_restore_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "steer_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "steer_family_0",
        "kv_membership_check",
        params={
            "snapshot": output("steer_quiet", "snapshot"),
            "engine": "prefill-0",
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
        "steer_family_1",
        "kv_membership_check",
        params={
            "snapshot": output("steer_quiet", "snapshot"),
            "engine": "prefill-1",
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
            "relation": "all",
        },
    )
    case.step(
        "baseline_w0_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r0",
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
        "baseline_w0_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r0", "requests")},
    )
    case.step(
        "baseline_w0_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r0_before", "snapshot"),
            "requests": output("baseline_w0_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r1",
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
        "baseline_w0_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r1", "requests")},
    )
    case.step(
        "baseline_w0_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r1_before", "snapshot"),
            "requests": output("baseline_w0_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r2",
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
        "baseline_w0_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r2", "requests")},
    )
    case.step(
        "baseline_w0_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r2_before", "snapshot"),
            "requests": output("baseline_w0_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w0_r3",
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
        "baseline_w0_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w0_r3", "requests")},
    )
    case.step(
        "baseline_w0_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w0_r3_before", "snapshot"),
            "requests": output("baseline_w0_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w0_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("baseline_w0_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "baseline_w1_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r0",
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
        "baseline_w1_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r0", "requests")},
    )
    case.step(
        "baseline_w1_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r0_before", "snapshot"),
            "requests": output("baseline_w1_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r1",
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
        "baseline_w1_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r1", "requests")},
    )
    case.step(
        "baseline_w1_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r1_before", "snapshot"),
            "requests": output("baseline_w1_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r2",
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
        "baseline_w1_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r2", "requests")},
    )
    case.step(
        "baseline_w1_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r2_before", "snapshot"),
            "requests": output("baseline_w1_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w1_r3",
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
        "baseline_w1_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w1_r3", "requests")},
    )
    case.step(
        "baseline_w1_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w1_r3_before", "snapshot"),
            "requests": output("baseline_w1_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w1_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("baseline_w1_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "baseline_w2_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r0",
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
        "baseline_w2_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r0", "requests")},
    )
    case.step(
        "baseline_w2_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r0_before", "snapshot"),
            "requests": output("baseline_w2_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r1",
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
        "baseline_w2_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r1", "requests")},
    )
    case.step(
        "baseline_w2_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r1_before", "snapshot"),
            "requests": output("baseline_w2_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r2",
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
        "baseline_w2_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r2", "requests")},
    )
    case.step(
        "baseline_w2_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r2_before", "snapshot"),
            "requests": output("baseline_w2_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "baseline_w2_r3",
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
        "baseline_w2_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("baseline_w2_r3", "requests")},
    )
    case.step(
        "baseline_w2_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("baseline_w2_r3_before", "snapshot"),
            "requests": output("baseline_w2_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "baseline_w2_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("baseline_w2_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "baseline_digest",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturate_leader",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("saturation_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "saturation_w0_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r0",
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
    case.step("saturation_w0_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w0_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r1",
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
    case.step("saturation_w0_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w0_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r2",
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
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w0_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w0_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w0_r3",
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
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w0_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r0", "requests")},
    )
    case.step(
        "saturation_w0_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r0_before", "snapshot"),
            "requests": output("saturation_w0_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r1", "requests")},
    )
    case.step(
        "saturation_w0_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r1_before", "snapshot"),
            "requests": output("saturation_w0_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r2", "requests")},
    )
    case.step(
        "saturation_w0_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r2_before", "snapshot"),
            "requests": output("saturation_w0_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w0_r3", "requests")},
    )
    case.step(
        "saturation_w0_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w0_r3_before", "snapshot"),
            "requests": output("saturation_w0_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w0_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r0",
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
    case.step("saturation_w1_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w1_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r1",
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
    case.step("saturation_w1_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w1_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r2",
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
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w1_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w1_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w1_r3",
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
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w1_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r0", "requests")},
    )
    case.step(
        "saturation_w1_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r0_before", "snapshot"),
            "requests": output("saturation_w1_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r1", "requests")},
    )
    case.step(
        "saturation_w1_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r1_before", "snapshot"),
            "requests": output("saturation_w1_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r2", "requests")},
    )
    case.step(
        "saturation_w1_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r2_before", "snapshot"),
            "requests": output("saturation_w1_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w1_r3", "requests")},
    )
    case.step(
        "saturation_w1_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w1_r3_before", "snapshot"),
            "requests": output("saturation_w1_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w1_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r0",
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
    case.step("saturation_w2_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w2_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r1",
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
    case.step("saturation_w2_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w2_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r2",
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
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w2_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w2_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w2_r3",
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
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w2_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r0", "requests")},
    )
    case.step(
        "saturation_w2_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r0_before", "snapshot"),
            "requests": output("saturation_w2_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r1", "requests")},
    )
    case.step(
        "saturation_w2_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r1_before", "snapshot"),
            "requests": output("saturation_w2_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r2", "requests")},
    )
    case.step(
        "saturation_w2_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r2_before", "snapshot"),
            "requests": output("saturation_w2_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w2_r3", "requests")},
    )
    case.step(
        "saturation_w2_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w2_r3_before", "snapshot"),
            "requests": output("saturation_w2_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w2_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r0",
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
    case.step("saturation_w3_r0_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w3_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r1",
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
    case.step("saturation_w3_r1_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w3_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r2",
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
            "stream_timeout_s": 30,
        },
    )
    case.step("saturation_w3_r2_spacing", "balance_pause", params={"seconds": 0.12})
    case.step(
        "saturation_w3_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "saturation_w3_r3",
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
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "saturation_w3_r0_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r0", "requests")},
    )
    case.step(
        "saturation_w3_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r0_before", "snapshot"),
            "requests": output("saturation_w3_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_r1_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r1", "requests")},
    )
    case.step(
        "saturation_w3_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r1_before", "snapshot"),
            "requests": output("saturation_w3_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_r2_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r2", "requests")},
    )
    case.step(
        "saturation_w3_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r2_before", "snapshot"),
            "requests": output("saturation_w3_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_r3_done",
        "wait",
        timeout_s=30,
        params={"requests": output("saturation_w3_r3", "requests")},
    )
    case.step(
        "saturation_w3_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("saturation_w3_r3_before", "snapshot"),
            "requests": output("saturation_w3_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "saturation_w3_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recover_leader",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("recovery_settle", "balance_pause", params={"seconds": 1.5})
    case.step(
        "saturation_quiet",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "recovery_w0_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r0",
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
        "recovery_w0_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r0", "requests")},
    )
    case.step(
        "recovery_w0_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r0_before", "snapshot"),
            "requests": output("recovery_w0_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r1",
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
        "recovery_w0_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r1", "requests")},
    )
    case.step(
        "recovery_w0_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r1_before", "snapshot"),
            "requests": output("recovery_w0_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r2",
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
        "recovery_w0_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r2", "requests")},
    )
    case.step(
        "recovery_w0_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r2_before", "snapshot"),
            "requests": output("recovery_w0_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w0_r3",
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
        "recovery_w0_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w0_r3", "requests")},
    )
    case.step(
        "recovery_w0_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w0_r3_before", "snapshot"),
            "requests": output("recovery_w0_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w0_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w0_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w1_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r0",
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
        "recovery_w1_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r0", "requests")},
    )
    case.step(
        "recovery_w1_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r0_before", "snapshot"),
            "requests": output("recovery_w1_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r1",
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
        "recovery_w1_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r1", "requests")},
    )
    case.step(
        "recovery_w1_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r1_before", "snapshot"),
            "requests": output("recovery_w1_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r2",
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
        "recovery_w1_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r2", "requests")},
    )
    case.step(
        "recovery_w1_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r2_before", "snapshot"),
            "requests": output("recovery_w1_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w1_r3",
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
        "recovery_w1_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w1_r3", "requests")},
    )
    case.step(
        "recovery_w1_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w1_r3_before", "snapshot"),
            "requests": output("recovery_w1_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w1_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w1_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w2_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r0",
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
        "recovery_w2_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r0", "requests")},
    )
    case.step(
        "recovery_w2_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r0_before", "snapshot"),
            "requests": output("recovery_w2_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r1",
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
        "recovery_w2_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r1", "requests")},
    )
    case.step(
        "recovery_w2_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r1_before", "snapshot"),
            "requests": output("recovery_w2_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r2",
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
        "recovery_w2_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r2", "requests")},
    )
    case.step(
        "recovery_w2_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r2_before", "snapshot"),
            "requests": output("recovery_w2_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w2_r3",
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
        "recovery_w2_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w2_r3", "requests")},
    )
    case.step(
        "recovery_w2_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w2_r3_before", "snapshot"),
            "requests": output("recovery_w2_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w2_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w2_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w3_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r0",
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
        "recovery_w3_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r0", "requests")},
    )
    case.step(
        "recovery_w3_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r0_before", "snapshot"),
            "requests": output("recovery_w3_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r1",
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
        "recovery_w3_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r1", "requests")},
    )
    case.step(
        "recovery_w3_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r1_before", "snapshot"),
            "requests": output("recovery_w3_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r2",
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
        "recovery_w3_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r2", "requests")},
    )
    case.step(
        "recovery_w3_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r2_before", "snapshot"),
            "requests": output("recovery_w3_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w3_r3",
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
        "recovery_w3_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w3_r3", "requests")},
    )
    case.step(
        "recovery_w3_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w3_r3_before", "snapshot"),
            "requests": output("recovery_w3_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w3_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w3_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w4_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r0",
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
        "recovery_w4_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r0", "requests")},
    )
    case.step(
        "recovery_w4_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r0_before", "snapshot"),
            "requests": output("recovery_w4_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r1",
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
        "recovery_w4_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r1", "requests")},
    )
    case.step(
        "recovery_w4_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r1_before", "snapshot"),
            "requests": output("recovery_w4_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r2",
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
        "recovery_w4_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r2", "requests")},
    )
    case.step(
        "recovery_w4_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r2_before", "snapshot"),
            "requests": output("recovery_w4_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w4_r3",
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
        "recovery_w4_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w4_r3", "requests")},
    )
    case.step(
        "recovery_w4_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w4_r3_before", "snapshot"),
            "requests": output("recovery_w4_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w4_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w4_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_w5_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r0",
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
        "recovery_w5_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r0", "requests")},
    )
    case.step(
        "recovery_w5_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r0_before", "snapshot"),
            "requests": output("recovery_w5_r0", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r1",
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
        "recovery_w5_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r1", "requests")},
    )
    case.step(
        "recovery_w5_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r1_before", "snapshot"),
            "requests": output("recovery_w5_r1", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r2",
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
        "recovery_w5_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r2", "requests")},
    )
    case.step(
        "recovery_w5_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r2_before", "snapshot"),
            "requests": output("recovery_w5_r2", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "recovery_w5_r3",
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
        "recovery_w5_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("recovery_w5_r3", "requests")},
    )
    case.step(
        "recovery_w5_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("recovery_w5_r3_before", "snapshot"),
            "requests": output("recovery_w5_r3", "requests"),
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
            "min_contiguous": 8,
        },
    )
    case.step(
        "recovery_w5_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step("recovery_w5_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "recovery_digest",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "phase_observations",
        "kv_phase_observation",
        params={
            "saturation": [
                output("saturation_w0_r0_hit", "sample"),
                output("saturation_w0_r1_hit", "sample"),
                output("saturation_w0_r2_hit", "sample"),
                output("saturation_w0_r3_hit", "sample"),
                output("saturation_w1_r0_hit", "sample"),
                output("saturation_w1_r1_hit", "sample"),
                output("saturation_w1_r2_hit", "sample"),
                output("saturation_w1_r3_hit", "sample"),
                output("saturation_w2_r0_hit", "sample"),
                output("saturation_w2_r1_hit", "sample"),
                output("saturation_w2_r2_hit", "sample"),
                output("saturation_w2_r3_hit", "sample"),
                output("saturation_w3_r0_hit", "sample"),
                output("saturation_w3_r1_hit", "sample"),
                output("saturation_w3_r2_hit", "sample"),
                output("saturation_w3_r3_hit", "sample"),
            ],
            "recovery_windows": [
                [
                    output("recovery_w0_r0_hit", "sample"),
                    output("recovery_w0_r1_hit", "sample"),
                    output("recovery_w0_r2_hit", "sample"),
                    output("recovery_w0_r3_hit", "sample"),
                ],
                [
                    output("recovery_w1_r0_hit", "sample"),
                    output("recovery_w1_r1_hit", "sample"),
                    output("recovery_w1_r2_hit", "sample"),
                    output("recovery_w1_r3_hit", "sample"),
                ],
                [
                    output("recovery_w2_r0_hit", "sample"),
                    output("recovery_w2_r1_hit", "sample"),
                    output("recovery_w2_r2_hit", "sample"),
                    output("recovery_w2_r3_hit", "sample"),
                ],
                [
                    output("recovery_w3_r0_hit", "sample"),
                    output("recovery_w3_r1_hit", "sample"),
                    output("recovery_w3_r2_hit", "sample"),
                    output("recovery_w3_r3_hit", "sample"),
                ],
                [
                    output("recovery_w4_r0_hit", "sample"),
                    output("recovery_w4_r1_hit", "sample"),
                    output("recovery_w4_r2_hit", "sample"),
                    output("recovery_w4_r3_hit", "sample"),
                ],
                [
                    output("recovery_w5_r0_hit", "sample"),
                    output("recovery_w5_r1_hit", "sample"),
                    output("recovery_w5_r2_hit", "sample"),
                    output("recovery_w5_r3_hit", "sample"),
                ],
            ],
            "family_keys": [
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
            "leader": "prefill-0",
            "ready_rate": 0.75,
            "phase_snapshots": {
                "steer": output("steer_quiet", "snapshot"),
                "baseline": output("baseline_digest", "snapshot"),
                "saturation": output("saturation_quiet", "snapshot"),
                "recovery": output("recovery_digest", "snapshot"),
            },
            "families": [
                [
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
                [
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
            ],
        },
    )
    case.step(
        "holder_flips",
        "kv_window_transitions",
        params={
            "snapshots": [
                output("baseline_w0_end", "snapshot"),
                output("baseline_w1_end", "snapshot"),
                output("baseline_w2_end", "snapshot"),
                output("saturation_w0_end", "snapshot"),
                output("saturation_w1_end", "snapshot"),
                output("saturation_w2_end", "snapshot"),
                output("saturation_w3_end", "snapshot"),
                output("recovery_w0_end", "snapshot"),
                output("recovery_w1_end", "snapshot"),
                output("recovery_w2_end", "snapshot"),
                output("recovery_w3_end", "snapshot"),
                output("recovery_w4_end", "snapshot"),
                output("recovery_w5_end", "snapshot"),
            ],
            "families": [
                [
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
                [
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
            ],
        },
    )
    case.step(
        "all_phase_requests",
        "kv_hit_completeness",
        params={
            "samples": [
                output("baseline_w0_r0_hit", "sample"),
                output("baseline_w0_r1_hit", "sample"),
                output("baseline_w0_r2_hit", "sample"),
                output("baseline_w0_r3_hit", "sample"),
                output("baseline_w1_r0_hit", "sample"),
                output("baseline_w1_r1_hit", "sample"),
                output("baseline_w1_r2_hit", "sample"),
                output("baseline_w1_r3_hit", "sample"),
                output("baseline_w2_r0_hit", "sample"),
                output("baseline_w2_r1_hit", "sample"),
                output("baseline_w2_r2_hit", "sample"),
                output("baseline_w2_r3_hit", "sample"),
                output("saturation_w0_r0_hit", "sample"),
                output("saturation_w0_r1_hit", "sample"),
                output("saturation_w0_r2_hit", "sample"),
                output("saturation_w0_r3_hit", "sample"),
                output("saturation_w1_r0_hit", "sample"),
                output("saturation_w1_r1_hit", "sample"),
                output("saturation_w1_r2_hit", "sample"),
                output("saturation_w1_r3_hit", "sample"),
                output("saturation_w2_r0_hit", "sample"),
                output("saturation_w2_r1_hit", "sample"),
                output("saturation_w2_r2_hit", "sample"),
                output("saturation_w2_r3_hit", "sample"),
                output("saturation_w3_r0_hit", "sample"),
                output("saturation_w3_r1_hit", "sample"),
                output("saturation_w3_r2_hit", "sample"),
                output("saturation_w3_r3_hit", "sample"),
                output("recovery_w0_r0_hit", "sample"),
                output("recovery_w0_r1_hit", "sample"),
                output("recovery_w0_r2_hit", "sample"),
                output("recovery_w0_r3_hit", "sample"),
                output("recovery_w1_r0_hit", "sample"),
                output("recovery_w1_r1_hit", "sample"),
                output("recovery_w1_r2_hit", "sample"),
                output("recovery_w1_r3_hit", "sample"),
                output("recovery_w2_r0_hit", "sample"),
                output("recovery_w2_r1_hit", "sample"),
                output("recovery_w2_r2_hit", "sample"),
                output("recovery_w2_r3_hit", "sample"),
                output("recovery_w3_r0_hit", "sample"),
                output("recovery_w3_r1_hit", "sample"),
                output("recovery_w3_r2_hit", "sample"),
                output("recovery_w3_r3_hit", "sample"),
                output("recovery_w4_r0_hit", "sample"),
                output("recovery_w4_r1_hit", "sample"),
                output("recovery_w4_r2_hit", "sample"),
                output("recovery_w4_r3_hit", "sample"),
                output("recovery_w5_r0_hit", "sample"),
                output("recovery_w5_r1_hit", "sample"),
                output("recovery_w5_r2_hit", "sample"),
                output("recovery_w5_r3_hit", "sample"),
            ],
            "min_samples": 52,
        },
    )
    case.step(
        "baseline_hit",
        "kv_hit_rate_check",
        params={
            "samples": [
                output("baseline_w0_r0_hit", "sample"),
                output("baseline_w0_r1_hit", "sample"),
                output("baseline_w0_r2_hit", "sample"),
                output("baseline_w0_r3_hit", "sample"),
                output("baseline_w1_r0_hit", "sample"),
                output("baseline_w1_r1_hit", "sample"),
                output("baseline_w1_r2_hit", "sample"),
                output("baseline_w1_r3_hit", "sample"),
                output("baseline_w2_r0_hit", "sample"),
                output("baseline_w2_r1_hit", "sample"),
                output("baseline_w2_r2_hit", "sample"),
                output("baseline_w2_r3_hit", "sample"),
            ],
            "min_samples": 12,
            "bands": {"strict": 0.9, "normal": 0.85, "loose": 0.8},
        },
    )
    case.step(
        "saturation_hit",
        "kv_hit_rate_check",
        params={
            "samples": [
                output("saturation_w0_r0_hit", "sample"),
                output("saturation_w0_r1_hit", "sample"),
                output("saturation_w0_r2_hit", "sample"),
                output("saturation_w0_r3_hit", "sample"),
                output("saturation_w1_r0_hit", "sample"),
                output("saturation_w1_r1_hit", "sample"),
                output("saturation_w1_r2_hit", "sample"),
                output("saturation_w1_r3_hit", "sample"),
                output("saturation_w2_r0_hit", "sample"),
                output("saturation_w2_r1_hit", "sample"),
                output("saturation_w2_r2_hit", "sample"),
                output("saturation_w2_r3_hit", "sample"),
                output("saturation_w3_r0_hit", "sample"),
                output("saturation_w3_r1_hit", "sample"),
                output("saturation_w3_r2_hit", "sample"),
                output("saturation_w3_r3_hit", "sample"),
            ],
            "min_samples": 16,
            "bands": {"strict": 0.5, "normal": 0.4, "loose": 0.3},
        },
    )
    case.step(
        "recovery_steady_hit",
        "kv_hit_rate_check",
        params={
            "samples": [
                output("recovery_w4_r0_hit", "sample"),
                output("recovery_w4_r1_hit", "sample"),
                output("recovery_w4_r2_hit", "sample"),
                output("recovery_w4_r3_hit", "sample"),
                output("recovery_w5_r0_hit", "sample"),
                output("recovery_w5_r1_hit", "sample"),
                output("recovery_w5_r2_hit", "sample"),
                output("recovery_w5_r3_hit", "sample"),
            ],
            "min_samples": 8,
            "bands": {"strict": 0.85, "normal": 0.8, "loose": 0.75},
        },
    )
    case.step(
        "replication",
        "kv_replication_check",
        params={
            "snapshot": output("recovery_w5_end", "snapshot"),
            "families": [
                [
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
                [
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
            ],
            "bands": {"strict": 1.5, "normal": 1.75, "loose": 2},
            "max_holders": 2,
        },
    )
    case.step("cleanup", "teardown")


def leader_spill_multi(case):
    from ..scenario.actions.cache_storm import CONFIG_LIMITS

    config = {
        name: case.number(name, default, minimum=low, maximum=high)
        for name, (default, low, high) in CONFIG_LIMITS.items()
    }
    config["interval_s"] = case.number(
        "interval_s", 0.15, minimum=0.05, maximum=2, integer=False
    )
    config["steady_interval_s"] = case.number(
        "steady_interval_s", 0.4, minimum=0.1, maximum=2, integer=False
    )
    config["min_busy_share"] = case.number(
        "min_busy_share", 0.9, minimum=0, maximum=1, integer=False
    )
    for metric, default, maximum in (
        ("hit", 1, 1),
        ("eviction", 0, 10000),
        ("holders", 1, 32),
    ):
        for grade in ("strict", "normal", "loose"):
            key = f"{metric}_{grade}"
            config[key] = case.number(
                key, default, minimum=0, maximum=maximum, integer=False
            )
    if config["background_blocks"] > config["receiver_retention_blocks"]:
        raise ValueError("background prefix must fit receiver retention")
    if config["normal_ms"] / 1000 >= config["interval_s"]:
        raise ValueError("background cadence must allow normal execution to finish")
    if case.environment["n_prefill"] < 2 or not 1 <= case.environment["n_decode"] <= 2:
        raise ValueError("storm requires multiple prefills and one or two decodes")
    if case.environment.get("prefill_cache_blocks", 0) <= config["hot_blocks"]:
        raise ValueError("physical execution pool must fit hot prefix plus reserve")
    case.step("setup", "setup", timeout_s=180)
    case.step("prepare", "storm_prepare", timeout_s=90, params=config)
    handle = {"storm": output("prepare", "storm")}
    for phase in ("baseline", "saturation", "recovery"):
        if phase == "saturation":
            case.step("baseline_drain", "storm_drain", params=handle, timeout_s=60)
            case.step("slow_leader", "storm_speed", params={**handle, "slow": True})
        if phase == "recovery":
            case.step("restore_leader", "storm_speed", params={**handle, "slow": False})
        for index in range(config[f"{phase}_windows"]):
            case.step(
                f"{phase}_{index}",
                "storm_window",
                timeout_s=config["window_s"] + 5,
                params={**handle, "phase": phase, "index": index},
            )
    case.step("all_requests", "storm_drain", params=handle, timeout_s=65)
    case.step("validity", "storm_validate", params=handle, timeout_s=30)
    case.step("healthy", "storm_health", params=handle)


VARIANTS = {
    "leader_spill_multi": {
        "build": leader_spill_multi,
        "profiles": ["single-batch"],
        "metadata": {
            "findings": ["healthy.hit", "healthy.eviction", "healthy.holders"]
        },
    },
    "prefix_batch": {
        "build": prefix_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "prefix_nonbatch": {
        "build": prefix_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "hot_tension": {
        "build": hot_tension,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "mixed_tiers": {
        "build": mixed_tiers,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "leader_spill_batch": {
        "build": leader_spill_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "findings": ["saturation_hit.M3"],
            "requires": ["enqueue_batch"],
        },
    },
    "leader_spill_nonbatch": {
        "build": leader_spill_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {
            "findings": ["saturation_hit.M3"],
        },
    },
}
