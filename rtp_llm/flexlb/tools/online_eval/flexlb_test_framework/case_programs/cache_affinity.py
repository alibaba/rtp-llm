"""Prefix fidelity and free-flow diversity, hot holder total-share tension, and full/half/zero-hit tier contrast."""

from ..case_config import output


def prefix_batch(case):
    case.step("setup", "setup", timeout_s=case.value("prefix_batch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("prefix_batch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("prefix_batch.slow"),
    )
    case.step("perf_sync", "balance_pause", params=case.value("prefix_batch.perf_sync"))
    case.step(
        "seed_first",
        "request",
        params=case.value("prefix_batch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("prefix_batch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "prefix_batch.first_holder", {"requests": output("seed_first", "requests")}
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("prefix_batch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.second_terminal_timeout_s"),
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
        timeout_s=case.value("prefix_batch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("prefix_batch.restore_perf"),
    )
    case.step(
        "cache_sync", "balance_pause", params=case.value("prefix_batch.cache_sync")
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
        "mixed_0",
        "request",
        params=case.value("prefix_batch.mixed_0"),
    )
    case.step(
        "mixed_0_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_0_terminal_timeout_s"),
        params={"requests": output("mixed_0", "requests")},
    )
    case.step(
        "mixed_1",
        "request",
        params=case.value("prefix_batch.mixed_1"),
    )
    case.step(
        "mixed_1_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_1_terminal_timeout_s"),
        params={"requests": output("mixed_1", "requests")},
    )
    case.step(
        "mixed_2",
        "request",
        params=case.value("prefix_batch.mixed_2"),
    )
    case.step(
        "mixed_2_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_2_terminal_timeout_s"),
        params={"requests": output("mixed_2", "requests")},
    )
    case.step(
        "mixed_3",
        "request",
        params=case.value("prefix_batch.mixed_3"),
    )
    case.step(
        "mixed_3_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_3_terminal_timeout_s"),
        params={"requests": output("mixed_3", "requests")},
    )
    case.step(
        "mixed_4",
        "request",
        params=case.value("prefix_batch.mixed_4"),
    )
    case.step(
        "mixed_4_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_4_terminal_timeout_s"),
        params={"requests": output("mixed_4", "requests")},
    )
    case.step(
        "mixed_5",
        "request",
        params=case.value("prefix_batch.mixed_5"),
    )
    case.step(
        "mixed_5_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_5_terminal_timeout_s"),
        params={"requests": output("mixed_5", "requests")},
    )
    case.step(
        "mixed_6",
        "request",
        params=case.value("prefix_batch.mixed_6"),
    )
    case.step(
        "mixed_6_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_6_terminal_timeout_s"),
        params={"requests": output("mixed_6", "requests")},
    )
    case.step(
        "mixed_7",
        "request",
        params=case.value("prefix_batch.mixed_7"),
    )
    case.step(
        "mixed_7_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_7_terminal_timeout_s"),
        params={"requests": output("mixed_7", "requests")},
    )
    case.step(
        "mixed_8",
        "request",
        params=case.value("prefix_batch.mixed_8"),
    )
    case.step(
        "mixed_8_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_8_terminal_timeout_s"),
        params={"requests": output("mixed_8", "requests")},
    )
    case.step(
        "mixed_9",
        "request",
        params=case.value("prefix_batch.mixed_9"),
    )
    case.step(
        "mixed_9_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_9_terminal_timeout_s"),
        params={"requests": output("mixed_9", "requests")},
    )
    case.step(
        "mixed_10",
        "request",
        params=case.value("prefix_batch.mixed_10"),
    )
    case.step(
        "mixed_10_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_10_terminal_timeout_s"),
        params={"requests": output("mixed_10", "requests")},
    )
    case.step(
        "mixed_11",
        "request",
        params=case.value("prefix_batch.mixed_11"),
    )
    case.step(
        "mixed_11_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_11_terminal_timeout_s"),
        params={"requests": output("mixed_11", "requests")},
    )
    case.step(
        "mixed_12",
        "request",
        params=case.value("prefix_batch.mixed_12"),
    )
    case.step(
        "mixed_12_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_12_terminal_timeout_s"),
        params={"requests": output("mixed_12", "requests")},
    )
    case.step(
        "mixed_13",
        "request",
        params=case.value("prefix_batch.mixed_13"),
    )
    case.step(
        "mixed_13_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_13_terminal_timeout_s"),
        params={"requests": output("mixed_13", "requests")},
    )
    case.step(
        "mixed_14",
        "request",
        params=case.value("prefix_batch.mixed_14"),
    )
    case.step(
        "mixed_14_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_14_terminal_timeout_s"),
        params={"requests": output("mixed_14", "requests")},
    )
    case.step(
        "mixed_15",
        "request",
        params=case.value("prefix_batch.mixed_15"),
    )
    case.step(
        "mixed_15_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_15_terminal_timeout_s"),
        params={"requests": output("mixed_15", "requests")},
    )
    case.step(
        "mixed_16",
        "request",
        params=case.value("prefix_batch.mixed_16"),
    )
    case.step(
        "mixed_16_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_16_terminal_timeout_s"),
        params={"requests": output("mixed_16", "requests")},
    )
    case.step(
        "mixed_17",
        "request",
        params=case.value("prefix_batch.mixed_17"),
    )
    case.step(
        "mixed_17_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_17_terminal_timeout_s"),
        params={"requests": output("mixed_17", "requests")},
    )
    case.step(
        "mixed_18",
        "request",
        params=case.value("prefix_batch.mixed_18"),
    )
    case.step(
        "mixed_18_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_18_terminal_timeout_s"),
        params={"requests": output("mixed_18", "requests")},
    )
    case.step(
        "mixed_19",
        "request",
        params=case.value("prefix_batch.mixed_19"),
    )
    case.step(
        "mixed_19_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_19_terminal_timeout_s"),
        params={"requests": output("mixed_19", "requests")},
    )
    case.step(
        "mixed_20",
        "request",
        params=case.value("prefix_batch.mixed_20"),
    )
    case.step(
        "mixed_20_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_20_terminal_timeout_s"),
        params={"requests": output("mixed_20", "requests")},
    )
    case.step(
        "mixed_21",
        "request",
        params=case.value("prefix_batch.mixed_21"),
    )
    case.step(
        "mixed_21_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_21_terminal_timeout_s"),
        params={"requests": output("mixed_21", "requests")},
    )
    case.step(
        "mixed_22",
        "request",
        params=case.value("prefix_batch.mixed_22"),
    )
    case.step(
        "mixed_22_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_22_terminal_timeout_s"),
        params={"requests": output("mixed_22", "requests")},
    )
    case.step(
        "mixed_23",
        "request",
        params=case.value("prefix_batch.mixed_23"),
    )
    case.step(
        "mixed_23_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_23_terminal_timeout_s"),
        params={"requests": output("mixed_23", "requests")},
    )
    case.step(
        "mixed_24",
        "request",
        params=case.value("prefix_batch.mixed_24"),
    )
    case.step(
        "mixed_24_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_24_terminal_timeout_s"),
        params={"requests": output("mixed_24", "requests")},
    )
    case.step(
        "mixed_25",
        "request",
        params=case.value("prefix_batch.mixed_25"),
    )
    case.step(
        "mixed_25_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_25_terminal_timeout_s"),
        params={"requests": output("mixed_25", "requests")},
    )
    case.step(
        "mixed_26",
        "request",
        params=case.value("prefix_batch.mixed_26"),
    )
    case.step(
        "mixed_26_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_26_terminal_timeout_s"),
        params={"requests": output("mixed_26", "requests")},
    )
    case.step(
        "mixed_27",
        "request",
        params=case.value("prefix_batch.mixed_27"),
    )
    case.step(
        "mixed_27_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_27_terminal_timeout_s"),
        params={"requests": output("mixed_27", "requests")},
    )
    case.step(
        "mixed_28",
        "request",
        params=case.value("prefix_batch.mixed_28"),
    )
    case.step(
        "mixed_28_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_28_terminal_timeout_s"),
        params={"requests": output("mixed_28", "requests")},
    )
    case.step(
        "mixed_29",
        "request",
        params=case.value("prefix_batch.mixed_29"),
    )
    case.step(
        "mixed_29_terminal",
        "wait",
        timeout_s=case.value("prefix_batch.mixed_29_terminal_timeout_s"),
        params={"requests": output("mixed_29", "requests")},
    )
    case.step(
        "family_fidelity",
        "kv_fidelity_check",
        params=case.params(
            "prefix_batch.family_fidelity",
            {
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
            },
        ),
    )
    case.step(
        "free_spread",
        "kv_union_check",
        params=case.params(
            "prefix_batch.free_spread",
            {
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
                ]
            },
        ),
    )
    case.step("cleanup", "teardown")


def prefix_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("prefix_nonbatch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("prefix_nonbatch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("prefix_nonbatch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("prefix_nonbatch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("prefix_nonbatch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("prefix_nonbatch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "prefix_nonbatch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("prefix_nonbatch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.second_terminal_timeout_s"),
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
        timeout_s=case.value("prefix_nonbatch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("prefix_nonbatch.restore_perf"),
    )
    case.step(
        "cache_sync", "balance_pause", params=case.value("prefix_nonbatch.cache_sync")
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
        "mixed_0",
        "request",
        params=case.value("prefix_nonbatch.mixed_0"),
    )
    case.step(
        "mixed_0_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_0_terminal_timeout_s"),
        params={"requests": output("mixed_0", "requests")},
    )
    case.step(
        "mixed_1",
        "request",
        params=case.value("prefix_nonbatch.mixed_1"),
    )
    case.step(
        "mixed_1_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_1_terminal_timeout_s"),
        params={"requests": output("mixed_1", "requests")},
    )
    case.step(
        "mixed_2",
        "request",
        params=case.value("prefix_nonbatch.mixed_2"),
    )
    case.step(
        "mixed_2_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_2_terminal_timeout_s"),
        params={"requests": output("mixed_2", "requests")},
    )
    case.step(
        "mixed_3",
        "request",
        params=case.value("prefix_nonbatch.mixed_3"),
    )
    case.step(
        "mixed_3_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_3_terminal_timeout_s"),
        params={"requests": output("mixed_3", "requests")},
    )
    case.step(
        "mixed_4",
        "request",
        params=case.value("prefix_nonbatch.mixed_4"),
    )
    case.step(
        "mixed_4_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_4_terminal_timeout_s"),
        params={"requests": output("mixed_4", "requests")},
    )
    case.step(
        "mixed_5",
        "request",
        params=case.value("prefix_nonbatch.mixed_5"),
    )
    case.step(
        "mixed_5_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_5_terminal_timeout_s"),
        params={"requests": output("mixed_5", "requests")},
    )
    case.step(
        "mixed_6",
        "request",
        params=case.value("prefix_nonbatch.mixed_6"),
    )
    case.step(
        "mixed_6_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_6_terminal_timeout_s"),
        params={"requests": output("mixed_6", "requests")},
    )
    case.step(
        "mixed_7",
        "request",
        params=case.value("prefix_nonbatch.mixed_7"),
    )
    case.step(
        "mixed_7_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_7_terminal_timeout_s"),
        params={"requests": output("mixed_7", "requests")},
    )
    case.step(
        "mixed_8",
        "request",
        params=case.value("prefix_nonbatch.mixed_8"),
    )
    case.step(
        "mixed_8_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_8_terminal_timeout_s"),
        params={"requests": output("mixed_8", "requests")},
    )
    case.step(
        "mixed_9",
        "request",
        params=case.value("prefix_nonbatch.mixed_9"),
    )
    case.step(
        "mixed_9_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_9_terminal_timeout_s"),
        params={"requests": output("mixed_9", "requests")},
    )
    case.step(
        "mixed_10",
        "request",
        params=case.value("prefix_nonbatch.mixed_10"),
    )
    case.step(
        "mixed_10_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_10_terminal_timeout_s"),
        params={"requests": output("mixed_10", "requests")},
    )
    case.step(
        "mixed_11",
        "request",
        params=case.value("prefix_nonbatch.mixed_11"),
    )
    case.step(
        "mixed_11_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_11_terminal_timeout_s"),
        params={"requests": output("mixed_11", "requests")},
    )
    case.step(
        "mixed_12",
        "request",
        params=case.value("prefix_nonbatch.mixed_12"),
    )
    case.step(
        "mixed_12_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_12_terminal_timeout_s"),
        params={"requests": output("mixed_12", "requests")},
    )
    case.step(
        "mixed_13",
        "request",
        params=case.value("prefix_nonbatch.mixed_13"),
    )
    case.step(
        "mixed_13_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_13_terminal_timeout_s"),
        params={"requests": output("mixed_13", "requests")},
    )
    case.step(
        "mixed_14",
        "request",
        params=case.value("prefix_nonbatch.mixed_14"),
    )
    case.step(
        "mixed_14_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_14_terminal_timeout_s"),
        params={"requests": output("mixed_14", "requests")},
    )
    case.step(
        "mixed_15",
        "request",
        params=case.value("prefix_nonbatch.mixed_15"),
    )
    case.step(
        "mixed_15_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_15_terminal_timeout_s"),
        params={"requests": output("mixed_15", "requests")},
    )
    case.step(
        "mixed_16",
        "request",
        params=case.value("prefix_nonbatch.mixed_16"),
    )
    case.step(
        "mixed_16_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_16_terminal_timeout_s"),
        params={"requests": output("mixed_16", "requests")},
    )
    case.step(
        "mixed_17",
        "request",
        params=case.value("prefix_nonbatch.mixed_17"),
    )
    case.step(
        "mixed_17_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_17_terminal_timeout_s"),
        params={"requests": output("mixed_17", "requests")},
    )
    case.step(
        "mixed_18",
        "request",
        params=case.value("prefix_nonbatch.mixed_18"),
    )
    case.step(
        "mixed_18_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_18_terminal_timeout_s"),
        params={"requests": output("mixed_18", "requests")},
    )
    case.step(
        "mixed_19",
        "request",
        params=case.value("prefix_nonbatch.mixed_19"),
    )
    case.step(
        "mixed_19_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_19_terminal_timeout_s"),
        params={"requests": output("mixed_19", "requests")},
    )
    case.step(
        "mixed_20",
        "request",
        params=case.value("prefix_nonbatch.mixed_20"),
    )
    case.step(
        "mixed_20_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_20_terminal_timeout_s"),
        params={"requests": output("mixed_20", "requests")},
    )
    case.step(
        "mixed_21",
        "request",
        params=case.value("prefix_nonbatch.mixed_21"),
    )
    case.step(
        "mixed_21_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_21_terminal_timeout_s"),
        params={"requests": output("mixed_21", "requests")},
    )
    case.step(
        "mixed_22",
        "request",
        params=case.value("prefix_nonbatch.mixed_22"),
    )
    case.step(
        "mixed_22_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_22_terminal_timeout_s"),
        params={"requests": output("mixed_22", "requests")},
    )
    case.step(
        "mixed_23",
        "request",
        params=case.value("prefix_nonbatch.mixed_23"),
    )
    case.step(
        "mixed_23_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_23_terminal_timeout_s"),
        params={"requests": output("mixed_23", "requests")},
    )
    case.step(
        "mixed_24",
        "request",
        params=case.value("prefix_nonbatch.mixed_24"),
    )
    case.step(
        "mixed_24_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_24_terminal_timeout_s"),
        params={"requests": output("mixed_24", "requests")},
    )
    case.step(
        "mixed_25",
        "request",
        params=case.value("prefix_nonbatch.mixed_25"),
    )
    case.step(
        "mixed_25_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_25_terminal_timeout_s"),
        params={"requests": output("mixed_25", "requests")},
    )
    case.step(
        "mixed_26",
        "request",
        params=case.value("prefix_nonbatch.mixed_26"),
    )
    case.step(
        "mixed_26_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_26_terminal_timeout_s"),
        params={"requests": output("mixed_26", "requests")},
    )
    case.step(
        "mixed_27",
        "request",
        params=case.value("prefix_nonbatch.mixed_27"),
    )
    case.step(
        "mixed_27_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_27_terminal_timeout_s"),
        params={"requests": output("mixed_27", "requests")},
    )
    case.step(
        "mixed_28",
        "request",
        params=case.value("prefix_nonbatch.mixed_28"),
    )
    case.step(
        "mixed_28_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_28_terminal_timeout_s"),
        params={"requests": output("mixed_28", "requests")},
    )
    case.step(
        "mixed_29",
        "request",
        params=case.value("prefix_nonbatch.mixed_29"),
    )
    case.step(
        "mixed_29_terminal",
        "wait",
        timeout_s=case.value("prefix_nonbatch.mixed_29_terminal_timeout_s"),
        params={"requests": output("mixed_29", "requests")},
    )
    case.step(
        "family_fidelity",
        "kv_fidelity_check",
        params=case.params(
            "prefix_nonbatch.family_fidelity",
            {
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
            },
        ),
    )
    case.step(
        "free_spread",
        "kv_union_check",
        params=case.params(
            "prefix_nonbatch.free_spread",
            {
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
                ]
            },
        ),
    )
    case.step("cleanup", "teardown")


def hot_tension(case):
    case.step("setup", "setup", timeout_s=case.value("hot_tension.setup_timeout_s"))
    case.step(
        "seed",
        "request",
        params=case.value("hot_tension.seed"),
    )
    case.step(
        "seed_terminal",
        "wait",
        timeout_s=case.value("hot_tension.seed_terminal_timeout_s"),
        params={"requests": output("seed", "requests")},
    )
    case.step("holder", "kv_landing", params={"requests": output("seed", "requests")})
    case.step(
        "cache_sync", "balance_pause", params=case.value("hot_tension.cache_sync")
    )
    case.step(
        "mixed_0",
        "request",
        params=case.value("hot_tension.mixed_0"),
    )
    case.step(
        "mixed_0_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_0_terminal_timeout_s"),
        params={"requests": output("mixed_0", "requests")},
    )
    case.step(
        "mixed_1",
        "request",
        params=case.value("hot_tension.mixed_1"),
    )
    case.step(
        "mixed_1_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_1_terminal_timeout_s"),
        params={"requests": output("mixed_1", "requests")},
    )
    case.step(
        "mixed_2",
        "request",
        params=case.value("hot_tension.mixed_2"),
    )
    case.step(
        "mixed_2_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_2_terminal_timeout_s"),
        params={"requests": output("mixed_2", "requests")},
    )
    case.step(
        "mixed_3",
        "request",
        params=case.value("hot_tension.mixed_3"),
    )
    case.step(
        "mixed_3_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_3_terminal_timeout_s"),
        params={"requests": output("mixed_3", "requests")},
    )
    case.step(
        "mixed_4",
        "request",
        params=case.value("hot_tension.mixed_4"),
    )
    case.step(
        "mixed_4_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_4_terminal_timeout_s"),
        params={"requests": output("mixed_4", "requests")},
    )
    case.step(
        "mixed_5",
        "request",
        params=case.value("hot_tension.mixed_5"),
    )
    case.step(
        "mixed_5_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_5_terminal_timeout_s"),
        params={"requests": output("mixed_5", "requests")},
    )
    case.step(
        "mixed_6",
        "request",
        params=case.value("hot_tension.mixed_6"),
    )
    case.step(
        "mixed_6_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_6_terminal_timeout_s"),
        params={"requests": output("mixed_6", "requests")},
    )
    case.step(
        "mixed_7",
        "request",
        params=case.value("hot_tension.mixed_7"),
    )
    case.step(
        "mixed_7_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_7_terminal_timeout_s"),
        params={"requests": output("mixed_7", "requests")},
    )
    case.step(
        "mixed_8",
        "request",
        params=case.value("hot_tension.mixed_8"),
    )
    case.step(
        "mixed_8_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_8_terminal_timeout_s"),
        params={"requests": output("mixed_8", "requests")},
    )
    case.step(
        "mixed_9",
        "request",
        params=case.value("hot_tension.mixed_9"),
    )
    case.step(
        "mixed_9_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_9_terminal_timeout_s"),
        params={"requests": output("mixed_9", "requests")},
    )
    case.step(
        "mixed_10",
        "request",
        params=case.value("hot_tension.mixed_10"),
    )
    case.step(
        "mixed_10_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_10_terminal_timeout_s"),
        params={"requests": output("mixed_10", "requests")},
    )
    case.step(
        "mixed_11",
        "request",
        params=case.value("hot_tension.mixed_11"),
    )
    case.step(
        "mixed_11_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_11_terminal_timeout_s"),
        params={"requests": output("mixed_11", "requests")},
    )
    case.step(
        "mixed_12",
        "request",
        params=case.value("hot_tension.mixed_12"),
    )
    case.step(
        "mixed_12_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_12_terminal_timeout_s"),
        params={"requests": output("mixed_12", "requests")},
    )
    case.step(
        "mixed_13",
        "request",
        params=case.value("hot_tension.mixed_13"),
    )
    case.step(
        "mixed_13_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_13_terminal_timeout_s"),
        params={"requests": output("mixed_13", "requests")},
    )
    case.step(
        "mixed_14",
        "request",
        params=case.value("hot_tension.mixed_14"),
    )
    case.step(
        "mixed_14_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_14_terminal_timeout_s"),
        params={"requests": output("mixed_14", "requests")},
    )
    case.step(
        "mixed_15",
        "request",
        params=case.value("hot_tension.mixed_15"),
    )
    case.step(
        "mixed_15_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_15_terminal_timeout_s"),
        params={"requests": output("mixed_15", "requests")},
    )
    case.step(
        "mixed_16",
        "request",
        params=case.value("hot_tension.mixed_16"),
    )
    case.step(
        "mixed_16_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_16_terminal_timeout_s"),
        params={"requests": output("mixed_16", "requests")},
    )
    case.step(
        "mixed_17",
        "request",
        params=case.value("hot_tension.mixed_17"),
    )
    case.step(
        "mixed_17_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_17_terminal_timeout_s"),
        params={"requests": output("mixed_17", "requests")},
    )
    case.step(
        "mixed_18",
        "request",
        params=case.value("hot_tension.mixed_18"),
    )
    case.step(
        "mixed_18_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_18_terminal_timeout_s"),
        params={"requests": output("mixed_18", "requests")},
    )
    case.step(
        "mixed_19",
        "request",
        params=case.value("hot_tension.mixed_19"),
    )
    case.step(
        "mixed_19_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_19_terminal_timeout_s"),
        params={"requests": output("mixed_19", "requests")},
    )
    case.step(
        "mixed_20",
        "request",
        params=case.value("hot_tension.mixed_20"),
    )
    case.step(
        "mixed_20_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_20_terminal_timeout_s"),
        params={"requests": output("mixed_20", "requests")},
    )
    case.step(
        "mixed_21",
        "request",
        params=case.value("hot_tension.mixed_21"),
    )
    case.step(
        "mixed_21_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_21_terminal_timeout_s"),
        params={"requests": output("mixed_21", "requests")},
    )
    case.step(
        "mixed_22",
        "request",
        params=case.value("hot_tension.mixed_22"),
    )
    case.step(
        "mixed_22_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_22_terminal_timeout_s"),
        params={"requests": output("mixed_22", "requests")},
    )
    case.step(
        "mixed_23",
        "request",
        params=case.value("hot_tension.mixed_23"),
    )
    case.step(
        "mixed_23_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_23_terminal_timeout_s"),
        params={"requests": output("mixed_23", "requests")},
    )
    case.step(
        "mixed_24",
        "request",
        params=case.value("hot_tension.mixed_24"),
    )
    case.step(
        "mixed_24_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_24_terminal_timeout_s"),
        params={"requests": output("mixed_24", "requests")},
    )
    case.step(
        "mixed_25",
        "request",
        params=case.value("hot_tension.mixed_25"),
    )
    case.step(
        "mixed_25_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_25_terminal_timeout_s"),
        params={"requests": output("mixed_25", "requests")},
    )
    case.step(
        "mixed_26",
        "request",
        params=case.value("hot_tension.mixed_26"),
    )
    case.step(
        "mixed_26_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_26_terminal_timeout_s"),
        params={"requests": output("mixed_26", "requests")},
    )
    case.step(
        "mixed_27",
        "request",
        params=case.value("hot_tension.mixed_27"),
    )
    case.step(
        "mixed_27_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_27_terminal_timeout_s"),
        params={"requests": output("mixed_27", "requests")},
    )
    case.step(
        "mixed_28",
        "request",
        params=case.value("hot_tension.mixed_28"),
    )
    case.step(
        "mixed_28_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_28_terminal_timeout_s"),
        params={"requests": output("mixed_28", "requests")},
    )
    case.step(
        "mixed_29",
        "request",
        params=case.value("hot_tension.mixed_29"),
    )
    case.step(
        "mixed_29_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_29_terminal_timeout_s"),
        params={"requests": output("mixed_29", "requests")},
    )
    case.step(
        "mixed_30",
        "request",
        params=case.value("hot_tension.mixed_30"),
    )
    case.step(
        "mixed_30_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_30_terminal_timeout_s"),
        params={"requests": output("mixed_30", "requests")},
    )
    case.step(
        "mixed_31",
        "request",
        params=case.value("hot_tension.mixed_31"),
    )
    case.step(
        "mixed_31_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_31_terminal_timeout_s"),
        params={"requests": output("mixed_31", "requests")},
    )
    case.step(
        "mixed_32",
        "request",
        params=case.value("hot_tension.mixed_32"),
    )
    case.step(
        "mixed_32_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_32_terminal_timeout_s"),
        params={"requests": output("mixed_32", "requests")},
    )
    case.step(
        "mixed_33",
        "request",
        params=case.value("hot_tension.mixed_33"),
    )
    case.step(
        "mixed_33_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_33_terminal_timeout_s"),
        params={"requests": output("mixed_33", "requests")},
    )
    case.step(
        "mixed_34",
        "request",
        params=case.value("hot_tension.mixed_34"),
    )
    case.step(
        "mixed_34_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_34_terminal_timeout_s"),
        params={"requests": output("mixed_34", "requests")},
    )
    case.step(
        "mixed_35",
        "request",
        params=case.value("hot_tension.mixed_35"),
    )
    case.step(
        "mixed_35_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_35_terminal_timeout_s"),
        params={"requests": output("mixed_35", "requests")},
    )
    case.step(
        "mixed_36",
        "request",
        params=case.value("hot_tension.mixed_36"),
    )
    case.step(
        "mixed_36_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_36_terminal_timeout_s"),
        params={"requests": output("mixed_36", "requests")},
    )
    case.step(
        "mixed_37",
        "request",
        params=case.value("hot_tension.mixed_37"),
    )
    case.step(
        "mixed_37_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_37_terminal_timeout_s"),
        params={"requests": output("mixed_37", "requests")},
    )
    case.step(
        "mixed_38",
        "request",
        params=case.value("hot_tension.mixed_38"),
    )
    case.step(
        "mixed_38_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_38_terminal_timeout_s"),
        params={"requests": output("mixed_38", "requests")},
    )
    case.step(
        "mixed_39",
        "request",
        params=case.value("hot_tension.mixed_39"),
    )
    case.step(
        "mixed_39_terminal",
        "wait",
        timeout_s=case.value("hot_tension.mixed_39_terminal_timeout_s"),
        params={"requests": output("mixed_39", "requests")},
    )
    case.step(
        "family_fidelity",
        "kv_fidelity_check",
        params=case.params(
            "hot_tension.family_fidelity",
            {
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
            },
        ),
    )
    case.step(
        "holder_total",
        "kv_holder_share_check",
        params=case.params(
            "hot_tension.holder_total",
            {
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
            },
        ),
    )
    case.step(
        "free_not_starved",
        "kv_off_holder_check",
        params=case.params(
            "hot_tension.free_not_starved",
            {
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
            },
        ),
    )
    case.step("cleanup", "teardown")


def mixed_tiers(case):
    case.step("setup", "setup", timeout_s=case.value("mixed_tiers.setup_timeout_s"))
    case.step(
        "full_seed",
        "request",
        params=case.value("mixed_tiers.full_seed"),
    )
    case.step(
        "full_seed_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_seed_terminal_timeout_s"),
        params={"requests": output("full_seed", "requests")},
    )
    case.step(
        "full_holder",
        "kv_landing",
        params={"requests": output("full_seed", "requests")},
    )
    case.step("full_sync", "balance_pause", params=case.value("mixed_tiers.full_sync"))
    case.step(
        "full_0",
        "request",
        params=case.value("mixed_tiers.full_0"),
    )
    case.step(
        "full_0_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_0_terminal_timeout_s"),
        params={"requests": output("full_0", "requests")},
    )
    case.step(
        "full_1",
        "request",
        params=case.value("mixed_tiers.full_1"),
    )
    case.step(
        "full_1_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_1_terminal_timeout_s"),
        params={"requests": output("full_1", "requests")},
    )
    case.step(
        "full_2",
        "request",
        params=case.value("mixed_tiers.full_2"),
    )
    case.step(
        "full_2_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_2_terminal_timeout_s"),
        params={"requests": output("full_2", "requests")},
    )
    case.step(
        "full_3",
        "request",
        params=case.value("mixed_tiers.full_3"),
    )
    case.step(
        "full_3_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_3_terminal_timeout_s"),
        params={"requests": output("full_3", "requests")},
    )
    case.step(
        "full_4",
        "request",
        params=case.value("mixed_tiers.full_4"),
    )
    case.step(
        "full_4_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_4_terminal_timeout_s"),
        params={"requests": output("full_4", "requests")},
    )
    case.step(
        "full_5",
        "request",
        params=case.value("mixed_tiers.full_5"),
    )
    case.step(
        "full_5_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_5_terminal_timeout_s"),
        params={"requests": output("full_5", "requests")},
    )
    case.step(
        "full_6",
        "request",
        params=case.value("mixed_tiers.full_6"),
    )
    case.step(
        "full_6_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_6_terminal_timeout_s"),
        params={"requests": output("full_6", "requests")},
    )
    case.step(
        "full_7",
        "request",
        params=case.value("mixed_tiers.full_7"),
    )
    case.step(
        "full_7_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_7_terminal_timeout_s"),
        params={"requests": output("full_7", "requests")},
    )
    case.step(
        "full_8",
        "request",
        params=case.value("mixed_tiers.full_8"),
    )
    case.step(
        "full_8_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_8_terminal_timeout_s"),
        params={"requests": output("full_8", "requests")},
    )
    case.step(
        "full_9",
        "request",
        params=case.value("mixed_tiers.full_9"),
    )
    case.step(
        "full_9_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.full_9_terminal_timeout_s"),
        params={"requests": output("full_9", "requests")},
    )
    case.step(
        "half_seed",
        "request",
        params=case.value("mixed_tiers.half_seed"),
    )
    case.step(
        "half_seed_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_seed_terminal_timeout_s"),
        params={"requests": output("half_seed", "requests")},
    )
    case.step(
        "half_holder",
        "kv_landing",
        params={"requests": output("half_seed", "requests")},
    )
    case.step("half_sync", "balance_pause", params=case.value("mixed_tiers.half_sync"))
    case.step(
        "half_0",
        "request",
        params=case.value("mixed_tiers.half_0"),
    )
    case.step(
        "half_0_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_0_terminal_timeout_s"),
        params={"requests": output("half_0", "requests")},
    )
    case.step(
        "half_1",
        "request",
        params=case.value("mixed_tiers.half_1"),
    )
    case.step(
        "half_1_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_1_terminal_timeout_s"),
        params={"requests": output("half_1", "requests")},
    )
    case.step(
        "half_2",
        "request",
        params=case.value("mixed_tiers.half_2"),
    )
    case.step(
        "half_2_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_2_terminal_timeout_s"),
        params={"requests": output("half_2", "requests")},
    )
    case.step(
        "half_3",
        "request",
        params=case.value("mixed_tiers.half_3"),
    )
    case.step(
        "half_3_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_3_terminal_timeout_s"),
        params={"requests": output("half_3", "requests")},
    )
    case.step(
        "half_4",
        "request",
        params=case.value("mixed_tiers.half_4"),
    )
    case.step(
        "half_4_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_4_terminal_timeout_s"),
        params={"requests": output("half_4", "requests")},
    )
    case.step(
        "half_5",
        "request",
        params=case.value("mixed_tiers.half_5"),
    )
    case.step(
        "half_5_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_5_terminal_timeout_s"),
        params={"requests": output("half_5", "requests")},
    )
    case.step(
        "half_6",
        "request",
        params=case.value("mixed_tiers.half_6"),
    )
    case.step(
        "half_6_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_6_terminal_timeout_s"),
        params={"requests": output("half_6", "requests")},
    )
    case.step(
        "half_7",
        "request",
        params=case.value("mixed_tiers.half_7"),
    )
    case.step(
        "half_7_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_7_terminal_timeout_s"),
        params={"requests": output("half_7", "requests")},
    )
    case.step(
        "half_8",
        "request",
        params=case.value("mixed_tiers.half_8"),
    )
    case.step(
        "half_8_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_8_terminal_timeout_s"),
        params={"requests": output("half_8", "requests")},
    )
    case.step(
        "half_9",
        "request",
        params=case.value("mixed_tiers.half_9"),
    )
    case.step(
        "half_9_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.half_9_terminal_timeout_s"),
        params={"requests": output("half_9", "requests")},
    )
    case.step(
        "zero_0",
        "request",
        params=case.value("mixed_tiers.zero_0"),
    )
    case.step(
        "zero_0_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_0_terminal_timeout_s"),
        params={"requests": output("zero_0", "requests")},
    )
    case.step(
        "zero_1",
        "request",
        params=case.value("mixed_tiers.zero_1"),
    )
    case.step(
        "zero_1_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_1_terminal_timeout_s"),
        params={"requests": output("zero_1", "requests")},
    )
    case.step(
        "zero_2",
        "request",
        params=case.value("mixed_tiers.zero_2"),
    )
    case.step(
        "zero_2_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_2_terminal_timeout_s"),
        params={"requests": output("zero_2", "requests")},
    )
    case.step(
        "zero_3",
        "request",
        params=case.value("mixed_tiers.zero_3"),
    )
    case.step(
        "zero_3_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_3_terminal_timeout_s"),
        params={"requests": output("zero_3", "requests")},
    )
    case.step(
        "zero_4",
        "request",
        params=case.value("mixed_tiers.zero_4"),
    )
    case.step(
        "zero_4_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_4_terminal_timeout_s"),
        params={"requests": output("zero_4", "requests")},
    )
    case.step(
        "zero_5",
        "request",
        params=case.value("mixed_tiers.zero_5"),
    )
    case.step(
        "zero_5_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_5_terminal_timeout_s"),
        params={"requests": output("zero_5", "requests")},
    )
    case.step(
        "zero_6",
        "request",
        params=case.value("mixed_tiers.zero_6"),
    )
    case.step(
        "zero_6_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_6_terminal_timeout_s"),
        params={"requests": output("zero_6", "requests")},
    )
    case.step(
        "zero_7",
        "request",
        params=case.value("mixed_tiers.zero_7"),
    )
    case.step(
        "zero_7_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_7_terminal_timeout_s"),
        params={"requests": output("zero_7", "requests")},
    )
    case.step(
        "zero_8",
        "request",
        params=case.value("mixed_tiers.zero_8"),
    )
    case.step(
        "zero_8_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_8_terminal_timeout_s"),
        params={"requests": output("zero_8", "requests")},
    )
    case.step(
        "zero_9",
        "request",
        params=case.value("mixed_tiers.zero_9"),
    )
    case.step(
        "zero_9_terminal",
        "wait",
        timeout_s=case.value("mixed_tiers.zero_9_terminal_timeout_s"),
        params={"requests": output("zero_9", "requests")},
    )
    case.step(
        "full_concentration",
        "kv_affinity_check",
        params=case.params(
            "mixed_tiers.full_concentration",
            {
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
            },
        ),
    )
    case.step(
        "half_concentration",
        "kv_affinity_check",
        params=case.params(
            "mixed_tiers.half_concentration",
            {
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
            },
        ),
    )
    case.step(
        "zero_spread",
        "kv_union_check",
        params=case.params(
            "mixed_tiers.zero_spread",
            {
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
                ]
            },
        ),
    )
    case.step("cleanup", "teardown")


def leader_spill_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("leader_spill_batch.setup_timeout_s")
    )
    case.step(
        "steer_a_settle",
        "balance_pause",
        params=case.value("leader_spill_batch.steer_a_settle"),
    )
    case.step(
        "p0",
        "request",
        params=case.value("leader_spill_batch.p0"),
    )
    case.step(
        "seed_f",
        "request",
        params=case.value("leader_spill_batch.seed_f"),
    )
    case.step(
        "p0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.p0_done_timeout_s"),
        params={"requests": output("p0", "requests")},
    )
    case.step(
        "seed_f_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.seed_f_done_timeout_s"),
        params={"requests": output("seed_f", "requests")},
    )
    case.step(
        "p0_holder",
        "kv_landing",
        params={"requests": output("p0", "requests")},
    )
    case.step(
        "seed_f_holder",
        "kv_landing",
        params={"requests": output("seed_f", "requests")},
    )
    case.step(
        "holders_distinct",
        "kv_distinct",
        params={
            "first": output("p0_holder", "engine"),
            "second": output("seed_f_holder", "engine"),
        },
    )
    case.step(
        "filler_sync", "balance_pause", params=case.value("leader_spill_batch.filler_sync")
    )
    case.step(
        "filler_a",
        "request",
        params=case.value("leader_spill_batch.filler_a"),
    )
    case.step(
        "filler_a_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.filler_a_gap"),
    )
    case.step(
        "steer_a",
        "request",
        params=case.value("leader_spill_batch.steer_a"),
    )
    case.step(
        "steer_a_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.steer_a_done_timeout_s"),
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
        params=case.params(
            "leader_spill_batch.steer_a_placement",
            {
                "first": output("steer_a_holder", "engine"),
                "second": output("p0_holder", "engine"),
            },
        ),
    )
    case.step(
        "steer_a_quiet",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.steer_a_quiet_timeout_s"),
        params=case.value("leader_spill_batch.steer_a_quiet"),
    )
    case.step(
        "steer_b_settle",
        "balance_pause",
        params=case.value("leader_spill_batch.steer_b_settle"),
    )
    case.step(
        "filler_b",
        "request",
        params=case.value("leader_spill_batch.filler_b"),
    )
    case.step(
        "filler_b_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.filler_b_gap"),
    )
    case.step(
        "steer_b",
        "request",
        params=case.value("leader_spill_batch.steer_b"),
    )
    case.step(
        "steer_b_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.steer_b_done_timeout_s"),
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
        params=case.params(
            "leader_spill_batch.steer_b_placement",
            {
                "first": output("steer_b_holder", "engine"),
                "second": output("seed_f_holder", "engine"),
            },
        ),
    )
    case.step(
        "steer_restore_settle",
        "balance_pause",
        params=case.value("leader_spill_batch.steer_restore_settle"),
    )
    case.step(
        "steer_quiet",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.steer_quiet_timeout_s"),
        params=case.value("leader_spill_batch.steer_quiet"),
    )
    case.step(
        "steer_family_0",
        "kv_membership_check",
        params=case.params(
            "leader_spill_batch.steer_family_0",
            {"snapshot": output("steer_quiet", "snapshot")},
        ),
    )
    case.step(
        "steer_family_1",
        "kv_membership_check",
        params=case.params(
            "leader_spill_batch.steer_family_1",
            {"snapshot": output("steer_quiet", "snapshot")},
        ),
    )
    case.step(
        "baseline_w0_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w0_r0_before"),
    )
    case.step(
        "baseline_w0_r0",
        "request",
        params=case.value("leader_spill_batch.baseline_w0_r0"),
    )
    case.step(
        "baseline_w0_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r0_done_timeout_s"),
        params={"requests": output("baseline_w0_r0", "requests")},
    )
    case.step(
        "baseline_w0_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w0_r0_hit",
            {
                "snapshot": output("baseline_w0_r0_before", "snapshot"),
                "requests": output("baseline_w0_r0", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w0_r1_before"),
    )
    case.step(
        "baseline_w0_r1",
        "request",
        params=case.value("leader_spill_batch.baseline_w0_r1"),
    )
    case.step(
        "baseline_w0_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r1_done_timeout_s"),
        params={"requests": output("baseline_w0_r1", "requests")},
    )
    case.step(
        "baseline_w0_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w0_r1_hit",
            {
                "snapshot": output("baseline_w0_r1_before", "snapshot"),
                "requests": output("baseline_w0_r1", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w0_r2_before"),
    )
    case.step(
        "baseline_w0_r2",
        "request",
        params=case.value("leader_spill_batch.baseline_w0_r2"),
    )
    case.step(
        "baseline_w0_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r2_done_timeout_s"),
        params={"requests": output("baseline_w0_r2", "requests")},
    )
    case.step(
        "baseline_w0_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w0_r2_hit",
            {
                "snapshot": output("baseline_w0_r2_before", "snapshot"),
                "requests": output("baseline_w0_r2", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w0_r3_before"),
    )
    case.step(
        "baseline_w0_r3",
        "request",
        params=case.value("leader_spill_batch.baseline_w0_r3"),
    )
    case.step(
        "baseline_w0_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w0_r3_done_timeout_s"),
        params={"requests": output("baseline_w0_r3", "requests")},
    )
    case.step(
        "baseline_w0_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w0_r3_hit",
            {
                "snapshot": output("baseline_w0_r3_before", "snapshot"),
                "requests": output("baseline_w0_r3", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w0_end_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w0_end"),
    )
    case.step(
        "baseline_w0_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.baseline_w0_sync"),
    )
    case.step(
        "baseline_w1_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w1_r0_before"),
    )
    case.step(
        "baseline_w1_r0",
        "request",
        params=case.value("leader_spill_batch.baseline_w1_r0"),
    )
    case.step(
        "baseline_w1_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r0_done_timeout_s"),
        params={"requests": output("baseline_w1_r0", "requests")},
    )
    case.step(
        "baseline_w1_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w1_r0_hit",
            {
                "snapshot": output("baseline_w1_r0_before", "snapshot"),
                "requests": output("baseline_w1_r0", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w1_r1_before"),
    )
    case.step(
        "baseline_w1_r1",
        "request",
        params=case.value("leader_spill_batch.baseline_w1_r1"),
    )
    case.step(
        "baseline_w1_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r1_done_timeout_s"),
        params={"requests": output("baseline_w1_r1", "requests")},
    )
    case.step(
        "baseline_w1_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w1_r1_hit",
            {
                "snapshot": output("baseline_w1_r1_before", "snapshot"),
                "requests": output("baseline_w1_r1", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w1_r2_before"),
    )
    case.step(
        "baseline_w1_r2",
        "request",
        params=case.value("leader_spill_batch.baseline_w1_r2"),
    )
    case.step(
        "baseline_w1_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r2_done_timeout_s"),
        params={"requests": output("baseline_w1_r2", "requests")},
    )
    case.step(
        "baseline_w1_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w1_r2_hit",
            {
                "snapshot": output("baseline_w1_r2_before", "snapshot"),
                "requests": output("baseline_w1_r2", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w1_r3_before"),
    )
    case.step(
        "baseline_w1_r3",
        "request",
        params=case.value("leader_spill_batch.baseline_w1_r3"),
    )
    case.step(
        "baseline_w1_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w1_r3_done_timeout_s"),
        params={"requests": output("baseline_w1_r3", "requests")},
    )
    case.step(
        "baseline_w1_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w1_r3_hit",
            {
                "snapshot": output("baseline_w1_r3_before", "snapshot"),
                "requests": output("baseline_w1_r3", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w1_end_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w1_end"),
    )
    case.step(
        "baseline_w1_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.baseline_w1_sync"),
    )
    case.step(
        "baseline_w2_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w2_r0_before"),
    )
    case.step(
        "baseline_w2_r0",
        "request",
        params=case.value("leader_spill_batch.baseline_w2_r0"),
    )
    case.step(
        "baseline_w2_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r0_done_timeout_s"),
        params={"requests": output("baseline_w2_r0", "requests")},
    )
    case.step(
        "baseline_w2_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w2_r0_hit",
            {
                "snapshot": output("baseline_w2_r0_before", "snapshot"),
                "requests": output("baseline_w2_r0", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w2_r1_before"),
    )
    case.step(
        "baseline_w2_r1",
        "request",
        params=case.value("leader_spill_batch.baseline_w2_r1"),
    )
    case.step(
        "baseline_w2_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r1_done_timeout_s"),
        params={"requests": output("baseline_w2_r1", "requests")},
    )
    case.step(
        "baseline_w2_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w2_r1_hit",
            {
                "snapshot": output("baseline_w2_r1_before", "snapshot"),
                "requests": output("baseline_w2_r1", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w2_r2_before"),
    )
    case.step(
        "baseline_w2_r2",
        "request",
        params=case.value("leader_spill_batch.baseline_w2_r2"),
    )
    case.step(
        "baseline_w2_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r2_done_timeout_s"),
        params={"requests": output("baseline_w2_r2", "requests")},
    )
    case.step(
        "baseline_w2_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w2_r2_hit",
            {
                "snapshot": output("baseline_w2_r2_before", "snapshot"),
                "requests": output("baseline_w2_r2", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w2_r3_before"),
    )
    case.step(
        "baseline_w2_r3",
        "request",
        params=case.value("leader_spill_batch.baseline_w2_r3"),
    )
    case.step(
        "baseline_w2_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.baseline_w2_r3_done_timeout_s"),
        params={"requests": output("baseline_w2_r3", "requests")},
    )
    case.step(
        "baseline_w2_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.baseline_w2_r3_hit",
            {
                "snapshot": output("baseline_w2_r3_before", "snapshot"),
                "requests": output("baseline_w2_r3", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_w2_end_timeout_s"),
        params=case.value("leader_spill_batch.baseline_w2_end"),
    )
    case.step(
        "baseline_w2_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.baseline_w2_sync"),
    )
    case.step(
        "baseline_digest",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.baseline_digest_timeout_s"),
        params=case.value("leader_spill_batch.baseline_digest"),
    )
    case.step(
        "saturation_settle",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_settle"),
    )
    case.step(
        "saturation_w0_f0",
        "request",
        params=case.value("leader_spill_batch.saturation_w0_f0"),
    )
    case.step(
        "saturation_w0_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w0_f0_gap"),
    )
    case.step(
        "saturation_w0_f1",
        "request",
        params=case.value("leader_spill_batch.saturation_w0_f1"),
    )
    case.step(
        "saturation_w0_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w0_f1_gap"),
    )
    case.step(
        "saturation_w0_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w0_r0_before"),
    )
    case.step(
        "saturation_w0_r0",
        "request",
        params=case.value("leader_spill_batch.saturation_w0_r0"),
    )
    case.step(
        "saturation_w0_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w0_r0_spacing"),
    )
    case.step(
        "saturation_w0_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w0_r1_before"),
    )
    case.step(
        "saturation_w0_r1",
        "request",
        params=case.value("leader_spill_batch.saturation_w0_r1"),
    )
    case.step(
        "saturation_w0_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w0_r1_spacing"),
    )
    case.step(
        "saturation_w0_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w0_r2_before"),
    )
    case.step(
        "saturation_w0_r2",
        "request",
        params=case.value("leader_spill_batch.saturation_w0_r2"),
    )
    case.step(
        "saturation_w0_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w0_r2_spacing"),
    )
    case.step(
        "saturation_w0_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w0_r3_before"),
    )
    case.step(
        "saturation_w0_r3",
        "request",
        params=case.value("leader_spill_batch.saturation_w0_r3"),
    )
    case.step(
        "saturation_w0_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r0_done_timeout_s"),
        params={"requests": output("saturation_w0_r0", "requests")},
    )
    case.step(
        "saturation_w0_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w0_r0_hit",
            {
                "snapshot": output("saturation_w0_r0_before", "snapshot"),
                "requests": output("saturation_w0_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r1_done_timeout_s"),
        params={"requests": output("saturation_w0_r1", "requests")},
    )
    case.step(
        "saturation_w0_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w0_r1_hit",
            {
                "snapshot": output("saturation_w0_r1_before", "snapshot"),
                "requests": output("saturation_w0_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r2_done_timeout_s"),
        params={"requests": output("saturation_w0_r2", "requests")},
    )
    case.step(
        "saturation_w0_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w0_r2_hit",
            {
                "snapshot": output("saturation_w0_r2_before", "snapshot"),
                "requests": output("saturation_w0_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w0_r3_done_timeout_s"),
        params={"requests": output("saturation_w0_r3", "requests")},
    )
    case.step(
        "saturation_w0_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w0_r3_hit",
            {
                "snapshot": output("saturation_w0_r3_before", "snapshot"),
                "requests": output("saturation_w0_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w0_end_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w0_end"),
    )
    case.step(
        "saturation_w0_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w0_f0_done_timeout_s"),
        params={"requests": output("saturation_w0_f0", "requests")},
    )
    case.step(
        "saturation_w0_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w0_f1_done_timeout_s"),
        params={"requests": output("saturation_w0_f1", "requests")},
    )
    case.step(
        "saturation_w1_f0",
        "request",
        params=case.value("leader_spill_batch.saturation_w1_f0"),
    )
    case.step(
        "saturation_w1_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w1_f0_gap"),
    )
    case.step(
        "saturation_w1_f1",
        "request",
        params=case.value("leader_spill_batch.saturation_w1_f1"),
    )
    case.step(
        "saturation_w1_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w1_f1_gap"),
    )
    case.step(
        "saturation_w1_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w1_r0_before"),
    )
    case.step(
        "saturation_w1_r0",
        "request",
        params=case.value("leader_spill_batch.saturation_w1_r0"),
    )
    case.step(
        "saturation_w1_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w1_r0_spacing"),
    )
    case.step(
        "saturation_w1_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w1_r1_before"),
    )
    case.step(
        "saturation_w1_r1",
        "request",
        params=case.value("leader_spill_batch.saturation_w1_r1"),
    )
    case.step(
        "saturation_w1_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w1_r1_spacing"),
    )
    case.step(
        "saturation_w1_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w1_r2_before"),
    )
    case.step(
        "saturation_w1_r2",
        "request",
        params=case.value("leader_spill_batch.saturation_w1_r2"),
    )
    case.step(
        "saturation_w1_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w1_r2_spacing"),
    )
    case.step(
        "saturation_w1_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w1_r3_before"),
    )
    case.step(
        "saturation_w1_r3",
        "request",
        params=case.value("leader_spill_batch.saturation_w1_r3"),
    )
    case.step(
        "saturation_w1_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r0_done_timeout_s"),
        params={"requests": output("saturation_w1_r0", "requests")},
    )
    case.step(
        "saturation_w1_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w1_r0_hit",
            {
                "snapshot": output("saturation_w1_r0_before", "snapshot"),
                "requests": output("saturation_w1_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r1_done_timeout_s"),
        params={"requests": output("saturation_w1_r1", "requests")},
    )
    case.step(
        "saturation_w1_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w1_r1_hit",
            {
                "snapshot": output("saturation_w1_r1_before", "snapshot"),
                "requests": output("saturation_w1_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r2_done_timeout_s"),
        params={"requests": output("saturation_w1_r2", "requests")},
    )
    case.step(
        "saturation_w1_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w1_r2_hit",
            {
                "snapshot": output("saturation_w1_r2_before", "snapshot"),
                "requests": output("saturation_w1_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w1_r3_done_timeout_s"),
        params={"requests": output("saturation_w1_r3", "requests")},
    )
    case.step(
        "saturation_w1_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w1_r3_hit",
            {
                "snapshot": output("saturation_w1_r3_before", "snapshot"),
                "requests": output("saturation_w1_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w1_end_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w1_end"),
    )
    case.step(
        "saturation_w1_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w1_f0_done_timeout_s"),
        params={"requests": output("saturation_w1_f0", "requests")},
    )
    case.step(
        "saturation_w1_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w1_f1_done_timeout_s"),
        params={"requests": output("saturation_w1_f1", "requests")},
    )
    case.step(
        "saturation_w2_f0",
        "request",
        params=case.value("leader_spill_batch.saturation_w2_f0"),
    )
    case.step(
        "saturation_w2_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w2_f0_gap"),
    )
    case.step(
        "saturation_w2_f1",
        "request",
        params=case.value("leader_spill_batch.saturation_w2_f1"),
    )
    case.step(
        "saturation_w2_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w2_f1_gap"),
    )
    case.step(
        "saturation_w2_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w2_r0_before"),
    )
    case.step(
        "saturation_w2_r0",
        "request",
        params=case.value("leader_spill_batch.saturation_w2_r0"),
    )
    case.step(
        "saturation_w2_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w2_r0_spacing"),
    )
    case.step(
        "saturation_w2_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w2_r1_before"),
    )
    case.step(
        "saturation_w2_r1",
        "request",
        params=case.value("leader_spill_batch.saturation_w2_r1"),
    )
    case.step(
        "saturation_w2_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w2_r1_spacing"),
    )
    case.step(
        "saturation_w2_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w2_r2_before"),
    )
    case.step(
        "saturation_w2_r2",
        "request",
        params=case.value("leader_spill_batch.saturation_w2_r2"),
    )
    case.step(
        "saturation_w2_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w2_r2_spacing"),
    )
    case.step(
        "saturation_w2_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w2_r3_before"),
    )
    case.step(
        "saturation_w2_r3",
        "request",
        params=case.value("leader_spill_batch.saturation_w2_r3"),
    )
    case.step(
        "saturation_w2_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r0_done_timeout_s"),
        params={"requests": output("saturation_w2_r0", "requests")},
    )
    case.step(
        "saturation_w2_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w2_r0_hit",
            {
                "snapshot": output("saturation_w2_r0_before", "snapshot"),
                "requests": output("saturation_w2_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r1_done_timeout_s"),
        params={"requests": output("saturation_w2_r1", "requests")},
    )
    case.step(
        "saturation_w2_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w2_r1_hit",
            {
                "snapshot": output("saturation_w2_r1_before", "snapshot"),
                "requests": output("saturation_w2_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r2_done_timeout_s"),
        params={"requests": output("saturation_w2_r2", "requests")},
    )
    case.step(
        "saturation_w2_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w2_r2_hit",
            {
                "snapshot": output("saturation_w2_r2_before", "snapshot"),
                "requests": output("saturation_w2_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w2_r3_done_timeout_s"),
        params={"requests": output("saturation_w2_r3", "requests")},
    )
    case.step(
        "saturation_w2_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w2_r3_hit",
            {
                "snapshot": output("saturation_w2_r3_before", "snapshot"),
                "requests": output("saturation_w2_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w2_end_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w2_end"),
    )
    case.step(
        "saturation_w2_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w2_f0_done_timeout_s"),
        params={"requests": output("saturation_w2_f0", "requests")},
    )
    case.step(
        "saturation_w2_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w2_f1_done_timeout_s"),
        params={"requests": output("saturation_w2_f1", "requests")},
    )
    case.step(
        "saturation_w3_f0",
        "request",
        params=case.value("leader_spill_batch.saturation_w3_f0"),
    )
    case.step(
        "saturation_w3_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w3_f0_gap"),
    )
    case.step(
        "saturation_w3_f1",
        "request",
        params=case.value("leader_spill_batch.saturation_w3_f1"),
    )
    case.step(
        "saturation_w3_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w3_f1_gap"),
    )
    case.step(
        "saturation_w3_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w3_r0_before"),
    )
    case.step(
        "saturation_w3_r0",
        "request",
        params=case.value("leader_spill_batch.saturation_w3_r0"),
    )
    case.step(
        "saturation_w3_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w3_r0_spacing"),
    )
    case.step(
        "saturation_w3_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w3_r1_before"),
    )
    case.step(
        "saturation_w3_r1",
        "request",
        params=case.value("leader_spill_batch.saturation_w3_r1"),
    )
    case.step(
        "saturation_w3_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w3_r1_spacing"),
    )
    case.step(
        "saturation_w3_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w3_r2_before"),
    )
    case.step(
        "saturation_w3_r2",
        "request",
        params=case.value("leader_spill_batch.saturation_w3_r2"),
    )
    case.step(
        "saturation_w3_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_batch.saturation_w3_r2_spacing"),
    )
    case.step(
        "saturation_w3_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w3_r3_before"),
    )
    case.step(
        "saturation_w3_r3",
        "request",
        params=case.value("leader_spill_batch.saturation_w3_r3"),
    )
    case.step(
        "saturation_w3_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r0_done_timeout_s"),
        params={"requests": output("saturation_w3_r0", "requests")},
    )
    case.step(
        "saturation_w3_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w3_r0_hit",
            {
                "snapshot": output("saturation_w3_r0_before", "snapshot"),
                "requests": output("saturation_w3_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r1_done_timeout_s"),
        params={"requests": output("saturation_w3_r1", "requests")},
    )
    case.step(
        "saturation_w3_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w3_r1_hit",
            {
                "snapshot": output("saturation_w3_r1_before", "snapshot"),
                "requests": output("saturation_w3_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r2_done_timeout_s"),
        params={"requests": output("saturation_w3_r2", "requests")},
    )
    case.step(
        "saturation_w3_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w3_r2_hit",
            {
                "snapshot": output("saturation_w3_r2_before", "snapshot"),
                "requests": output("saturation_w3_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w3_r3_done_timeout_s"),
        params={"requests": output("saturation_w3_r3", "requests")},
    )
    case.step(
        "saturation_w3_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.saturation_w3_r3_hit",
            {
                "snapshot": output("saturation_w3_r3_before", "snapshot"),
                "requests": output("saturation_w3_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_w3_end_timeout_s"),
        params=case.value("leader_spill_batch.saturation_w3_end"),
    )
    case.step(
        "saturation_w3_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w3_f0_done_timeout_s"),
        params={"requests": output("saturation_w3_f0", "requests")},
    )
    case.step(
        "saturation_w3_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.saturation_w3_f1_done_timeout_s"),
        params={"requests": output("saturation_w3_f1", "requests")},
    )
    case.step(
        "recovery_settle",
        "balance_pause",
        params=case.value("leader_spill_batch.recovery_settle"),
    )
    case.step(
        "saturation_quiet",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.saturation_quiet_timeout_s"),
        params=case.value("leader_spill_batch.saturation_quiet"),
    )
    case.step(
        "recovery_w0_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w0_r0_before"),
    )
    case.step(
        "recovery_w0_r0",
        "request",
        params=case.value("leader_spill_batch.recovery_w0_r0"),
    )
    case.step(
        "recovery_w0_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r0_done_timeout_s"),
        params={"requests": output("recovery_w0_r0", "requests")},
    )
    case.step(
        "recovery_w0_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w0_r0_hit",
            {
                "snapshot": output("recovery_w0_r0_before", "snapshot"),
                "requests": output("recovery_w0_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w0_r1_before"),
    )
    case.step(
        "recovery_w0_r1",
        "request",
        params=case.value("leader_spill_batch.recovery_w0_r1"),
    )
    case.step(
        "recovery_w0_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r1_done_timeout_s"),
        params={"requests": output("recovery_w0_r1", "requests")},
    )
    case.step(
        "recovery_w0_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w0_r1_hit",
            {
                "snapshot": output("recovery_w0_r1_before", "snapshot"),
                "requests": output("recovery_w0_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w0_r2_before"),
    )
    case.step(
        "recovery_w0_r2",
        "request",
        params=case.value("leader_spill_batch.recovery_w0_r2"),
    )
    case.step(
        "recovery_w0_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r2_done_timeout_s"),
        params={"requests": output("recovery_w0_r2", "requests")},
    )
    case.step(
        "recovery_w0_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w0_r2_hit",
            {
                "snapshot": output("recovery_w0_r2_before", "snapshot"),
                "requests": output("recovery_w0_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w0_r3_before"),
    )
    case.step(
        "recovery_w0_r3",
        "request",
        params=case.value("leader_spill_batch.recovery_w0_r3"),
    )
    case.step(
        "recovery_w0_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w0_r3_done_timeout_s"),
        params={"requests": output("recovery_w0_r3", "requests")},
    )
    case.step(
        "recovery_w0_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w0_r3_hit",
            {
                "snapshot": output("recovery_w0_r3_before", "snapshot"),
                "requests": output("recovery_w0_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w0_end_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w0_end"),
    )
    case.step(
        "recovery_w0_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.recovery_w0_sync"),
    )
    case.step(
        "recovery_w1_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w1_r0_before"),
    )
    case.step(
        "recovery_w1_r0",
        "request",
        params=case.value("leader_spill_batch.recovery_w1_r0"),
    )
    case.step(
        "recovery_w1_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r0_done_timeout_s"),
        params={"requests": output("recovery_w1_r0", "requests")},
    )
    case.step(
        "recovery_w1_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w1_r0_hit",
            {
                "snapshot": output("recovery_w1_r0_before", "snapshot"),
                "requests": output("recovery_w1_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w1_r1_before"),
    )
    case.step(
        "recovery_w1_r1",
        "request",
        params=case.value("leader_spill_batch.recovery_w1_r1"),
    )
    case.step(
        "recovery_w1_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r1_done_timeout_s"),
        params={"requests": output("recovery_w1_r1", "requests")},
    )
    case.step(
        "recovery_w1_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w1_r1_hit",
            {
                "snapshot": output("recovery_w1_r1_before", "snapshot"),
                "requests": output("recovery_w1_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w1_r2_before"),
    )
    case.step(
        "recovery_w1_r2",
        "request",
        params=case.value("leader_spill_batch.recovery_w1_r2"),
    )
    case.step(
        "recovery_w1_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r2_done_timeout_s"),
        params={"requests": output("recovery_w1_r2", "requests")},
    )
    case.step(
        "recovery_w1_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w1_r2_hit",
            {
                "snapshot": output("recovery_w1_r2_before", "snapshot"),
                "requests": output("recovery_w1_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w1_r3_before"),
    )
    case.step(
        "recovery_w1_r3",
        "request",
        params=case.value("leader_spill_batch.recovery_w1_r3"),
    )
    case.step(
        "recovery_w1_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w1_r3_done_timeout_s"),
        params={"requests": output("recovery_w1_r3", "requests")},
    )
    case.step(
        "recovery_w1_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w1_r3_hit",
            {
                "snapshot": output("recovery_w1_r3_before", "snapshot"),
                "requests": output("recovery_w1_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w1_end_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w1_end"),
    )
    case.step(
        "recovery_w1_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.recovery_w1_sync"),
    )
    case.step(
        "recovery_w2_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w2_r0_before"),
    )
    case.step(
        "recovery_w2_r0",
        "request",
        params=case.value("leader_spill_batch.recovery_w2_r0"),
    )
    case.step(
        "recovery_w2_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r0_done_timeout_s"),
        params={"requests": output("recovery_w2_r0", "requests")},
    )
    case.step(
        "recovery_w2_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w2_r0_hit",
            {
                "snapshot": output("recovery_w2_r0_before", "snapshot"),
                "requests": output("recovery_w2_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w2_r1_before"),
    )
    case.step(
        "recovery_w2_r1",
        "request",
        params=case.value("leader_spill_batch.recovery_w2_r1"),
    )
    case.step(
        "recovery_w2_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r1_done_timeout_s"),
        params={"requests": output("recovery_w2_r1", "requests")},
    )
    case.step(
        "recovery_w2_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w2_r1_hit",
            {
                "snapshot": output("recovery_w2_r1_before", "snapshot"),
                "requests": output("recovery_w2_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w2_r2_before"),
    )
    case.step(
        "recovery_w2_r2",
        "request",
        params=case.value("leader_spill_batch.recovery_w2_r2"),
    )
    case.step(
        "recovery_w2_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r2_done_timeout_s"),
        params={"requests": output("recovery_w2_r2", "requests")},
    )
    case.step(
        "recovery_w2_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w2_r2_hit",
            {
                "snapshot": output("recovery_w2_r2_before", "snapshot"),
                "requests": output("recovery_w2_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w2_r3_before"),
    )
    case.step(
        "recovery_w2_r3",
        "request",
        params=case.value("leader_spill_batch.recovery_w2_r3"),
    )
    case.step(
        "recovery_w2_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w2_r3_done_timeout_s"),
        params={"requests": output("recovery_w2_r3", "requests")},
    )
    case.step(
        "recovery_w2_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w2_r3_hit",
            {
                "snapshot": output("recovery_w2_r3_before", "snapshot"),
                "requests": output("recovery_w2_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w2_end_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w2_end"),
    )
    case.step(
        "recovery_w2_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.recovery_w2_sync"),
    )
    case.step(
        "recovery_w3_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w3_r0_before"),
    )
    case.step(
        "recovery_w3_r0",
        "request",
        params=case.value("leader_spill_batch.recovery_w3_r0"),
    )
    case.step(
        "recovery_w3_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r0_done_timeout_s"),
        params={"requests": output("recovery_w3_r0", "requests")},
    )
    case.step(
        "recovery_w3_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w3_r0_hit",
            {
                "snapshot": output("recovery_w3_r0_before", "snapshot"),
                "requests": output("recovery_w3_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w3_r1_before"),
    )
    case.step(
        "recovery_w3_r1",
        "request",
        params=case.value("leader_spill_batch.recovery_w3_r1"),
    )
    case.step(
        "recovery_w3_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r1_done_timeout_s"),
        params={"requests": output("recovery_w3_r1", "requests")},
    )
    case.step(
        "recovery_w3_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w3_r1_hit",
            {
                "snapshot": output("recovery_w3_r1_before", "snapshot"),
                "requests": output("recovery_w3_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w3_r2_before"),
    )
    case.step(
        "recovery_w3_r2",
        "request",
        params=case.value("leader_spill_batch.recovery_w3_r2"),
    )
    case.step(
        "recovery_w3_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r2_done_timeout_s"),
        params={"requests": output("recovery_w3_r2", "requests")},
    )
    case.step(
        "recovery_w3_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w3_r2_hit",
            {
                "snapshot": output("recovery_w3_r2_before", "snapshot"),
                "requests": output("recovery_w3_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w3_r3_before"),
    )
    case.step(
        "recovery_w3_r3",
        "request",
        params=case.value("leader_spill_batch.recovery_w3_r3"),
    )
    case.step(
        "recovery_w3_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w3_r3_done_timeout_s"),
        params={"requests": output("recovery_w3_r3", "requests")},
    )
    case.step(
        "recovery_w3_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w3_r3_hit",
            {
                "snapshot": output("recovery_w3_r3_before", "snapshot"),
                "requests": output("recovery_w3_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w3_end_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w3_end"),
    )
    case.step(
        "recovery_w3_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.recovery_w3_sync"),
    )
    case.step(
        "recovery_w4_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w4_r0_before"),
    )
    case.step(
        "recovery_w4_r0",
        "request",
        params=case.value("leader_spill_batch.recovery_w4_r0"),
    )
    case.step(
        "recovery_w4_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r0_done_timeout_s"),
        params={"requests": output("recovery_w4_r0", "requests")},
    )
    case.step(
        "recovery_w4_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w4_r0_hit",
            {
                "snapshot": output("recovery_w4_r0_before", "snapshot"),
                "requests": output("recovery_w4_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w4_r1_before"),
    )
    case.step(
        "recovery_w4_r1",
        "request",
        params=case.value("leader_spill_batch.recovery_w4_r1"),
    )
    case.step(
        "recovery_w4_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r1_done_timeout_s"),
        params={"requests": output("recovery_w4_r1", "requests")},
    )
    case.step(
        "recovery_w4_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w4_r1_hit",
            {
                "snapshot": output("recovery_w4_r1_before", "snapshot"),
                "requests": output("recovery_w4_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w4_r2_before"),
    )
    case.step(
        "recovery_w4_r2",
        "request",
        params=case.value("leader_spill_batch.recovery_w4_r2"),
    )
    case.step(
        "recovery_w4_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r2_done_timeout_s"),
        params={"requests": output("recovery_w4_r2", "requests")},
    )
    case.step(
        "recovery_w4_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w4_r2_hit",
            {
                "snapshot": output("recovery_w4_r2_before", "snapshot"),
                "requests": output("recovery_w4_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w4_r3_before"),
    )
    case.step(
        "recovery_w4_r3",
        "request",
        params=case.value("leader_spill_batch.recovery_w4_r3"),
    )
    case.step(
        "recovery_w4_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w4_r3_done_timeout_s"),
        params={"requests": output("recovery_w4_r3", "requests")},
    )
    case.step(
        "recovery_w4_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w4_r3_hit",
            {
                "snapshot": output("recovery_w4_r3_before", "snapshot"),
                "requests": output("recovery_w4_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w4_end_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w4_end"),
    )
    case.step(
        "recovery_w4_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.recovery_w4_sync"),
    )
    case.step(
        "recovery_w5_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r0_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w5_r0_before"),
    )
    case.step(
        "recovery_w5_r0",
        "request",
        params=case.value("leader_spill_batch.recovery_w5_r0"),
    )
    case.step(
        "recovery_w5_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r0_done_timeout_s"),
        params={"requests": output("recovery_w5_r0", "requests")},
    )
    case.step(
        "recovery_w5_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w5_r0_hit",
            {
                "snapshot": output("recovery_w5_r0_before", "snapshot"),
                "requests": output("recovery_w5_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r1_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w5_r1_before"),
    )
    case.step(
        "recovery_w5_r1",
        "request",
        params=case.value("leader_spill_batch.recovery_w5_r1"),
    )
    case.step(
        "recovery_w5_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r1_done_timeout_s"),
        params={"requests": output("recovery_w5_r1", "requests")},
    )
    case.step(
        "recovery_w5_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w5_r1_hit",
            {
                "snapshot": output("recovery_w5_r1_before", "snapshot"),
                "requests": output("recovery_w5_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r2_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w5_r2_before"),
    )
    case.step(
        "recovery_w5_r2",
        "request",
        params=case.value("leader_spill_batch.recovery_w5_r2"),
    )
    case.step(
        "recovery_w5_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r2_done_timeout_s"),
        params={"requests": output("recovery_w5_r2", "requests")},
    )
    case.step(
        "recovery_w5_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w5_r2_hit",
            {
                "snapshot": output("recovery_w5_r2_before", "snapshot"),
                "requests": output("recovery_w5_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r3_before_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w5_r3_before"),
    )
    case.step(
        "recovery_w5_r3",
        "request",
        params=case.value("leader_spill_batch.recovery_w5_r3"),
    )
    case.step(
        "recovery_w5_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_batch.recovery_w5_r3_done_timeout_s"),
        params={"requests": output("recovery_w5_r3", "requests")},
    )
    case.step(
        "recovery_w5_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_batch.recovery_w5_r3_hit",
            {
                "snapshot": output("recovery_w5_r3_before", "snapshot"),
                "requests": output("recovery_w5_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_w5_end_timeout_s"),
        params=case.value("leader_spill_batch.recovery_w5_end"),
    )
    case.step(
        "recovery_w5_sync",
        "balance_pause",
        params=case.value("leader_spill_batch.recovery_w5_sync"),
    )
    case.step(
        "recovery_digest",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_batch.recovery_digest_timeout_s"),
        params=case.value("leader_spill_batch.recovery_digest"),
    )
    case.step(
        "phase_observations",
        "kv_phase_observation",
        params=case.params(
            "leader_spill_batch.phase_observations",
            {
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
                "phase_snapshots": {
                    "steer": output("steer_quiet", "snapshot"),
                    "baseline": output("baseline_digest", "snapshot"),
                    "saturation": output("saturation_quiet", "snapshot"),
                    "recovery": output("recovery_digest", "snapshot"),
                },
            },
        ),
    )
    case.step(
        "holder_flips",
        "kv_window_transitions",
        params=case.params(
            "leader_spill_batch.holder_flips",
            {
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
                ]
            },
        ),
    )
    case.step(
        "all_phase_requests",
        "kv_hit_completeness",
        params=case.params(
            "leader_spill_batch.all_phase_requests",
            {
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
                ]
            },
        ),
    )
    case.step(
        "baseline_hit",
        "kv_hit_rate_check",
        params=case.params(
            "leader_spill_batch.baseline_hit",
            {
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
                ]
            },
        ),
    )
    case.step(
        "saturation_hit",
        "kv_hit_rate_check",
        params=case.params(
            "leader_spill_batch.saturation_hit",
            {
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
                ]
            },
        ),
    )
    case.step(
        "recovery_steady_hit",
        "kv_hit_rate_check",
        params=case.params(
            "leader_spill_batch.recovery_steady_hit",
            {
                "samples": [
                    output("recovery_w4_r0_hit", "sample"),
                    output("recovery_w4_r1_hit", "sample"),
                    output("recovery_w4_r2_hit", "sample"),
                    output("recovery_w4_r3_hit", "sample"),
                    output("recovery_w5_r0_hit", "sample"),
                    output("recovery_w5_r1_hit", "sample"),
                    output("recovery_w5_r2_hit", "sample"),
                    output("recovery_w5_r3_hit", "sample"),
                ]
            },
        ),
    )
    case.step(
        "replication",
        "kv_replication_check",
        params=case.params(
            "leader_spill_batch.replication",
            {"snapshot": output("recovery_w5_end", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def leader_spill_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("leader_spill_nonbatch.setup_timeout_s")
    )
    case.step(
        "steer_a_settle",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.steer_a_settle"),
    )
    case.step(
        "p0",
        "request",
        params=case.value("leader_spill_nonbatch.p0"),
    )
    case.step(
        "seed_f",
        "request",
        params=case.value("leader_spill_nonbatch.seed_f"),
    )
    case.step(
        "p0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.p0_done_timeout_s"),
        params={"requests": output("p0", "requests")},
    )
    case.step(
        "seed_f_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.seed_f_done_timeout_s"),
        params={"requests": output("seed_f", "requests")},
    )
    case.step(
        "p0_holder",
        "kv_landing",
        params={"requests": output("p0", "requests")},
    )
    case.step(
        "seed_f_holder",
        "kv_landing",
        params={"requests": output("seed_f", "requests")},
    )
    case.step(
        "holders_distinct",
        "kv_distinct",
        params={
            "first": output("p0_holder", "engine"),
            "second": output("seed_f_holder", "engine"),
        },
    )
    case.step(
        "filler_sync", "balance_pause", params=case.value("leader_spill_nonbatch.filler_sync")
    )
    case.step(
        "filler_a",
        "request",
        params=case.value("leader_spill_nonbatch.filler_a"),
    )
    case.step(
        "filler_a_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.filler_a_gap"),
    )
    case.step(
        "steer_a",
        "request",
        params=case.value("leader_spill_nonbatch.steer_a"),
    )
    case.step(
        "steer_a_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.steer_a_done_timeout_s"),
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
        params=case.params(
            "leader_spill_nonbatch.steer_a_placement",
            {
                "first": output("steer_a_holder", "engine"),
                "second": output("p0_holder", "engine"),
            },
        ),
    )
    case.step(
        "steer_a_quiet",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.steer_a_quiet_timeout_s"),
        params=case.value("leader_spill_nonbatch.steer_a_quiet"),
    )
    case.step(
        "steer_b_settle",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.steer_b_settle"),
    )
    case.step(
        "filler_b",
        "request",
        params=case.value("leader_spill_nonbatch.filler_b"),
    )
    case.step(
        "filler_b_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.filler_b_gap"),
    )
    case.step(
        "steer_b",
        "request",
        params=case.value("leader_spill_nonbatch.steer_b"),
    )
    case.step(
        "steer_b_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.steer_b_done_timeout_s"),
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
        params=case.params(
            "leader_spill_nonbatch.steer_b_placement",
            {
                "first": output("steer_b_holder", "engine"),
                "second": output("seed_f_holder", "engine"),
            },
        ),
    )
    case.step(
        "steer_restore_settle",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.steer_restore_settle"),
    )
    case.step(
        "steer_quiet",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.steer_quiet_timeout_s"),
        params=case.value("leader_spill_nonbatch.steer_quiet"),
    )
    case.step(
        "steer_family_0",
        "kv_membership_check",
        params=case.params(
            "leader_spill_nonbatch.steer_family_0",
            {"snapshot": output("steer_quiet", "snapshot")},
        ),
    )
    case.step(
        "steer_family_1",
        "kv_membership_check",
        params=case.params(
            "leader_spill_nonbatch.steer_family_1",
            {"snapshot": output("steer_quiet", "snapshot")},
        ),
    )
    case.step(
        "baseline_w0_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w0_r0_before"),
    )
    case.step(
        "baseline_w0_r0",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w0_r0"),
    )
    case.step(
        "baseline_w0_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r0_done_timeout_s"),
        params={"requests": output("baseline_w0_r0", "requests")},
    )
    case.step(
        "baseline_w0_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w0_r0_hit",
            {
                "snapshot": output("baseline_w0_r0_before", "snapshot"),
                "requests": output("baseline_w0_r0", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w0_r1_before"),
    )
    case.step(
        "baseline_w0_r1",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w0_r1"),
    )
    case.step(
        "baseline_w0_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r1_done_timeout_s"),
        params={"requests": output("baseline_w0_r1", "requests")},
    )
    case.step(
        "baseline_w0_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w0_r1_hit",
            {
                "snapshot": output("baseline_w0_r1_before", "snapshot"),
                "requests": output("baseline_w0_r1", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w0_r2_before"),
    )
    case.step(
        "baseline_w0_r2",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w0_r2"),
    )
    case.step(
        "baseline_w0_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r2_done_timeout_s"),
        params={"requests": output("baseline_w0_r2", "requests")},
    )
    case.step(
        "baseline_w0_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w0_r2_hit",
            {
                "snapshot": output("baseline_w0_r2_before", "snapshot"),
                "requests": output("baseline_w0_r2", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w0_r3_before"),
    )
    case.step(
        "baseline_w0_r3",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w0_r3"),
    )
    case.step(
        "baseline_w0_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_r3_done_timeout_s"),
        params={"requests": output("baseline_w0_r3", "requests")},
    )
    case.step(
        "baseline_w0_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w0_r3_hit",
            {
                "snapshot": output("baseline_w0_r3_before", "snapshot"),
                "requests": output("baseline_w0_r3", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w0_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w0_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w0_end"),
    )
    case.step(
        "baseline_w0_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.baseline_w0_sync"),
    )
    case.step(
        "baseline_w1_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w1_r0_before"),
    )
    case.step(
        "baseline_w1_r0",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w1_r0"),
    )
    case.step(
        "baseline_w1_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r0_done_timeout_s"),
        params={"requests": output("baseline_w1_r0", "requests")},
    )
    case.step(
        "baseline_w1_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w1_r0_hit",
            {
                "snapshot": output("baseline_w1_r0_before", "snapshot"),
                "requests": output("baseline_w1_r0", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w1_r1_before"),
    )
    case.step(
        "baseline_w1_r1",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w1_r1"),
    )
    case.step(
        "baseline_w1_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r1_done_timeout_s"),
        params={"requests": output("baseline_w1_r1", "requests")},
    )
    case.step(
        "baseline_w1_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w1_r1_hit",
            {
                "snapshot": output("baseline_w1_r1_before", "snapshot"),
                "requests": output("baseline_w1_r1", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w1_r2_before"),
    )
    case.step(
        "baseline_w1_r2",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w1_r2"),
    )
    case.step(
        "baseline_w1_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r2_done_timeout_s"),
        params={"requests": output("baseline_w1_r2", "requests")},
    )
    case.step(
        "baseline_w1_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w1_r2_hit",
            {
                "snapshot": output("baseline_w1_r2_before", "snapshot"),
                "requests": output("baseline_w1_r2", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w1_r3_before"),
    )
    case.step(
        "baseline_w1_r3",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w1_r3"),
    )
    case.step(
        "baseline_w1_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_r3_done_timeout_s"),
        params={"requests": output("baseline_w1_r3", "requests")},
    )
    case.step(
        "baseline_w1_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w1_r3_hit",
            {
                "snapshot": output("baseline_w1_r3_before", "snapshot"),
                "requests": output("baseline_w1_r3", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w1_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w1_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w1_end"),
    )
    case.step(
        "baseline_w1_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.baseline_w1_sync"),
    )
    case.step(
        "baseline_w2_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w2_r0_before"),
    )
    case.step(
        "baseline_w2_r0",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w2_r0"),
    )
    case.step(
        "baseline_w2_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r0_done_timeout_s"),
        params={"requests": output("baseline_w2_r0", "requests")},
    )
    case.step(
        "baseline_w2_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w2_r0_hit",
            {
                "snapshot": output("baseline_w2_r0_before", "snapshot"),
                "requests": output("baseline_w2_r0", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w2_r1_before"),
    )
    case.step(
        "baseline_w2_r1",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w2_r1"),
    )
    case.step(
        "baseline_w2_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r1_done_timeout_s"),
        params={"requests": output("baseline_w2_r1", "requests")},
    )
    case.step(
        "baseline_w2_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w2_r1_hit",
            {
                "snapshot": output("baseline_w2_r1_before", "snapshot"),
                "requests": output("baseline_w2_r1", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w2_r2_before"),
    )
    case.step(
        "baseline_w2_r2",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w2_r2"),
    )
    case.step(
        "baseline_w2_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r2_done_timeout_s"),
        params={"requests": output("baseline_w2_r2", "requests")},
    )
    case.step(
        "baseline_w2_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w2_r2_hit",
            {
                "snapshot": output("baseline_w2_r2_before", "snapshot"),
                "requests": output("baseline_w2_r2", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w2_r3_before"),
    )
    case.step(
        "baseline_w2_r3",
        "request",
        params=case.value("leader_spill_nonbatch.baseline_w2_r3"),
    )
    case.step(
        "baseline_w2_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_r3_done_timeout_s"),
        params={"requests": output("baseline_w2_r3", "requests")},
    )
    case.step(
        "baseline_w2_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.baseline_w2_r3_hit",
            {
                "snapshot": output("baseline_w2_r3_before", "snapshot"),
                "requests": output("baseline_w2_r3", "requests"),
            },
        ),
    )
    case.step(
        "baseline_w2_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_w2_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_w2_end"),
    )
    case.step(
        "baseline_w2_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.baseline_w2_sync"),
    )
    case.step(
        "baseline_digest",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.baseline_digest_timeout_s"),
        params=case.value("leader_spill_nonbatch.baseline_digest"),
    )
    case.step(
        "saturation_settle",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_settle"),
    )
    case.step(
        "saturation_w0_f0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w0_f0"),
    )
    case.step(
        "saturation_w0_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w0_f0_gap"),
    )
    case.step(
        "saturation_w0_f1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w0_f1"),
    )
    case.step(
        "saturation_w0_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w0_f1_gap"),
    )
    case.step(
        "saturation_w0_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w0_r0_before"),
    )
    case.step(
        "saturation_w0_r0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w0_r0"),
    )
    case.step(
        "saturation_w0_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w0_r0_spacing"),
    )
    case.step(
        "saturation_w0_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w0_r1_before"),
    )
    case.step(
        "saturation_w0_r1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w0_r1"),
    )
    case.step(
        "saturation_w0_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w0_r1_spacing"),
    )
    case.step(
        "saturation_w0_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w0_r2_before"),
    )
    case.step(
        "saturation_w0_r2",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w0_r2"),
    )
    case.step(
        "saturation_w0_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w0_r2_spacing"),
    )
    case.step(
        "saturation_w0_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w0_r3_before"),
    )
    case.step(
        "saturation_w0_r3",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w0_r3"),
    )
    case.step(
        "saturation_w0_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r0_done_timeout_s"),
        params={"requests": output("saturation_w0_r0", "requests")},
    )
    case.step(
        "saturation_w0_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w0_r0_hit",
            {
                "snapshot": output("saturation_w0_r0_before", "snapshot"),
                "requests": output("saturation_w0_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r1_done_timeout_s"),
        params={"requests": output("saturation_w0_r1", "requests")},
    )
    case.step(
        "saturation_w0_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w0_r1_hit",
            {
                "snapshot": output("saturation_w0_r1_before", "snapshot"),
                "requests": output("saturation_w0_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r2_done_timeout_s"),
        params={"requests": output("saturation_w0_r2", "requests")},
    )
    case.step(
        "saturation_w0_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w0_r2_hit",
            {
                "snapshot": output("saturation_w0_r2_before", "snapshot"),
                "requests": output("saturation_w0_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_r3_done_timeout_s"),
        params={"requests": output("saturation_w0_r3", "requests")},
    )
    case.step(
        "saturation_w0_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w0_r3_hit",
            {
                "snapshot": output("saturation_w0_r3_before", "snapshot"),
                "requests": output("saturation_w0_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w0_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w0_end"),
    )
    case.step(
        "saturation_w0_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_f0_done_timeout_s"),
        params={"requests": output("saturation_w0_f0", "requests")},
    )
    case.step(
        "saturation_w0_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w0_f1_done_timeout_s"),
        params={"requests": output("saturation_w0_f1", "requests")},
    )
    case.step(
        "saturation_w1_f0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w1_f0"),
    )
    case.step(
        "saturation_w1_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w1_f0_gap"),
    )
    case.step(
        "saturation_w1_f1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w1_f1"),
    )
    case.step(
        "saturation_w1_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w1_f1_gap"),
    )
    case.step(
        "saturation_w1_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w1_r0_before"),
    )
    case.step(
        "saturation_w1_r0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w1_r0"),
    )
    case.step(
        "saturation_w1_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w1_r0_spacing"),
    )
    case.step(
        "saturation_w1_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w1_r1_before"),
    )
    case.step(
        "saturation_w1_r1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w1_r1"),
    )
    case.step(
        "saturation_w1_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w1_r1_spacing"),
    )
    case.step(
        "saturation_w1_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w1_r2_before"),
    )
    case.step(
        "saturation_w1_r2",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w1_r2"),
    )
    case.step(
        "saturation_w1_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w1_r2_spacing"),
    )
    case.step(
        "saturation_w1_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w1_r3_before"),
    )
    case.step(
        "saturation_w1_r3",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w1_r3"),
    )
    case.step(
        "saturation_w1_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r0_done_timeout_s"),
        params={"requests": output("saturation_w1_r0", "requests")},
    )
    case.step(
        "saturation_w1_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w1_r0_hit",
            {
                "snapshot": output("saturation_w1_r0_before", "snapshot"),
                "requests": output("saturation_w1_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r1_done_timeout_s"),
        params={"requests": output("saturation_w1_r1", "requests")},
    )
    case.step(
        "saturation_w1_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w1_r1_hit",
            {
                "snapshot": output("saturation_w1_r1_before", "snapshot"),
                "requests": output("saturation_w1_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r2_done_timeout_s"),
        params={"requests": output("saturation_w1_r2", "requests")},
    )
    case.step(
        "saturation_w1_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w1_r2_hit",
            {
                "snapshot": output("saturation_w1_r2_before", "snapshot"),
                "requests": output("saturation_w1_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_r3_done_timeout_s"),
        params={"requests": output("saturation_w1_r3", "requests")},
    )
    case.step(
        "saturation_w1_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w1_r3_hit",
            {
                "snapshot": output("saturation_w1_r3_before", "snapshot"),
                "requests": output("saturation_w1_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w1_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w1_end"),
    )
    case.step(
        "saturation_w1_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_f0_done_timeout_s"),
        params={"requests": output("saturation_w1_f0", "requests")},
    )
    case.step(
        "saturation_w1_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w1_f1_done_timeout_s"),
        params={"requests": output("saturation_w1_f1", "requests")},
    )
    case.step(
        "saturation_w2_f0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w2_f0"),
    )
    case.step(
        "saturation_w2_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w2_f0_gap"),
    )
    case.step(
        "saturation_w2_f1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w2_f1"),
    )
    case.step(
        "saturation_w2_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w2_f1_gap"),
    )
    case.step(
        "saturation_w2_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w2_r0_before"),
    )
    case.step(
        "saturation_w2_r0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w2_r0"),
    )
    case.step(
        "saturation_w2_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w2_r0_spacing"),
    )
    case.step(
        "saturation_w2_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w2_r1_before"),
    )
    case.step(
        "saturation_w2_r1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w2_r1"),
    )
    case.step(
        "saturation_w2_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w2_r1_spacing"),
    )
    case.step(
        "saturation_w2_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w2_r2_before"),
    )
    case.step(
        "saturation_w2_r2",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w2_r2"),
    )
    case.step(
        "saturation_w2_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w2_r2_spacing"),
    )
    case.step(
        "saturation_w2_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w2_r3_before"),
    )
    case.step(
        "saturation_w2_r3",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w2_r3"),
    )
    case.step(
        "saturation_w2_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r0_done_timeout_s"),
        params={"requests": output("saturation_w2_r0", "requests")},
    )
    case.step(
        "saturation_w2_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w2_r0_hit",
            {
                "snapshot": output("saturation_w2_r0_before", "snapshot"),
                "requests": output("saturation_w2_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r1_done_timeout_s"),
        params={"requests": output("saturation_w2_r1", "requests")},
    )
    case.step(
        "saturation_w2_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w2_r1_hit",
            {
                "snapshot": output("saturation_w2_r1_before", "snapshot"),
                "requests": output("saturation_w2_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r2_done_timeout_s"),
        params={"requests": output("saturation_w2_r2", "requests")},
    )
    case.step(
        "saturation_w2_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w2_r2_hit",
            {
                "snapshot": output("saturation_w2_r2_before", "snapshot"),
                "requests": output("saturation_w2_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_r3_done_timeout_s"),
        params={"requests": output("saturation_w2_r3", "requests")},
    )
    case.step(
        "saturation_w2_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w2_r3_hit",
            {
                "snapshot": output("saturation_w2_r3_before", "snapshot"),
                "requests": output("saturation_w2_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w2_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w2_end"),
    )
    case.step(
        "saturation_w2_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_f0_done_timeout_s"),
        params={"requests": output("saturation_w2_f0", "requests")},
    )
    case.step(
        "saturation_w2_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w2_f1_done_timeout_s"),
        params={"requests": output("saturation_w2_f1", "requests")},
    )
    case.step(
        "saturation_w3_f0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w3_f0"),
    )
    case.step(
        "saturation_w3_f0_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w3_f0_gap"),
    )
    case.step(
        "saturation_w3_f1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w3_f1"),
    )
    case.step(
        "saturation_w3_f1_gap",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w3_f1_gap"),
    )
    case.step(
        "saturation_w3_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w3_r0_before"),
    )
    case.step(
        "saturation_w3_r0",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w3_r0"),
    )
    case.step(
        "saturation_w3_r0_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w3_r0_spacing"),
    )
    case.step(
        "saturation_w3_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w3_r1_before"),
    )
    case.step(
        "saturation_w3_r1",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w3_r1"),
    )
    case.step(
        "saturation_w3_r1_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w3_r1_spacing"),
    )
    case.step(
        "saturation_w3_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w3_r2_before"),
    )
    case.step(
        "saturation_w3_r2",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w3_r2"),
    )
    case.step(
        "saturation_w3_r2_spacing",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.saturation_w3_r2_spacing"),
    )
    case.step(
        "saturation_w3_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w3_r3_before"),
    )
    case.step(
        "saturation_w3_r3",
        "request",
        params=case.value("leader_spill_nonbatch.saturation_w3_r3"),
    )
    case.step(
        "saturation_w3_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r0_done_timeout_s"),
        params={"requests": output("saturation_w3_r0", "requests")},
    )
    case.step(
        "saturation_w3_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w3_r0_hit",
            {
                "snapshot": output("saturation_w3_r0_before", "snapshot"),
                "requests": output("saturation_w3_r0", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r1_done_timeout_s"),
        params={"requests": output("saturation_w3_r1", "requests")},
    )
    case.step(
        "saturation_w3_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w3_r1_hit",
            {
                "snapshot": output("saturation_w3_r1_before", "snapshot"),
                "requests": output("saturation_w3_r1", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r2_done_timeout_s"),
        params={"requests": output("saturation_w3_r2", "requests")},
    )
    case.step(
        "saturation_w3_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w3_r2_hit",
            {
                "snapshot": output("saturation_w3_r2_before", "snapshot"),
                "requests": output("saturation_w3_r2", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_r3_done_timeout_s"),
        params={"requests": output("saturation_w3_r3", "requests")},
    )
    case.step(
        "saturation_w3_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.saturation_w3_r3_hit",
            {
                "snapshot": output("saturation_w3_r3_before", "snapshot"),
                "requests": output("saturation_w3_r3", "requests"),
            },
        ),
    )
    case.step(
        "saturation_w3_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_w3_end"),
    )
    case.step(
        "saturation_w3_f0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_f0_done_timeout_s"),
        params={"requests": output("saturation_w3_f0", "requests")},
    )
    case.step(
        "saturation_w3_f1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.saturation_w3_f1_done_timeout_s"),
        params={"requests": output("saturation_w3_f1", "requests")},
    )
    case.step(
        "recovery_settle",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.recovery_settle"),
    )
    case.step(
        "saturation_quiet",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.saturation_quiet_timeout_s"),
        params=case.value("leader_spill_nonbatch.saturation_quiet"),
    )
    case.step(
        "recovery_w0_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w0_r0_before"),
    )
    case.step(
        "recovery_w0_r0",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w0_r0"),
    )
    case.step(
        "recovery_w0_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r0_done_timeout_s"),
        params={"requests": output("recovery_w0_r0", "requests")},
    )
    case.step(
        "recovery_w0_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w0_r0_hit",
            {
                "snapshot": output("recovery_w0_r0_before", "snapshot"),
                "requests": output("recovery_w0_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w0_r1_before"),
    )
    case.step(
        "recovery_w0_r1",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w0_r1"),
    )
    case.step(
        "recovery_w0_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r1_done_timeout_s"),
        params={"requests": output("recovery_w0_r1", "requests")},
    )
    case.step(
        "recovery_w0_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w0_r1_hit",
            {
                "snapshot": output("recovery_w0_r1_before", "snapshot"),
                "requests": output("recovery_w0_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w0_r2_before"),
    )
    case.step(
        "recovery_w0_r2",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w0_r2"),
    )
    case.step(
        "recovery_w0_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r2_done_timeout_s"),
        params={"requests": output("recovery_w0_r2", "requests")},
    )
    case.step(
        "recovery_w0_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w0_r2_hit",
            {
                "snapshot": output("recovery_w0_r2_before", "snapshot"),
                "requests": output("recovery_w0_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w0_r3_before"),
    )
    case.step(
        "recovery_w0_r3",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w0_r3"),
    )
    case.step(
        "recovery_w0_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_r3_done_timeout_s"),
        params={"requests": output("recovery_w0_r3", "requests")},
    )
    case.step(
        "recovery_w0_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w0_r3_hit",
            {
                "snapshot": output("recovery_w0_r3_before", "snapshot"),
                "requests": output("recovery_w0_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w0_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w0_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w0_end"),
    )
    case.step(
        "recovery_w0_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.recovery_w0_sync"),
    )
    case.step(
        "recovery_w1_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w1_r0_before"),
    )
    case.step(
        "recovery_w1_r0",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w1_r0"),
    )
    case.step(
        "recovery_w1_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r0_done_timeout_s"),
        params={"requests": output("recovery_w1_r0", "requests")},
    )
    case.step(
        "recovery_w1_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w1_r0_hit",
            {
                "snapshot": output("recovery_w1_r0_before", "snapshot"),
                "requests": output("recovery_w1_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w1_r1_before"),
    )
    case.step(
        "recovery_w1_r1",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w1_r1"),
    )
    case.step(
        "recovery_w1_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r1_done_timeout_s"),
        params={"requests": output("recovery_w1_r1", "requests")},
    )
    case.step(
        "recovery_w1_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w1_r1_hit",
            {
                "snapshot": output("recovery_w1_r1_before", "snapshot"),
                "requests": output("recovery_w1_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w1_r2_before"),
    )
    case.step(
        "recovery_w1_r2",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w1_r2"),
    )
    case.step(
        "recovery_w1_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r2_done_timeout_s"),
        params={"requests": output("recovery_w1_r2", "requests")},
    )
    case.step(
        "recovery_w1_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w1_r2_hit",
            {
                "snapshot": output("recovery_w1_r2_before", "snapshot"),
                "requests": output("recovery_w1_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w1_r3_before"),
    )
    case.step(
        "recovery_w1_r3",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w1_r3"),
    )
    case.step(
        "recovery_w1_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_r3_done_timeout_s"),
        params={"requests": output("recovery_w1_r3", "requests")},
    )
    case.step(
        "recovery_w1_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w1_r3_hit",
            {
                "snapshot": output("recovery_w1_r3_before", "snapshot"),
                "requests": output("recovery_w1_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w1_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w1_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w1_end"),
    )
    case.step(
        "recovery_w1_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.recovery_w1_sync"),
    )
    case.step(
        "recovery_w2_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w2_r0_before"),
    )
    case.step(
        "recovery_w2_r0",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w2_r0"),
    )
    case.step(
        "recovery_w2_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r0_done_timeout_s"),
        params={"requests": output("recovery_w2_r0", "requests")},
    )
    case.step(
        "recovery_w2_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w2_r0_hit",
            {
                "snapshot": output("recovery_w2_r0_before", "snapshot"),
                "requests": output("recovery_w2_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w2_r1_before"),
    )
    case.step(
        "recovery_w2_r1",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w2_r1"),
    )
    case.step(
        "recovery_w2_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r1_done_timeout_s"),
        params={"requests": output("recovery_w2_r1", "requests")},
    )
    case.step(
        "recovery_w2_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w2_r1_hit",
            {
                "snapshot": output("recovery_w2_r1_before", "snapshot"),
                "requests": output("recovery_w2_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w2_r2_before"),
    )
    case.step(
        "recovery_w2_r2",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w2_r2"),
    )
    case.step(
        "recovery_w2_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r2_done_timeout_s"),
        params={"requests": output("recovery_w2_r2", "requests")},
    )
    case.step(
        "recovery_w2_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w2_r2_hit",
            {
                "snapshot": output("recovery_w2_r2_before", "snapshot"),
                "requests": output("recovery_w2_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w2_r3_before"),
    )
    case.step(
        "recovery_w2_r3",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w2_r3"),
    )
    case.step(
        "recovery_w2_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_r3_done_timeout_s"),
        params={"requests": output("recovery_w2_r3", "requests")},
    )
    case.step(
        "recovery_w2_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w2_r3_hit",
            {
                "snapshot": output("recovery_w2_r3_before", "snapshot"),
                "requests": output("recovery_w2_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w2_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w2_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w2_end"),
    )
    case.step(
        "recovery_w2_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.recovery_w2_sync"),
    )
    case.step(
        "recovery_w3_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w3_r0_before"),
    )
    case.step(
        "recovery_w3_r0",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w3_r0"),
    )
    case.step(
        "recovery_w3_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r0_done_timeout_s"),
        params={"requests": output("recovery_w3_r0", "requests")},
    )
    case.step(
        "recovery_w3_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w3_r0_hit",
            {
                "snapshot": output("recovery_w3_r0_before", "snapshot"),
                "requests": output("recovery_w3_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w3_r1_before"),
    )
    case.step(
        "recovery_w3_r1",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w3_r1"),
    )
    case.step(
        "recovery_w3_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r1_done_timeout_s"),
        params={"requests": output("recovery_w3_r1", "requests")},
    )
    case.step(
        "recovery_w3_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w3_r1_hit",
            {
                "snapshot": output("recovery_w3_r1_before", "snapshot"),
                "requests": output("recovery_w3_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w3_r2_before"),
    )
    case.step(
        "recovery_w3_r2",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w3_r2"),
    )
    case.step(
        "recovery_w3_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r2_done_timeout_s"),
        params={"requests": output("recovery_w3_r2", "requests")},
    )
    case.step(
        "recovery_w3_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w3_r2_hit",
            {
                "snapshot": output("recovery_w3_r2_before", "snapshot"),
                "requests": output("recovery_w3_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w3_r3_before"),
    )
    case.step(
        "recovery_w3_r3",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w3_r3"),
    )
    case.step(
        "recovery_w3_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_r3_done_timeout_s"),
        params={"requests": output("recovery_w3_r3", "requests")},
    )
    case.step(
        "recovery_w3_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w3_r3_hit",
            {
                "snapshot": output("recovery_w3_r3_before", "snapshot"),
                "requests": output("recovery_w3_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w3_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w3_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w3_end"),
    )
    case.step(
        "recovery_w3_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.recovery_w3_sync"),
    )
    case.step(
        "recovery_w4_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w4_r0_before"),
    )
    case.step(
        "recovery_w4_r0",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w4_r0"),
    )
    case.step(
        "recovery_w4_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r0_done_timeout_s"),
        params={"requests": output("recovery_w4_r0", "requests")},
    )
    case.step(
        "recovery_w4_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w4_r0_hit",
            {
                "snapshot": output("recovery_w4_r0_before", "snapshot"),
                "requests": output("recovery_w4_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w4_r1_before"),
    )
    case.step(
        "recovery_w4_r1",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w4_r1"),
    )
    case.step(
        "recovery_w4_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r1_done_timeout_s"),
        params={"requests": output("recovery_w4_r1", "requests")},
    )
    case.step(
        "recovery_w4_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w4_r1_hit",
            {
                "snapshot": output("recovery_w4_r1_before", "snapshot"),
                "requests": output("recovery_w4_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w4_r2_before"),
    )
    case.step(
        "recovery_w4_r2",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w4_r2"),
    )
    case.step(
        "recovery_w4_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r2_done_timeout_s"),
        params={"requests": output("recovery_w4_r2", "requests")},
    )
    case.step(
        "recovery_w4_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w4_r2_hit",
            {
                "snapshot": output("recovery_w4_r2_before", "snapshot"),
                "requests": output("recovery_w4_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w4_r3_before"),
    )
    case.step(
        "recovery_w4_r3",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w4_r3"),
    )
    case.step(
        "recovery_w4_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_r3_done_timeout_s"),
        params={"requests": output("recovery_w4_r3", "requests")},
    )
    case.step(
        "recovery_w4_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w4_r3_hit",
            {
                "snapshot": output("recovery_w4_r3_before", "snapshot"),
                "requests": output("recovery_w4_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w4_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w4_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w4_end"),
    )
    case.step(
        "recovery_w4_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.recovery_w4_sync"),
    )
    case.step(
        "recovery_w5_r0_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r0_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w5_r0_before"),
    )
    case.step(
        "recovery_w5_r0",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w5_r0"),
    )
    case.step(
        "recovery_w5_r0_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r0_done_timeout_s"),
        params={"requests": output("recovery_w5_r0", "requests")},
    )
    case.step(
        "recovery_w5_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w5_r0_hit",
            {
                "snapshot": output("recovery_w5_r0_before", "snapshot"),
                "requests": output("recovery_w5_r0", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_r1_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r1_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w5_r1_before"),
    )
    case.step(
        "recovery_w5_r1",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w5_r1"),
    )
    case.step(
        "recovery_w5_r1_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r1_done_timeout_s"),
        params={"requests": output("recovery_w5_r1", "requests")},
    )
    case.step(
        "recovery_w5_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w5_r1_hit",
            {
                "snapshot": output("recovery_w5_r1_before", "snapshot"),
                "requests": output("recovery_w5_r1", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_r2_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r2_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w5_r2_before"),
    )
    case.step(
        "recovery_w5_r2",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w5_r2"),
    )
    case.step(
        "recovery_w5_r2_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r2_done_timeout_s"),
        params={"requests": output("recovery_w5_r2", "requests")},
    )
    case.step(
        "recovery_w5_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w5_r2_hit",
            {
                "snapshot": output("recovery_w5_r2_before", "snapshot"),
                "requests": output("recovery_w5_r2", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_r3_before",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r3_before_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w5_r3_before"),
    )
    case.step(
        "recovery_w5_r3",
        "request",
        params=case.value("leader_spill_nonbatch.recovery_w5_r3"),
    )
    case.step(
        "recovery_w5_r3_done",
        "wait",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_r3_done_timeout_s"),
        params={"requests": output("recovery_w5_r3", "requests")},
    )
    case.step(
        "recovery_w5_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "leader_spill_nonbatch.recovery_w5_r3_hit",
            {
                "snapshot": output("recovery_w5_r3_before", "snapshot"),
                "requests": output("recovery_w5_r3", "requests"),
            },
        ),
    )
    case.step(
        "recovery_w5_end",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_w5_end_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_w5_end"),
    )
    case.step(
        "recovery_w5_sync",
        "balance_pause",
        params=case.value("leader_spill_nonbatch.recovery_w5_sync"),
    )
    case.step(
        "recovery_digest",
        "kv_snapshot",
        timeout_s=case.value("leader_spill_nonbatch.recovery_digest_timeout_s"),
        params=case.value("leader_spill_nonbatch.recovery_digest"),
    )
    case.step(
        "phase_observations",
        "kv_phase_observation",
        params=case.params(
            "leader_spill_nonbatch.phase_observations",
            {
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
                "phase_snapshots": {
                    "steer": output("steer_quiet", "snapshot"),
                    "baseline": output("baseline_digest", "snapshot"),
                    "saturation": output("saturation_quiet", "snapshot"),
                    "recovery": output("recovery_digest", "snapshot"),
                },
            },
        ),
    )
    case.step(
        "holder_flips",
        "kv_window_transitions",
        params=case.params(
            "leader_spill_nonbatch.holder_flips",
            {
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
                ]
            },
        ),
    )
    case.step(
        "all_phase_requests",
        "kv_hit_completeness",
        params=case.params(
            "leader_spill_nonbatch.all_phase_requests",
            {
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
                ]
            },
        ),
    )
    case.step(
        "baseline_hit",
        "kv_hit_rate_check",
        params=case.params(
            "leader_spill_nonbatch.baseline_hit",
            {
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
                ]
            },
        ),
    )
    case.step(
        "saturation_hit",
        "kv_hit_rate_check",
        params=case.params(
            "leader_spill_nonbatch.saturation_hit",
            {
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
                ]
            },
        ),
    )
    case.step(
        "recovery_steady_hit",
        "kv_hit_rate_check",
        params=case.params(
            "leader_spill_nonbatch.recovery_steady_hit",
            {
                "samples": [
                    output("recovery_w4_r0_hit", "sample"),
                    output("recovery_w4_r1_hit", "sample"),
                    output("recovery_w4_r2_hit", "sample"),
                    output("recovery_w4_r3_hit", "sample"),
                    output("recovery_w5_r0_hit", "sample"),
                    output("recovery_w5_r1_hit", "sample"),
                    output("recovery_w5_r2_hit", "sample"),
                    output("recovery_w5_r3_hit", "sample"),
                ]
            },
        ),
    )
    case.step(
        "replication",
        "kv_replication_check",
        params=case.params(
            "leader_spill_nonbatch.replication",
            {"snapshot": output("recovery_w5_end", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def leader_spill_multi(case):
    config = {
        name: case.number(name)
        for name in case.value("leader_spill_multi.numeric_parameters")
    }
    if config["background_blocks"] > config["receiver_retention_blocks"]:
        raise ValueError("background prefix must fit receiver retention")
    if config["normal_ms"] / 1000 >= config["interval_s"]:
        raise ValueError("background cadence must allow normal execution to finish")
    topology = case.value("leader_spill_multi.topology")
    if (
        case.environment["n_prefill"] < topology["min_prefill"]
        or not topology["min_decode"]
        <= case.environment["n_decode"]
        <= topology["max_decode"]
    ):
        raise ValueError("storm topology is outside the YAML range")
    if case.environment["prefill_cache_blocks"] <= config["hot_blocks"]:
        raise ValueError("physical execution pool must fit hot prefix plus reserve")
    case.step(
        "setup", "setup", timeout_s=case.value("leader_spill_multi.setup_timeout_s")
    )
    case.step(
        "prepare",
        "storm_prepare",
        timeout_s=case.value("leader_spill_multi.prepare_timeout_s"),
        params=config,
    )
    handle = {"storm": output("prepare", "storm")}
    for phase in ("baseline", "saturation", "recovery"):
        if phase == "saturation":
            case.step(
                "baseline_drain",
                "storm_drain",
                params=handle,
                timeout_s=case.value("leader_spill_multi.baseline_drain_timeout_s"),
            )
            case.step(
                "slow_leader",
                "storm_speed",
                params=case.params("leader_spill_multi.slow_leader", {**handle}),
            )
        if phase == "recovery":
            case.step(
                "restore_leader",
                "storm_speed",
                params=case.params("leader_spill_multi.restore_leader", {**handle}),
            )
        for index in range(config[f"{phase}_windows"]):
            case.step(
                f"{phase}_{index}",
                "storm_window",
                timeout_s=config["window_s"]
                + case.value("leader_spill_multi.window_timeout_margin_s"),
                params={**handle, "phase": phase, "index": index},
            )
    case.step(
        "all_requests",
        "storm_drain",
        params=handle,
        timeout_s=case.value("leader_spill_multi.all_requests_timeout_s"),
    )
    case.step(
        "validity",
        "storm_validate",
        params=handle,
        timeout_s=case.value("leader_spill_multi.validity_timeout_s"),
    )
    case.observe("healthy", "storm_health", params=handle)
