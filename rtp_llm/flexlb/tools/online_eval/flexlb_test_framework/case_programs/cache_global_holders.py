"""Shared holder sets, partial and full eviction, graceful holder removal, and interleaved admit/evict routing evidence."""

from ..case_config import output


def shared_batch(case):
    case.step("setup", "setup", timeout_s=case.value("shared_batch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("shared_batch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("shared_batch.slow"),
    )
    case.step("perf_sync", "balance_pause", params=case.value("shared_batch.perf_sync"))
    case.step(
        "seed_first",
        "request",
        params=case.value("shared_batch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("shared_batch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "shared_batch.first_holder", {"requests": output("seed_first", "requests")}
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("shared_batch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("shared_batch.second_terminal_timeout_s"),
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
        timeout_s=case.value("shared_batch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("shared_batch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("shared_batch.seed_quiet_timeout_s"),
        params=case.value("shared_batch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "shared_batch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("shared_batch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("shared_batch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("shared_batch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("shared_batch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("shared_batch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params=case.value("shared_batch.continuation_5"),
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_5_terminal_timeout_s"),
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params=case.value("shared_batch.continuation_6"),
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_6_terminal_timeout_s"),
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params=case.value("shared_batch.continuation_7"),
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_7_terminal_timeout_s"),
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params=case.value("shared_batch.continuation_8"),
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_8_terminal_timeout_s"),
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params=case.value("shared_batch.continuation_9"),
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_9_terminal_timeout_s"),
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "continuation_10",
        "request",
        params=case.value("shared_batch.continuation_10"),
    )
    case.step(
        "continuation_10_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_10_terminal_timeout_s"),
        params={"requests": output("continuation_10", "requests")},
    )
    case.step(
        "continuation_11",
        "request",
        params=case.value("shared_batch.continuation_11"),
    )
    case.step(
        "continuation_11_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_11_terminal_timeout_s"),
        params={"requests": output("continuation_11", "requests")},
    )
    case.step(
        "continuation_12",
        "request",
        params=case.value("shared_batch.continuation_12"),
    )
    case.step(
        "continuation_12_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_12_terminal_timeout_s"),
        params={"requests": output("continuation_12", "requests")},
    )
    case.step(
        "continuation_13",
        "request",
        params=case.value("shared_batch.continuation_13"),
    )
    case.step(
        "continuation_13_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_13_terminal_timeout_s"),
        params={"requests": output("continuation_13", "requests")},
    )
    case.step(
        "continuation_14",
        "request",
        params=case.value("shared_batch.continuation_14"),
    )
    case.step(
        "continuation_14_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_14_terminal_timeout_s"),
        params={"requests": output("continuation_14", "requests")},
    )
    case.step(
        "continuation_15",
        "request",
        params=case.value("shared_batch.continuation_15"),
    )
    case.step(
        "continuation_15_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_15_terminal_timeout_s"),
        params={"requests": output("continuation_15", "requests")},
    )
    case.step(
        "continuation_16",
        "request",
        params=case.value("shared_batch.continuation_16"),
    )
    case.step(
        "continuation_16_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_16_terminal_timeout_s"),
        params={"requests": output("continuation_16", "requests")},
    )
    case.step(
        "continuation_17",
        "request",
        params=case.value("shared_batch.continuation_17"),
    )
    case.step(
        "continuation_17_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_17_terminal_timeout_s"),
        params={"requests": output("continuation_17", "requests")},
    )
    case.step(
        "continuation_18",
        "request",
        params=case.value("shared_batch.continuation_18"),
    )
    case.step(
        "continuation_18_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_18_terminal_timeout_s"),
        params={"requests": output("continuation_18", "requests")},
    )
    case.step(
        "continuation_19",
        "request",
        params=case.value("shared_batch.continuation_19"),
    )
    case.step(
        "continuation_19_terminal",
        "wait",
        timeout_s=case.value("shared_batch.continuation_19_terminal_timeout_s"),
        params={"requests": output("continuation_19", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "shared_batch.spread",
            {
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
                ]
            },
        ),
    )
    case.step(
        "holder_union",
        "kv_union_check",
        params=case.params(
            "shared_batch.holder_union",
            {
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
            },
        ),
    )
    case.step("cleanup", "teardown")


def shared_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("shared_nonbatch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("shared_nonbatch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("shared_nonbatch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("shared_nonbatch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("shared_nonbatch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("shared_nonbatch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "shared_nonbatch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("shared_nonbatch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.second_terminal_timeout_s"),
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
        timeout_s=case.value("shared_nonbatch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("shared_nonbatch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("shared_nonbatch.seed_quiet_timeout_s"),
        params=case.value("shared_nonbatch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "shared_nonbatch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("shared_nonbatch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("shared_nonbatch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("shared_nonbatch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("shared_nonbatch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("shared_nonbatch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params=case.value("shared_nonbatch.continuation_5"),
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_5_terminal_timeout_s"),
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params=case.value("shared_nonbatch.continuation_6"),
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_6_terminal_timeout_s"),
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params=case.value("shared_nonbatch.continuation_7"),
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_7_terminal_timeout_s"),
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params=case.value("shared_nonbatch.continuation_8"),
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_8_terminal_timeout_s"),
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params=case.value("shared_nonbatch.continuation_9"),
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_9_terminal_timeout_s"),
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "continuation_10",
        "request",
        params=case.value("shared_nonbatch.continuation_10"),
    )
    case.step(
        "continuation_10_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_10_terminal_timeout_s"),
        params={"requests": output("continuation_10", "requests")},
    )
    case.step(
        "continuation_11",
        "request",
        params=case.value("shared_nonbatch.continuation_11"),
    )
    case.step(
        "continuation_11_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_11_terminal_timeout_s"),
        params={"requests": output("continuation_11", "requests")},
    )
    case.step(
        "continuation_12",
        "request",
        params=case.value("shared_nonbatch.continuation_12"),
    )
    case.step(
        "continuation_12_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_12_terminal_timeout_s"),
        params={"requests": output("continuation_12", "requests")},
    )
    case.step(
        "continuation_13",
        "request",
        params=case.value("shared_nonbatch.continuation_13"),
    )
    case.step(
        "continuation_13_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_13_terminal_timeout_s"),
        params={"requests": output("continuation_13", "requests")},
    )
    case.step(
        "continuation_14",
        "request",
        params=case.value("shared_nonbatch.continuation_14"),
    )
    case.step(
        "continuation_14_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_14_terminal_timeout_s"),
        params={"requests": output("continuation_14", "requests")},
    )
    case.step(
        "continuation_15",
        "request",
        params=case.value("shared_nonbatch.continuation_15"),
    )
    case.step(
        "continuation_15_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_15_terminal_timeout_s"),
        params={"requests": output("continuation_15", "requests")},
    )
    case.step(
        "continuation_16",
        "request",
        params=case.value("shared_nonbatch.continuation_16"),
    )
    case.step(
        "continuation_16_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_16_terminal_timeout_s"),
        params={"requests": output("continuation_16", "requests")},
    )
    case.step(
        "continuation_17",
        "request",
        params=case.value("shared_nonbatch.continuation_17"),
    )
    case.step(
        "continuation_17_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_17_terminal_timeout_s"),
        params={"requests": output("continuation_17", "requests")},
    )
    case.step(
        "continuation_18",
        "request",
        params=case.value("shared_nonbatch.continuation_18"),
    )
    case.step(
        "continuation_18_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_18_terminal_timeout_s"),
        params={"requests": output("continuation_18", "requests")},
    )
    case.step(
        "continuation_19",
        "request",
        params=case.value("shared_nonbatch.continuation_19"),
    )
    case.step(
        "continuation_19_terminal",
        "wait",
        timeout_s=case.value("shared_nonbatch.continuation_19_terminal_timeout_s"),
        params={"requests": output("continuation_19", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "shared_nonbatch.spread",
            {
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
                ]
            },
        ),
    )
    case.step(
        "holder_union",
        "kv_union_check",
        params=case.params(
            "shared_nonbatch.holder_union",
            {
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
            },
        ),
    )
    case.step("cleanup", "teardown")


def release_batch(case):
    case.step("setup", "setup", timeout_s=case.value("release_batch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("release_batch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("release_batch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("release_batch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("release_batch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("release_batch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "release_batch.first_holder", {"requests": output("seed_first", "requests")}
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("release_batch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("release_batch.second_terminal_timeout_s"),
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
        timeout_s=case.value("release_batch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("release_batch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("release_batch.seed_quiet_timeout_s"),
        params=case.value("release_batch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "release_batch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "release_first",
        "kv_evict",
        params=case.params(
            "release_batch.release_first", {"engine": output("first_holder", "engine")}
        ),
    )
    case.step(
        "release_second",
        "kv_evict",
        params=case.params(
            "release_batch.release_second",
            {"engine": output("second_holder", "engine")},
        ),
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=case.value("release_batch.released_quiet_timeout_s"),
        params=case.value("release_batch.released_quiet"),
    )
    case.step(
        "no_ghost",
        "kv_holders_check",
        params=case.params(
            "release_batch.no_ghost", {"snapshot": output("released_quiet", "snapshot")}
        ),
    )
    case.step(
        "wave",
        "request",
        timeout_s=case.value("release_batch.wave_timeout_s"),
        params=case.value("release_batch.wave"),
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=case.value("release_batch.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "release_batch.spread", {"requests": [output("wave", "requests")]}
        ),
    )
    case.step("cleanup", "teardown")


def release_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("release_nonbatch.setup_timeout_s")
    )
    case.step("fleet", "balance_snapshot", params=case.value("release_nonbatch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("release_nonbatch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("release_nonbatch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("release_nonbatch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("release_nonbatch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "release_nonbatch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("release_nonbatch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("release_nonbatch.second_terminal_timeout_s"),
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
        timeout_s=case.value("release_nonbatch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("release_nonbatch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("release_nonbatch.seed_quiet_timeout_s"),
        params=case.value("release_nonbatch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "release_nonbatch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "release_first",
        "kv_evict",
        params=case.params(
            "release_nonbatch.release_first",
            {"engine": output("first_holder", "engine")},
        ),
    )
    case.step(
        "release_second",
        "kv_evict",
        params=case.params(
            "release_nonbatch.release_second",
            {"engine": output("second_holder", "engine")},
        ),
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=case.value("release_nonbatch.released_quiet_timeout_s"),
        params=case.value("release_nonbatch.released_quiet"),
    )
    case.step(
        "no_ghost",
        "kv_holders_check",
        params=case.params(
            "release_nonbatch.no_ghost",
            {"snapshot": output("released_quiet", "snapshot")},
        ),
    )
    case.step(
        "wave",
        "request",
        timeout_s=case.value("release_nonbatch.wave_timeout_s"),
        params=case.value("release_nonbatch.wave"),
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=case.value("release_nonbatch.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "release_nonbatch.spread", {"requests": [output("wave", "requests")]}
        ),
    )
    case.step("cleanup", "teardown")


def redirect_batch(case):
    case.step("setup", "setup", timeout_s=case.value("redirect_batch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("redirect_batch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("redirect_batch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("redirect_batch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("redirect_batch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("redirect_batch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "redirect_batch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("redirect_batch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.second_terminal_timeout_s"),
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
        timeout_s=case.value("redirect_batch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("redirect_batch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("redirect_batch.seed_quiet_timeout_s"),
        params=case.value("redirect_batch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "redirect_batch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "release_first",
        "kv_evict",
        params=case.params(
            "redirect_batch.release_first", {"engine": output("first_holder", "engine")}
        ),
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=case.value("redirect_batch.released_quiet_timeout_s"),
        params=case.value("redirect_batch.released_quiet"),
    )
    case.step(
        "first_empty",
        "kv_membership_check",
        params=case.params(
            "redirect_batch.first_empty",
            {
                "snapshot": output("released_quiet", "snapshot"),
                "engine": output("first_holder", "engine"),
            },
        ),
    )
    case.step(
        "sole_holder",
        "kv_holders_check",
        params=case.params(
            "redirect_batch.sole_holder",
            {
                "snapshot": output("released_quiet", "snapshot"),
                "holders": [output("second_holder", "engine")],
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("redirect_batch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("redirect_batch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("redirect_batch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("redirect_batch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("redirect_batch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params=case.value("redirect_batch.continuation_5"),
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_5_terminal_timeout_s"),
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params=case.value("redirect_batch.continuation_6"),
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_6_terminal_timeout_s"),
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params=case.value("redirect_batch.continuation_7"),
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_7_terminal_timeout_s"),
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params=case.value("redirect_batch.continuation_8"),
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_8_terminal_timeout_s"),
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params=case.value("redirect_batch.continuation_9"),
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=case.value("redirect_batch.continuation_9_terminal_timeout_s"),
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "redirect",
        "kv_fidelity_check",
        params=case.params(
            "redirect_batch.redirect",
            {
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
                "holder": output("second_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def redirect_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("redirect_nonbatch.setup_timeout_s")
    )
    case.step("fleet", "balance_snapshot", params=case.value("redirect_nonbatch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("redirect_nonbatch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("redirect_nonbatch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("redirect_nonbatch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("redirect_nonbatch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "redirect_nonbatch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("redirect_nonbatch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.second_terminal_timeout_s"),
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
        timeout_s=case.value("redirect_nonbatch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("redirect_nonbatch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("redirect_nonbatch.seed_quiet_timeout_s"),
        params=case.value("redirect_nonbatch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "redirect_nonbatch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "release_first",
        "kv_evict",
        params=case.params(
            "redirect_nonbatch.release_first",
            {"engine": output("first_holder", "engine")},
        ),
    )
    case.step(
        "released_quiet",
        "kv_snapshot",
        timeout_s=case.value("redirect_nonbatch.released_quiet_timeout_s"),
        params=case.value("redirect_nonbatch.released_quiet"),
    )
    case.step(
        "first_empty",
        "kv_membership_check",
        params=case.params(
            "redirect_nonbatch.first_empty",
            {
                "snapshot": output("released_quiet", "snapshot"),
                "engine": output("first_holder", "engine"),
            },
        ),
    )
    case.step(
        "sole_holder",
        "kv_holders_check",
        params=case.params(
            "redirect_nonbatch.sole_holder",
            {
                "snapshot": output("released_quiet", "snapshot"),
                "holders": [output("second_holder", "engine")],
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("redirect_nonbatch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("redirect_nonbatch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("redirect_nonbatch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("redirect_nonbatch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("redirect_nonbatch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params=case.value("redirect_nonbatch.continuation_5"),
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_5_terminal_timeout_s"),
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params=case.value("redirect_nonbatch.continuation_6"),
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_6_terminal_timeout_s"),
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params=case.value("redirect_nonbatch.continuation_7"),
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_7_terminal_timeout_s"),
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params=case.value("redirect_nonbatch.continuation_8"),
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_8_terminal_timeout_s"),
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params=case.value("redirect_nonbatch.continuation_9"),
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=case.value("redirect_nonbatch.continuation_9_terminal_timeout_s"),
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "redirect",
        "kv_fidelity_check",
        params=case.params(
            "redirect_nonbatch.redirect",
            {
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
                "holder": output("second_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def down_batch(case):
    case.step("setup", "setup", timeout_s=case.value("down_batch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("down_batch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("down_batch.slow"),
    )
    case.step("perf_sync", "balance_pause", params=case.value("down_batch.perf_sync"))
    case.step(
        "seed_first",
        "request",
        params=case.value("down_batch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("down_batch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "down_batch.first_holder", {"requests": output("seed_first", "requests")}
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("down_batch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("down_batch.second_terminal_timeout_s"),
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
        timeout_s=case.value("down_batch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("down_batch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("down_batch.seed_quiet_timeout_s"),
        params=case.value("down_batch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "down_batch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "third_worker",
        "kv_snapshot",
        timeout_s=case.value("down_batch.third_worker_timeout_s"),
        params=case.params(
            "down_batch.third_worker",
            {
                "exclude": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ]
            },
        ),
    )
    case.step(
        "third_empty",
        "kv_holders_check",
        params=case.params(
            "down_batch.third_empty", {"snapshot": output("third_worker", "snapshot")}
        ),
    )
    case.step(
        "remove_holder",
        "elastic_remove",
        timeout_s=case.value("down_batch.remove_holder_timeout_s"),
        params=case.params(
            "down_batch.remove_holder", {"engine": output("first_holder", "engine")}
        ),
    )
    case.step(
        "master_converged",
        "kv_master_alive",
        timeout_s=case.value("down_batch.master_converged_timeout_s"),
        params=case.value("down_batch.master_converged"),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("down_batch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("down_batch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("down_batch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("down_batch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("down_batch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("down_batch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("down_batch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("down_batch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("down_batch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("down_batch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "survivor_snapshot",
        "kv_snapshot",
        timeout_s=case.value("down_batch.survivor_snapshot_timeout_s"),
        params=case.params(
            "down_batch.survivor_snapshot",
            {"exclude": [output("first_holder", "engine")]},
        ),
    )
    case.step(
        "survivor_kept",
        "kv_membership_check",
        params=case.params(
            "down_batch.survivor_kept",
            {
                "snapshot": output("survivor_snapshot", "snapshot"),
                "engine": output("second_holder", "engine"),
            },
        ),
    )
    case.step(
        "survivor_fidelity",
        "kv_fidelity_check",
        params=case.params(
            "down_batch.survivor_fidelity",
            {
                "requests": [
                    output("continuation_0", "requests"),
                    output("continuation_1", "requests"),
                    output("continuation_2", "requests"),
                    output("continuation_3", "requests"),
                    output("continuation_4", "requests"),
                ],
                "holder": output("second_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def down_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("down_nonbatch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("down_nonbatch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("down_nonbatch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("down_nonbatch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("down_nonbatch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("down_nonbatch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "down_nonbatch.first_holder", {"requests": output("seed_first", "requests")}
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("down_nonbatch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("down_nonbatch.second_terminal_timeout_s"),
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
        timeout_s=case.value("down_nonbatch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("down_nonbatch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("down_nonbatch.seed_quiet_timeout_s"),
        params=case.value("down_nonbatch.seed_quiet"),
    )
    case.step(
        "shared_holders",
        "kv_holders_check",
        params=case.params(
            "down_nonbatch.shared_holders",
            {
                "snapshot": output("seed_quiet", "snapshot"),
                "holders": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ],
            },
        ),
    )
    case.step(
        "third_worker",
        "kv_snapshot",
        timeout_s=case.value("down_nonbatch.third_worker_timeout_s"),
        params=case.params(
            "down_nonbatch.third_worker",
            {
                "exclude": [
                    output("first_holder", "engine"),
                    output("second_holder", "engine"),
                ]
            },
        ),
    )
    case.step(
        "third_empty",
        "kv_holders_check",
        params=case.params(
            "down_nonbatch.third_empty",
            {"snapshot": output("third_worker", "snapshot")},
        ),
    )
    case.step(
        "remove_holder",
        "elastic_remove",
        timeout_s=case.value("down_nonbatch.remove_holder_timeout_s"),
        params=case.params(
            "down_nonbatch.remove_holder", {"engine": output("first_holder", "engine")}
        ),
    )
    case.step(
        "master_converged",
        "kv_master_alive",
        timeout_s=case.value("down_nonbatch.master_converged_timeout_s"),
        params=case.value("down_nonbatch.master_converged"),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("down_nonbatch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("down_nonbatch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("down_nonbatch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("down_nonbatch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("down_nonbatch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("down_nonbatch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("down_nonbatch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("down_nonbatch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("down_nonbatch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("down_nonbatch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "survivor_snapshot",
        "kv_snapshot",
        timeout_s=case.value("down_nonbatch.survivor_snapshot_timeout_s"),
        params=case.params(
            "down_nonbatch.survivor_snapshot",
            {"exclude": [output("first_holder", "engine")]},
        ),
    )
    case.step(
        "survivor_kept",
        "kv_membership_check",
        params=case.params(
            "down_nonbatch.survivor_kept",
            {
                "snapshot": output("survivor_snapshot", "snapshot"),
                "engine": output("second_holder", "engine"),
            },
        ),
    )
    case.step(
        "survivor_fidelity",
        "kv_fidelity_check",
        params=case.params(
            "down_nonbatch.survivor_fidelity",
            {
                "requests": [
                    output("continuation_0", "requests"),
                    output("continuation_1", "requests"),
                    output("continuation_2", "requests"),
                    output("continuation_3", "requests"),
                    output("continuation_4", "requests"),
                ],
                "holder": output("second_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def mixed_batch(case):
    case.step("setup", "setup", timeout_s=case.value("mixed_batch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("mixed_batch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("mixed_batch.slow"),
    )
    case.step("perf_sync", "balance_pause", params=case.value("mixed_batch.perf_sync"))
    case.step(
        "seed_first",
        "request",
        params=case.value("mixed_batch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("mixed_batch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "mixed_batch.first_holder", {"requests": output("seed_first", "requests")}
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("mixed_batch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.second_terminal_timeout_s"),
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
        timeout_s=case.value("mixed_batch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("mixed_batch.restore_perf"),
    )
    case.step(
        "first_release",
        "kv_evict",
        params=case.params(
            "mixed_batch.first_release", {"engine": output("first_holder", "engine")}
        ),
    )
    case.step(
        "family_one_0",
        "request",
        params=case.value("mixed_batch.family_one_0"),
    )
    case.step(
        "family_one_0_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.family_one_0_terminal_timeout_s"),
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
        params=case.params(
            "mixed_batch.second_release", {"engine": output("second_holder", "engine")}
        ),
    )
    case.step(
        "family_two_fleet",
        "balance_snapshot",
        params=case.value("mixed_batch.family_two_fleet"),
    )
    case.step(
        "family_two_slow",
        "engine_control",
        params=case.value("mixed_batch.family_two_slow"),
    )
    case.step(
        "family_two_perf_sync",
        "balance_pause",
        params=case.value("mixed_batch.family_two_perf_sync"),
    )
    case.step(
        "family_two_seed_first",
        "request",
        params=case.value("mixed_batch.family_two_seed_first"),
    )
    case.step(
        "family_two_first_pending",
        "balance_pending",
        timeout_s=case.value("mixed_batch.family_two_first_pending_timeout_s"),
        params={
            "requests": output("family_two_seed_first", "requests"),
            "fleet": output("family_two_fleet", "snapshot"),
        },
    )
    case.step(
        "family_two_first_holder",
        "kv_landing",
        params=case.params(
            "mixed_batch.family_two_first_holder",
            {"requests": output("family_two_seed_first", "requests")},
        ),
    )
    case.step(
        "family_two_seed_second",
        "request",
        params=case.value("mixed_batch.family_two_seed_second"),
    )
    case.step(
        "family_two_second_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.family_two_second_terminal_timeout_s"),
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
        timeout_s=case.value("mixed_batch.family_two_first_terminal_timeout_s"),
        params={"requests": output("family_two_seed_first", "requests")},
    )
    case.step(
        "family_two_restore_perf",
        "engine_control",
        params=case.value("mixed_batch.family_two_restore_perf"),
    )
    case.step(
        "family_two_release",
        "kv_evict",
        params=case.params(
            "mixed_batch.family_two_release",
            {"engine": output("family_two_first_holder", "engine")},
        ),
    )
    case.step(
        "mixed_quiet",
        "kv_snapshot",
        timeout_s=case.value("mixed_batch.mixed_quiet_timeout_s"),
        params=case.value("mixed_batch.mixed_quiet"),
    )
    case.step(
        "family_zero_empty",
        "kv_holders_check",
        params=case.params(
            "mixed_batch.family_zero_empty",
            {"snapshot": output("mixed_quiet", "snapshot")},
        ),
    )
    case.step(
        "family_one_sole",
        "kv_holders_check",
        params=case.params(
            "mixed_batch.family_one_sole",
            {
                "snapshot": output("mixed_quiet", "snapshot"),
                "holders": [output("family_one_holder", "engine")],
            },
        ),
    )
    case.step(
        "family_two_sole",
        "kv_holders_check",
        params=case.params(
            "mixed_batch.family_two_sole",
            {
                "snapshot": output("mixed_quiet", "snapshot"),
                "holders": [output("family_two_second_holder", "engine")],
            },
        ),
    )
    case.step(
        "wave",
        "request",
        timeout_s=case.value("mixed_batch.wave_timeout_s"),
        params=case.value("mixed_batch.wave"),
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "mixed_batch.spread", {"requests": [output("wave", "requests")]}
        ),
    )
    case.step(
        "continuation_f1_0",
        "request",
        params=case.value("mixed_batch.continuation_f1_0"),
    )
    case.step(
        "continuation_f1_0_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f1_0_terminal_timeout_s"),
        params={"requests": output("continuation_f1_0", "requests")},
    )
    case.step(
        "continuation_f1_1",
        "request",
        params=case.value("mixed_batch.continuation_f1_1"),
    )
    case.step(
        "continuation_f1_1_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f1_1_terminal_timeout_s"),
        params={"requests": output("continuation_f1_1", "requests")},
    )
    case.step(
        "continuation_f1_2",
        "request",
        params=case.value("mixed_batch.continuation_f1_2"),
    )
    case.step(
        "continuation_f1_2_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f1_2_terminal_timeout_s"),
        params={"requests": output("continuation_f1_2", "requests")},
    )
    case.step(
        "continuation_f1_3",
        "request",
        params=case.value("mixed_batch.continuation_f1_3"),
    )
    case.step(
        "continuation_f1_3_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f1_3_terminal_timeout_s"),
        params={"requests": output("continuation_f1_3", "requests")},
    )
    case.step(
        "continuation_f1_4",
        "request",
        params=case.value("mixed_batch.continuation_f1_4"),
    )
    case.step(
        "continuation_f1_4_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f1_4_terminal_timeout_s"),
        params={"requests": output("continuation_f1_4", "requests")},
    )
    case.step(
        "fidelity_f1",
        "kv_fidelity_check",
        params=case.params(
            "mixed_batch.fidelity_f1",
            {
                "requests": [
                    output("continuation_f1_0", "requests"),
                    output("continuation_f1_1", "requests"),
                    output("continuation_f1_2", "requests"),
                    output("continuation_f1_3", "requests"),
                    output("continuation_f1_4", "requests"),
                ],
                "holder": output("family_one_holder", "engine"),
            },
        ),
    )
    case.step(
        "continuation_f2_0",
        "request",
        params=case.value("mixed_batch.continuation_f2_0"),
    )
    case.step(
        "continuation_f2_0_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f2_0_terminal_timeout_s"),
        params={"requests": output("continuation_f2_0", "requests")},
    )
    case.step(
        "continuation_f2_1",
        "request",
        params=case.value("mixed_batch.continuation_f2_1"),
    )
    case.step(
        "continuation_f2_1_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f2_1_terminal_timeout_s"),
        params={"requests": output("continuation_f2_1", "requests")},
    )
    case.step(
        "continuation_f2_2",
        "request",
        params=case.value("mixed_batch.continuation_f2_2"),
    )
    case.step(
        "continuation_f2_2_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f2_2_terminal_timeout_s"),
        params={"requests": output("continuation_f2_2", "requests")},
    )
    case.step(
        "continuation_f2_3",
        "request",
        params=case.value("mixed_batch.continuation_f2_3"),
    )
    case.step(
        "continuation_f2_3_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f2_3_terminal_timeout_s"),
        params={"requests": output("continuation_f2_3", "requests")},
    )
    case.step(
        "continuation_f2_4",
        "request",
        params=case.value("mixed_batch.continuation_f2_4"),
    )
    case.step(
        "continuation_f2_4_terminal",
        "wait",
        timeout_s=case.value("mixed_batch.continuation_f2_4_terminal_timeout_s"),
        params={"requests": output("continuation_f2_4", "requests")},
    )
    case.step(
        "fidelity_f2",
        "kv_fidelity_check",
        params=case.params(
            "mixed_batch.fidelity_f2",
            {
                "requests": [
                    output("continuation_f2_0", "requests"),
                    output("continuation_f2_1", "requests"),
                    output("continuation_f2_2", "requests"),
                    output("continuation_f2_3", "requests"),
                    output("continuation_f2_4", "requests"),
                ],
                "holder": output("family_two_second_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def mixed_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("mixed_nonbatch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("mixed_nonbatch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("mixed_nonbatch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("mixed_nonbatch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("mixed_nonbatch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("mixed_nonbatch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "mixed_nonbatch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("mixed_nonbatch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.second_terminal_timeout_s"),
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
        timeout_s=case.value("mixed_nonbatch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("mixed_nonbatch.restore_perf"),
    )
    case.step(
        "first_release",
        "kv_evict",
        params=case.params(
            "mixed_nonbatch.first_release", {"engine": output("first_holder", "engine")}
        ),
    )
    case.step(
        "family_one_0",
        "request",
        params=case.value("mixed_nonbatch.family_one_0"),
    )
    case.step(
        "family_one_0_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.family_one_0_terminal_timeout_s"),
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
        params=case.params(
            "mixed_nonbatch.second_release",
            {"engine": output("second_holder", "engine")},
        ),
    )
    case.step(
        "family_two_fleet",
        "balance_snapshot",
        params=case.value("mixed_nonbatch.family_two_fleet"),
    )
    case.step(
        "family_two_slow",
        "engine_control",
        params=case.value("mixed_nonbatch.family_two_slow"),
    )
    case.step(
        "family_two_perf_sync",
        "balance_pause",
        params=case.value("mixed_nonbatch.family_two_perf_sync"),
    )
    case.step(
        "family_two_seed_first",
        "request",
        params=case.value("mixed_nonbatch.family_two_seed_first"),
    )
    case.step(
        "family_two_first_pending",
        "balance_pending",
        timeout_s=case.value("mixed_nonbatch.family_two_first_pending_timeout_s"),
        params={
            "requests": output("family_two_seed_first", "requests"),
            "fleet": output("family_two_fleet", "snapshot"),
        },
    )
    case.step(
        "family_two_first_holder",
        "kv_landing",
        params=case.params(
            "mixed_nonbatch.family_two_first_holder",
            {"requests": output("family_two_seed_first", "requests")},
        ),
    )
    case.step(
        "family_two_seed_second",
        "request",
        params=case.value("mixed_nonbatch.family_two_seed_second"),
    )
    case.step(
        "family_two_second_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.family_two_second_terminal_timeout_s"),
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
        timeout_s=case.value("mixed_nonbatch.family_two_first_terminal_timeout_s"),
        params={"requests": output("family_two_seed_first", "requests")},
    )
    case.step(
        "family_two_restore_perf",
        "engine_control",
        params=case.value("mixed_nonbatch.family_two_restore_perf"),
    )
    case.step(
        "family_two_release",
        "kv_evict",
        params=case.params(
            "mixed_nonbatch.family_two_release",
            {"engine": output("family_two_first_holder", "engine")},
        ),
    )
    case.step(
        "mixed_quiet",
        "kv_snapshot",
        timeout_s=case.value("mixed_nonbatch.mixed_quiet_timeout_s"),
        params=case.value("mixed_nonbatch.mixed_quiet"),
    )
    case.step(
        "family_zero_empty",
        "kv_holders_check",
        params=case.params(
            "mixed_nonbatch.family_zero_empty",
            {"snapshot": output("mixed_quiet", "snapshot")},
        ),
    )
    case.step(
        "family_one_sole",
        "kv_holders_check",
        params=case.params(
            "mixed_nonbatch.family_one_sole",
            {
                "snapshot": output("mixed_quiet", "snapshot"),
                "holders": [output("family_one_holder", "engine")],
            },
        ),
    )
    case.step(
        "family_two_sole",
        "kv_holders_check",
        params=case.params(
            "mixed_nonbatch.family_two_sole",
            {
                "snapshot": output("mixed_quiet", "snapshot"),
                "holders": [output("family_two_second_holder", "engine")],
            },
        ),
    )
    case.step(
        "wave",
        "request",
        timeout_s=case.value("mixed_nonbatch.wave_timeout_s"),
        params=case.value("mixed_nonbatch.wave"),
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "mixed_nonbatch.spread", {"requests": [output("wave", "requests")]}
        ),
    )
    case.step(
        "continuation_f1_0",
        "request",
        params=case.value("mixed_nonbatch.continuation_f1_0"),
    )
    case.step(
        "continuation_f1_0_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f1_0_terminal_timeout_s"),
        params={"requests": output("continuation_f1_0", "requests")},
    )
    case.step(
        "continuation_f1_1",
        "request",
        params=case.value("mixed_nonbatch.continuation_f1_1"),
    )
    case.step(
        "continuation_f1_1_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f1_1_terminal_timeout_s"),
        params={"requests": output("continuation_f1_1", "requests")},
    )
    case.step(
        "continuation_f1_2",
        "request",
        params=case.value("mixed_nonbatch.continuation_f1_2"),
    )
    case.step(
        "continuation_f1_2_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f1_2_terminal_timeout_s"),
        params={"requests": output("continuation_f1_2", "requests")},
    )
    case.step(
        "continuation_f1_3",
        "request",
        params=case.value("mixed_nonbatch.continuation_f1_3"),
    )
    case.step(
        "continuation_f1_3_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f1_3_terminal_timeout_s"),
        params={"requests": output("continuation_f1_3", "requests")},
    )
    case.step(
        "continuation_f1_4",
        "request",
        params=case.value("mixed_nonbatch.continuation_f1_4"),
    )
    case.step(
        "continuation_f1_4_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f1_4_terminal_timeout_s"),
        params={"requests": output("continuation_f1_4", "requests")},
    )
    case.step(
        "fidelity_f1",
        "kv_fidelity_check",
        params=case.params(
            "mixed_nonbatch.fidelity_f1",
            {
                "requests": [
                    output("continuation_f1_0", "requests"),
                    output("continuation_f1_1", "requests"),
                    output("continuation_f1_2", "requests"),
                    output("continuation_f1_3", "requests"),
                    output("continuation_f1_4", "requests"),
                ],
                "holder": output("family_one_holder", "engine"),
            },
        ),
    )
    case.step(
        "continuation_f2_0",
        "request",
        params=case.value("mixed_nonbatch.continuation_f2_0"),
    )
    case.step(
        "continuation_f2_0_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f2_0_terminal_timeout_s"),
        params={"requests": output("continuation_f2_0", "requests")},
    )
    case.step(
        "continuation_f2_1",
        "request",
        params=case.value("mixed_nonbatch.continuation_f2_1"),
    )
    case.step(
        "continuation_f2_1_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f2_1_terminal_timeout_s"),
        params={"requests": output("continuation_f2_1", "requests")},
    )
    case.step(
        "continuation_f2_2",
        "request",
        params=case.value("mixed_nonbatch.continuation_f2_2"),
    )
    case.step(
        "continuation_f2_2_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f2_2_terminal_timeout_s"),
        params={"requests": output("continuation_f2_2", "requests")},
    )
    case.step(
        "continuation_f2_3",
        "request",
        params=case.value("mixed_nonbatch.continuation_f2_3"),
    )
    case.step(
        "continuation_f2_3_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f2_3_terminal_timeout_s"),
        params={"requests": output("continuation_f2_3", "requests")},
    )
    case.step(
        "continuation_f2_4",
        "request",
        params=case.value("mixed_nonbatch.continuation_f2_4"),
    )
    case.step(
        "continuation_f2_4_terminal",
        "wait",
        timeout_s=case.value("mixed_nonbatch.continuation_f2_4_terminal_timeout_s"),
        params={"requests": output("continuation_f2_4", "requests")},
    )
    case.step(
        "fidelity_f2",
        "kv_fidelity_check",
        params=case.params(
            "mixed_nonbatch.fidelity_f2",
            {
                "requests": [
                    output("continuation_f2_0", "requests"),
                    output("continuation_f2_1", "requests"),
                    output("continuation_f2_2", "requests"),
                    output("continuation_f2_3", "requests"),
                    output("continuation_f2_4", "requests"),
                ],
                "holder": output("family_two_second_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")
