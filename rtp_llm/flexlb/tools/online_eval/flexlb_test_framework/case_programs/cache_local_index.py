"""Ledger-separated shared seeding, explicit per-engine gap/tail eviction, contiguous-prefix checks and five serial routing continuations."""

from ..case_config import output


def continuity_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("continuity_batch.setup_timeout_s")
    )
    case.step("fleet", "balance_snapshot", params=case.value("continuity_batch.fleet"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("continuity_batch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("continuity_batch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("continuity_batch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("continuity_batch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "continuity_batch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("continuity_batch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("continuity_batch.second_terminal_timeout_s"),
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
        timeout_s=case.value("continuity_batch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("continuity_batch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("continuity_batch.seed_quiet_timeout_s"),
        params=case.value("continuity_batch.seed_quiet"),
    )
    case.step(
        "carve_gap",
        "kv_evict",
        params=case.params(
            "continuity_batch.carve_gap", {"engine": output("first_holder", "engine")}
        ),
    )
    case.step(
        "carve_tail",
        "kv_evict",
        params=case.params(
            "continuity_batch.carve_tail", {"engine": output("second_holder", "engine")}
        ),
    )
    case.step(
        "carve_quiet",
        "kv_snapshot",
        timeout_s=case.value("continuity_batch.carve_quiet_timeout_s"),
        params=case.value("continuity_batch.carve_quiet"),
    )
    case.step(
        "gap_prefix",
        "kv_prefix_check",
        params=case.params(
            "continuity_batch.gap_prefix",
            {
                "snapshot": output("carve_quiet", "snapshot"),
                "engine": output("first_holder", "engine"),
            },
        ),
    )
    case.step(
        "tail_prefix",
        "kv_prefix_check",
        params=case.params(
            "continuity_batch.tail_prefix",
            {
                "snapshot": output("carve_quiet", "snapshot"),
                "engine": output("second_holder", "engine"),
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("continuity_batch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("continuity_batch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("continuity_batch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("continuity_batch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("continuity_batch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("continuity_batch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("continuity_batch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("continuity_batch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("continuity_batch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("continuity_batch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "affinity",
        "kv_affinity_check",
        params=case.params(
            "continuity_batch.affinity",
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


def continuity_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("continuity_nonbatch.setup_timeout_s")
    )
    case.step(
        "fleet", "balance_snapshot", params=case.value("continuity_nonbatch.fleet")
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("continuity_nonbatch.slow"),
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("continuity_nonbatch.perf_sync")
    )
    case.step(
        "seed_first",
        "request",
        params=case.value("continuity_nonbatch.seed_first"),
    )
    case.step(
        "first_pending",
        "balance_pending",
        timeout_s=case.value("continuity_nonbatch.first_pending_timeout_s"),
        params={
            "requests": output("seed_first", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "first_holder",
        "kv_landing",
        params=case.params(
            "continuity_nonbatch.first_holder",
            {"requests": output("seed_first", "requests")},
        ),
    )
    case.step(
        "seed_second",
        "request",
        params=case.value("continuity_nonbatch.seed_second"),
    )
    case.step(
        "second_terminal",
        "wait",
        timeout_s=case.value("continuity_nonbatch.second_terminal_timeout_s"),
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
        timeout_s=case.value("continuity_nonbatch.first_terminal_timeout_s"),
        params={"requests": output("seed_first", "requests")},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("continuity_nonbatch.restore_perf"),
    )
    case.step(
        "seed_quiet",
        "kv_snapshot",
        timeout_s=case.value("continuity_nonbatch.seed_quiet_timeout_s"),
        params=case.value("continuity_nonbatch.seed_quiet"),
    )
    case.step(
        "carve_gap",
        "kv_evict",
        params=case.params(
            "continuity_nonbatch.carve_gap",
            {"engine": output("first_holder", "engine")},
        ),
    )
    case.step(
        "carve_tail",
        "kv_evict",
        params=case.params(
            "continuity_nonbatch.carve_tail",
            {"engine": output("second_holder", "engine")},
        ),
    )
    case.step(
        "carve_quiet",
        "kv_snapshot",
        timeout_s=case.value("continuity_nonbatch.carve_quiet_timeout_s"),
        params=case.value("continuity_nonbatch.carve_quiet"),
    )
    case.step(
        "gap_prefix",
        "kv_prefix_check",
        params=case.params(
            "continuity_nonbatch.gap_prefix",
            {
                "snapshot": output("carve_quiet", "snapshot"),
                "engine": output("first_holder", "engine"),
            },
        ),
    )
    case.step(
        "tail_prefix",
        "kv_prefix_check",
        params=case.params(
            "continuity_nonbatch.tail_prefix",
            {
                "snapshot": output("carve_quiet", "snapshot"),
                "engine": output("second_holder", "engine"),
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("continuity_nonbatch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("continuity_nonbatch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("continuity_nonbatch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("continuity_nonbatch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("continuity_nonbatch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("continuity_nonbatch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("continuity_nonbatch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("continuity_nonbatch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("continuity_nonbatch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("continuity_nonbatch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "affinity",
        "kv_affinity_check",
        params=case.params(
            "continuity_nonbatch.affinity",
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


def evict_batch(case):
    case.step("setup", "setup", timeout_s=case.value("evict_batch.setup_timeout_s"))
    case.step(
        "prime",
        "request",
        params=case.value("evict_batch.prime"),
    )
    case.step(
        "prime_terminal",
        "wait",
        timeout_s=case.value("evict_batch.prime_terminal_timeout_s"),
        params={"requests": output("prime", "requests")},
    )
    case.step(
        "holder",
        "kv_landing",
        params=case.params(
            "evict_batch.holder", {"requests": output("prime", "requests")}
        ),
    )
    case.step(
        "prime_quiet",
        "kv_snapshot",
        timeout_s=case.value("evict_batch.prime_quiet_timeout_s"),
        params=case.value("evict_batch.prime_quiet"),
    )
    case.step(
        "positive",
        "request",
        params=case.value("evict_batch.positive"),
    )
    case.step(
        "positive_terminal",
        "wait",
        timeout_s=case.value("evict_batch.positive_terminal_timeout_s"),
        params={"requests": output("positive", "requests")},
    )
    case.step(
        "positive_holder",
        "kv_landing",
        params=case.params(
            "evict_batch.positive_holder", {"requests": output("positive", "requests")}
        ),
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
        params=case.params("evict_batch.evict", {"engine": output("holder", "engine")}),
    )
    case.step(
        "eviction_quiet",
        "kv_snapshot",
        timeout_s=case.value("evict_batch.eviction_quiet_timeout_s"),
        params=case.value("evict_batch.eviction_quiet"),
    )
    case.step(
        "eviction_membership",
        "kv_membership_check",
        params=case.params(
            "evict_batch.eviction_membership",
            {
                "snapshot": output("eviction_quiet", "snapshot"),
                "engine": output("holder", "engine"),
            },
        ),
    )
    case.step(
        "wave",
        "request",
        params=case.value("evict_batch.wave"),
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=case.value("evict_batch.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "evict_batch.spread", {"requests": [output("wave", "requests")]}
        ),
    )
    case.step("cleanup", "teardown")


def isolation_batch(case):
    case.step("setup", "setup", timeout_s=case.value("isolation_batch.setup_timeout_s"))
    case.step("fleet", "balance_snapshot", params=case.value("isolation_batch.fleet"))
    case.step(
        "perf_sync", "balance_pause", params=case.value("isolation_batch.perf_sync")
    )
    case.step(
        "seed_a",
        "request",
        params=case.value("isolation_batch.seed_a"),
    )
    case.step(
        "a_pending",
        "balance_pending",
        timeout_s=case.value("isolation_batch.a_pending_timeout_s"),
        params={
            "requests": output("seed_a", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "a_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.a_terminal_timeout_s"),
        params={"requests": output("seed_a", "requests")},
    )
    case.step(
        "a_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.a_holder", {"requests": output("seed_a", "requests")}
        ),
    )
    case.step(
        "seed_b",
        "request",
        params=case.value("isolation_batch.seed_b"),
    )
    case.step(
        "b_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.b_terminal_timeout_s"),
        params={"requests": output("seed_b", "requests")},
    )
    case.step(
        "b_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.b_holder", {"requests": output("seed_b", "requests")}
        ),
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
        "b_perf_sync", "balance_pause", params=case.value("isolation_batch.b_perf_sync")
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step("filler_0", "request", params=case.value("isolation_batch.filler"))
    case.step(
        "filler_0_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.filler_holder",
            {"requests": output("filler_0", "requests")},
        ),
    )
    case.step(
        "filler_0_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_0_holder", "engine"),
        },
    )
    case.step(
        "admit_0",
        "request",
        params=case.value("isolation_batch.admit_0"),
    )
    case.step(
        "admit_0_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.admit_0_terminal_timeout_s"),
        params={"requests": output("admit_0", "requests")},
    )
    case.step(
        "admit_0_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.admit_0_holder",
            {"requests": output("admit_0", "requests")},
        ),
    )
    case.step(
        "admit_0_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_0_holder", "engine"),
        },
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step(
        "filler_0_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.filler_terminal_timeout_s"),
        params={"requests": output("filler_0", "requests")},
    )
    case.step("filler_1", "request", params=case.value("isolation_batch.filler"))
    case.step(
        "filler_1_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.filler_holder",
            {"requests": output("filler_1", "requests")},
        ),
    )
    case.step(
        "filler_1_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_1_holder", "engine"),
        },
    )
    case.step(
        "admit_1",
        "request",
        params=case.value("isolation_batch.admit_1"),
    )
    case.step(
        "admit_1_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.admit_1_terminal_timeout_s"),
        params={"requests": output("admit_1", "requests")},
    )
    case.step(
        "admit_1_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.admit_1_holder",
            {"requests": output("admit_1", "requests")},
        ),
    )
    case.step(
        "admit_1_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_1_holder", "engine"),
        },
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step(
        "filler_1_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.filler_terminal_timeout_s"),
        params={"requests": output("filler_1", "requests")},
    )
    case.step("filler_2", "request", params=case.value("isolation_batch.filler"))
    case.step(
        "filler_2_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.filler_holder",
            {"requests": output("filler_2", "requests")},
        ),
    )
    case.step(
        "filler_2_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_2_holder", "engine"),
        },
    )
    case.step(
        "admit_2",
        "request",
        params=case.value("isolation_batch.admit_2"),
    )
    case.step(
        "admit_2_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.admit_2_terminal_timeout_s"),
        params={"requests": output("admit_2", "requests")},
    )
    case.step(
        "admit_2_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.admit_2_holder",
            {"requests": output("admit_2", "requests")},
        ),
    )
    case.step(
        "admit_2_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_2_holder", "engine"),
        },
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step(
        "filler_2_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.filler_terminal_timeout_s"),
        params={"requests": output("filler_2", "requests")},
    )
    case.step("filler_3", "request", params=case.value("isolation_batch.filler"))
    case.step(
        "filler_3_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.filler_holder",
            {"requests": output("filler_3", "requests")},
        ),
    )
    case.step(
        "filler_3_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_3_holder", "engine"),
        },
    )
    case.step(
        "admit_3",
        "request",
        params=case.value("isolation_batch.admit_3"),
    )
    case.step(
        "admit_3_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.admit_3_terminal_timeout_s"),
        params={"requests": output("admit_3", "requests")},
    )
    case.step(
        "admit_3_holder",
        "kv_landing",
        params=case.params(
            "isolation_batch.admit_3_holder",
            {"requests": output("admit_3", "requests")},
        ),
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
        "filler_3_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.filler_terminal_timeout_s"),
        params={"requests": output("filler_3", "requests")},
    )
    case.step(
        "admission_quiet",
        "kv_snapshot",
        timeout_s=case.value("isolation_batch.admission_quiet_timeout_s"),
        params=case.value("isolation_batch.admission_quiet"),
    )
    case.step(
        "a_membership",
        "kv_membership_check",
        params=case.params(
            "isolation_batch.a_membership",
            {
                "snapshot": output("admission_quiet", "snapshot"),
                "engine": output("a_holder", "engine"),
            },
        ),
    )
    case.step(
        "b_isolation",
        "kv_membership_check",
        params=case.params(
            "isolation_batch.b_isolation",
            {
                "snapshot": output("admission_quiet", "snapshot"),
                "engine": output("b_holder", "engine"),
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("isolation_batch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("isolation_batch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("isolation_batch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("isolation_batch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("isolation_batch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params=case.value("isolation_batch.continuation_5"),
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_5_terminal_timeout_s"),
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params=case.value("isolation_batch.continuation_6"),
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_6_terminal_timeout_s"),
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params=case.value("isolation_batch.continuation_7"),
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_7_terminal_timeout_s"),
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params=case.value("isolation_batch.continuation_8"),
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_8_terminal_timeout_s"),
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params=case.value("isolation_batch.continuation_9"),
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=case.value("isolation_batch.continuation_9_terminal_timeout_s"),
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "fidelity",
        "kv_fidelity_check",
        params=case.params(
            "isolation_batch.fidelity",
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
                "holder": output("a_holder", "engine"),
            },
        ),
    )
    case.step(
        "final_cache", "kv_snapshot", params={"targets": [output("b_holder", "engine")]}
    )
    case.step(
        "final_b_isolation",
        "kv_membership_check",
        params=case.params(
            "isolation_batch.final_b_isolation",
            {
                "snapshot": output("final_cache", "snapshot"),
                "engine": output("b_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def evict_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("evict_nonbatch.setup_timeout_s"))
    case.step(
        "prime",
        "request",
        params=case.value("evict_nonbatch.prime"),
    )
    case.step(
        "prime_terminal",
        "wait",
        timeout_s=case.value("evict_nonbatch.prime_terminal_timeout_s"),
        params={"requests": output("prime", "requests")},
    )
    case.step(
        "holder",
        "kv_landing",
        params=case.params(
            "evict_nonbatch.holder", {"requests": output("prime", "requests")}
        ),
    )
    case.step(
        "prime_quiet",
        "kv_snapshot",
        timeout_s=case.value("evict_nonbatch.prime_quiet_timeout_s"),
        params=case.value("evict_nonbatch.prime_quiet"),
    )
    case.step(
        "positive",
        "request",
        params=case.value("evict_nonbatch.positive"),
    )
    case.step(
        "positive_terminal",
        "wait",
        timeout_s=case.value("evict_nonbatch.positive_terminal_timeout_s"),
        params={"requests": output("positive", "requests")},
    )
    case.step(
        "positive_holder",
        "kv_landing",
        params=case.params(
            "evict_nonbatch.positive_holder",
            {"requests": output("positive", "requests")},
        ),
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
        params=case.params(
            "evict_nonbatch.evict", {"engine": output("holder", "engine")}
        ),
    )
    case.step(
        "eviction_quiet",
        "kv_snapshot",
        timeout_s=case.value("evict_nonbatch.eviction_quiet_timeout_s"),
        params=case.value("evict_nonbatch.eviction_quiet"),
    )
    case.step(
        "eviction_membership",
        "kv_membership_check",
        params=case.params(
            "evict_nonbatch.eviction_membership",
            {
                "snapshot": output("eviction_quiet", "snapshot"),
                "engine": output("holder", "engine"),
            },
        ),
    )
    case.step(
        "wave",
        "request",
        params=case.value("evict_nonbatch.wave"),
    )
    case.step(
        "wave_terminal",
        "wait",
        timeout_s=case.value("evict_nonbatch.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "spread",
        "kv_spread_check",
        params=case.params(
            "evict_nonbatch.spread", {"requests": [output("wave", "requests")]}
        ),
    )
    case.step("cleanup", "teardown")


def isolation_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("isolation_nonbatch.setup_timeout_s")
    )
    case.step(
        "fleet", "balance_snapshot", params=case.value("isolation_nonbatch.fleet")
    )
    case.step(
        "perf_sync", "balance_pause", params=case.value("isolation_nonbatch.perf_sync")
    )
    case.step(
        "seed_a",
        "request",
        params=case.value("isolation_nonbatch.seed_a"),
    )
    case.step(
        "a_pending",
        "balance_pending",
        timeout_s=case.value("isolation_nonbatch.a_pending_timeout_s"),
        params={
            "requests": output("seed_a", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "a_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.a_terminal_timeout_s"),
        params={"requests": output("seed_a", "requests")},
    )
    case.step(
        "a_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.a_holder", {"requests": output("seed_a", "requests")}
        ),
    )
    case.step(
        "seed_b",
        "request",
        params=case.value("isolation_nonbatch.seed_b"),
    )
    case.step(
        "b_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.b_terminal_timeout_s"),
        params={"requests": output("seed_b", "requests")},
    )
    case.step(
        "b_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.b_holder", {"requests": output("seed_b", "requests")}
        ),
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
        "b_perf_sync",
        "balance_pause",
        params=case.value("isolation_nonbatch.b_perf_sync"),
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step("filler_0", "request", params=case.value("isolation_nonbatch.filler"))
    case.step(
        "filler_0_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.filler_holder",
            {"requests": output("filler_0", "requests")},
        ),
    )
    case.step(
        "filler_0_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_0_holder", "engine"),
        },
    )
    case.step(
        "admit_0",
        "request",
        params=case.value("isolation_nonbatch.admit_0"),
    )
    case.step(
        "admit_0_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.admit_0_terminal_timeout_s"),
        params={"requests": output("admit_0", "requests")},
    )
    case.step(
        "admit_0_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.admit_0_holder",
            {"requests": output("admit_0", "requests")},
        ),
    )
    case.step(
        "admit_0_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_0_holder", "engine"),
        },
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step(
        "filler_0_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.filler_terminal_timeout_s"),
        params={"requests": output("filler_0", "requests")},
    )
    case.step("filler_1", "request", params=case.value("isolation_nonbatch.filler"))
    case.step(
        "filler_1_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.filler_holder",
            {"requests": output("filler_1", "requests")},
        ),
    )
    case.step(
        "filler_1_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_1_holder", "engine"),
        },
    )
    case.step(
        "admit_1",
        "request",
        params=case.value("isolation_nonbatch.admit_1"),
    )
    case.step(
        "admit_1_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.admit_1_terminal_timeout_s"),
        params={"requests": output("admit_1", "requests")},
    )
    case.step(
        "admit_1_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.admit_1_holder",
            {"requests": output("admit_1", "requests")},
        ),
    )
    case.step(
        "admit_1_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_1_holder", "engine"),
        },
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step(
        "filler_1_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.filler_terminal_timeout_s"),
        params={"requests": output("filler_1", "requests")},
    )
    case.step("filler_2", "request", params=case.value("isolation_nonbatch.filler"))
    case.step(
        "filler_2_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.filler_holder",
            {"requests": output("filler_2", "requests")},
        ),
    )
    case.step(
        "filler_2_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_2_holder", "engine"),
        },
    )
    case.step(
        "admit_2",
        "request",
        params=case.value("isolation_nonbatch.admit_2"),
    )
    case.step(
        "admit_2_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.admit_2_terminal_timeout_s"),
        params={"requests": output("admit_2", "requests")},
    )
    case.step(
        "admit_2_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.admit_2_holder",
            {"requests": output("admit_2", "requests")},
        ),
    )
    case.step(
        "admit_2_on_a",
        "kv_same",
        params={
            "first": output("a_holder", "engine"),
            "second": output("admit_2_holder", "engine"),
        },
    )
    # Seed-family traffic makes B's occupancy visible through real admission.
    case.step(
        "filler_2_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.filler_terminal_timeout_s"),
        params={"requests": output("filler_2", "requests")},
    )
    case.step("filler_3", "request", params=case.value("isolation_nonbatch.filler"))
    case.step(
        "filler_3_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.filler_holder",
            {"requests": output("filler_3", "requests")},
        ),
    )
    case.step(
        "filler_3_on_b",
        "kv_same",
        params={
            "first": output("b_holder", "engine"),
            "second": output("filler_3_holder", "engine"),
        },
    )
    case.step(
        "admit_3",
        "request",
        params=case.value("isolation_nonbatch.admit_3"),
    )
    case.step(
        "admit_3_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.admit_3_terminal_timeout_s"),
        params={"requests": output("admit_3", "requests")},
    )
    case.step(
        "admit_3_holder",
        "kv_landing",
        params=case.params(
            "isolation_nonbatch.admit_3_holder",
            {"requests": output("admit_3", "requests")},
        ),
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
        "filler_3_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.filler_terminal_timeout_s"),
        params={"requests": output("filler_3", "requests")},
    )
    case.step(
        "admission_quiet",
        "kv_snapshot",
        timeout_s=case.value("isolation_nonbatch.admission_quiet_timeout_s"),
        params=case.value("isolation_nonbatch.admission_quiet"),
    )
    case.step(
        "a_membership",
        "kv_membership_check",
        params=case.params(
            "isolation_nonbatch.a_membership",
            {
                "snapshot": output("admission_quiet", "snapshot"),
                "engine": output("a_holder", "engine"),
            },
        ),
    )
    case.step(
        "b_isolation",
        "kv_membership_check",
        params=case.params(
            "isolation_nonbatch.b_isolation",
            {
                "snapshot": output("admission_quiet", "snapshot"),
                "engine": output("b_holder", "engine"),
            },
        ),
    )
    case.step(
        "continuation_0",
        "request",
        params=case.value("isolation_nonbatch.continuation_0"),
    )
    case.step(
        "continuation_0_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_0_terminal_timeout_s"),
        params={"requests": output("continuation_0", "requests")},
    )
    case.step(
        "continuation_1",
        "request",
        params=case.value("isolation_nonbatch.continuation_1"),
    )
    case.step(
        "continuation_1_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_1_terminal_timeout_s"),
        params={"requests": output("continuation_1", "requests")},
    )
    case.step(
        "continuation_2",
        "request",
        params=case.value("isolation_nonbatch.continuation_2"),
    )
    case.step(
        "continuation_2_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_2_terminal_timeout_s"),
        params={"requests": output("continuation_2", "requests")},
    )
    case.step(
        "continuation_3",
        "request",
        params=case.value("isolation_nonbatch.continuation_3"),
    )
    case.step(
        "continuation_3_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_3_terminal_timeout_s"),
        params={"requests": output("continuation_3", "requests")},
    )
    case.step(
        "continuation_4",
        "request",
        params=case.value("isolation_nonbatch.continuation_4"),
    )
    case.step(
        "continuation_4_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_4_terminal_timeout_s"),
        params={"requests": output("continuation_4", "requests")},
    )
    case.step(
        "continuation_5",
        "request",
        params=case.value("isolation_nonbatch.continuation_5"),
    )
    case.step(
        "continuation_5_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_5_terminal_timeout_s"),
        params={"requests": output("continuation_5", "requests")},
    )
    case.step(
        "continuation_6",
        "request",
        params=case.value("isolation_nonbatch.continuation_6"),
    )
    case.step(
        "continuation_6_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_6_terminal_timeout_s"),
        params={"requests": output("continuation_6", "requests")},
    )
    case.step(
        "continuation_7",
        "request",
        params=case.value("isolation_nonbatch.continuation_7"),
    )
    case.step(
        "continuation_7_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_7_terminal_timeout_s"),
        params={"requests": output("continuation_7", "requests")},
    )
    case.step(
        "continuation_8",
        "request",
        params=case.value("isolation_nonbatch.continuation_8"),
    )
    case.step(
        "continuation_8_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_8_terminal_timeout_s"),
        params={"requests": output("continuation_8", "requests")},
    )
    case.step(
        "continuation_9",
        "request",
        params=case.value("isolation_nonbatch.continuation_9"),
    )
    case.step(
        "continuation_9_terminal",
        "wait",
        timeout_s=case.value("isolation_nonbatch.continuation_9_terminal_timeout_s"),
        params={"requests": output("continuation_9", "requests")},
    )
    case.step(
        "fidelity",
        "kv_fidelity_check",
        params=case.params(
            "isolation_nonbatch.fidelity",
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
                "holder": output("a_holder", "engine"),
            },
        ),
    )
    case.step(
        "final_cache", "kv_snapshot", params={"targets": [output("b_holder", "engine")]}
    )
    case.step(
        "final_b_isolation",
        "kv_membership_check",
        params=case.params(
            "isolation_nonbatch.final_b_isolation",
            {
                "snapshot": output("final_cache", "snapshot"),
                "engine": output("b_holder", "engine"),
            },
        ),
    )
    case.step("cleanup", "teardown")
