"""Explicit preemption cohorts and legacy design-final predicates."""

from ..case_config import output

METADATA = {
    "id": "priority_preemption",
    "description": "Explicit preemption cohorts and legacy design-final predicates.",
    "category": "priority",
}

PROFILES = ["single-nonbatch", "single-batch", "batch-window"]


def same_priority_zero_eviction(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params={
            "actual": output("placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer7", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "incoming", "priority": 50, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "same_priority",
        "preemption_same_priority",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step("master_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def prefill_queued(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params={
            "actual": output("r1_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 40, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 40, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer7", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "incoming", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_queued_first",
        params={
            "placeholder": output("r1_placeholder", "requests"),
            "wave": output("r1_wave", "requests"),
            "round": 1,
        },
    )
    case.step("r1_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 70,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params={
            "actual": output("r2_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer7", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "incoming", "priority": 90, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_queued_second",
        params={
            "placeholder": output("r2_placeholder", "requests"),
            "wave": output("r2_wave", "requests"),
            "round": 2,
        },
    )
    case.step("r2_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def timeout_attribution(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 12000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 90,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params={
            "actual": output("r1_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer7", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer8", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer9", "priority": 90, "input_len": 2048, "output_len": 2},
                {"tag": "peer10", "priority": 90, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=385,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_expiry",
        params={
            "placeholder": output("r1_placeholder", "requests"),
            "wave": output("r1_wave", "requests"),
            "round": 1,
        },
    )
    case.step("r1_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "round2_slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 10000},
        },
    )
    case.step("round2_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 70,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params={
            "actual": output("r2_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer7", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer8", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_expiry",
        params={
            "placeholder": output("r2_placeholder", "requests"),
            "wave": output("r2_wave", "requests"),
            "round": 2,
        },
    )
    case.step("r2_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def disabled_zero_eviction(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params={
            "actual": output("r1_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer7", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "incoming", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_disabled",
        params={
            "placeholder": output("r1_placeholder", "requests"),
            "wave": output("r1_wave", "requests"),
            "round": 1,
        },
    )
    case.step("r1_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 70,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params={
            "actual": output("r2_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer7", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "incoming", "priority": 90, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_disabled",
        params={
            "placeholder": output("r2_placeholder", "requests"),
            "wave": output("r2_wave", "requests"),
            "round": 2,
        },
    )
    case.step("r2_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def comparator_frozen_weak(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params={
            "actual": output("r1_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=175,
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_comparator_half",
        params={
            "placeholder": output("r1_placeholder", "requests"),
            "wave": output("r1_wave", "requests"),
            "ordering": "priority",
        },
    )
    case.step("r1_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "fifo_environment",
        "environment_reconfigure",
        timeout_s=300,
        params={
            "config_overrides": {
                "ordering": "fifo",
                "decision": "single",
                "dispatcher": "non_batch",
                "queue_timeout_ms": {"omit": True},
                "max_inflight_requests_per_worker": 1,
                "max_waiting_requests_per_prefill_worker": 8,
            }
        },
    )
    case.step("fifo_fleet", "priority_fleet")
    case.step(
        "fifo_slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fifo_fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("fifo_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params={
            "actual": output("r2_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fifo_fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=175,
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_comparator_half",
        params={
            "placeholder": output("r2_placeholder", "requests"),
            "wave": output("r2_wave", "requests"),
            "ordering": "fifo",
        },
    )
    case.step("r2_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "comparator_verdict",
        "preemption_comparator_pair",
        params={
            "priority_half": output("r1_same_priority", "passed"),
            "fifo_half": output("r2_same_priority", "passed"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fifo_fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def config_strict_reject(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "removed_auto_tpm",
        "environment_startup_probe",
        timeout_s=300,
        params={
            "config_overrides": {
                "ordering": "priority",
                "decision": "single",
                "dispatcher": "non_batch",
                "queue_timeout_ms": {"omit": True},
                "max_inflight_requests_per_worker": 1,
                "max_waiting_requests_per_prefill_worker": 8,
            },
            "mutation": "removed_auto_tpm",
        },
    )
    case.step(
        "fifo_default_priority",
        "environment_startup_probe",
        timeout_s=300,
        params={
            "config_overrides": {
                "ordering": "fifo",
                "decision": "single",
                "dispatcher": "non_batch",
                "queue_timeout_ms": {"omit": True},
                "max_inflight_requests_per_worker": 1,
                "max_waiting_requests_per_prefill_worker": 8,
            },
            "mutation": "fifo_default_priority",
        },
    )
    case.step(
        "owned_without_cancellation",
        "environment_startup_probe",
        timeout_s=300,
        params={
            "config_overrides": {
                "ordering": "priority",
                "decision": "single",
                "dispatcher": "non_batch",
                "queue_timeout_ms": {"omit": True},
                "max_inflight_requests_per_worker": 1,
                "max_waiting_requests_per_prefill_worker": 8,
                "preemption": {
                    "allowed_victim_stages": ["DECODE_RESERVED", "DECODE_ENGINE_OWNED"],
                    "engine_cancellation": {
                        "ack_timeout_ms": 50,
                        "completion_timeout_ms": 1000,
                    },
                },
            },
            "mutation": "owned_without_cancellation",
        },
    )
    case.step(
        "strict_rejection",
        "preemption_config_reject",
        params={
            "rejected": [
                output("removed_auto_tpm", "rejected"),
                output("fifo_default_priority", "rejected"),
                output("owned_without_cancellation", "rejected"),
            ],
            "environment_absent": output(
                "owned_without_cancellation", "environment_absent"
            ),
        },
    )
    case.step("teardown", "teardown")


def decode_engine_owned(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "preemption_decode_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill0"), output("fleet", "prefill1")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_occupants",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {
                    "tag": "occupant0",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant1",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant2",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant3",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
            ],
        },
    )
    case.step(
        "r1_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 6291456,
        },
    )
    case.step("r1_pressure_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_guard",
        "preemption_decode_guard",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ]
        },
    )
    case.step(
        "r1_incoming",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {"tag": "incoming", "priority": 70, "input_len": 2048, "output_len": 2}
            ],
        },
    )
    case.step(
        "r1_incoming_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_occupants_drain",
        "preemption_wait",
        timeout_s=140,
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_incoming_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_verdict",
        "preemption_decode_half",
        params={
            "occupants": output("r1_occupants", "requests"),
            "incoming": output("r1_incoming", "requests"),
            "phase": "reserved",
        },
    )
    case.step("r1_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "release_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 0,
        },
    )
    case.step("release_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r2_occupants",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {
                    "tag": "occupant0",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant1",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant2",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant3",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
            ],
        },
    )
    case.step(
        "r2_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_running",
        "preemption_decode_running",
        timeout_s=80,
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 6291456,
        },
    )
    case.step("r2_pressure_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r2_guard",
        "preemption_decode_guard",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ]
        },
    )
    case.step(
        "r2_incoming",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {"tag": "incoming", "priority": 70, "input_len": 2048, "output_len": 2}
            ],
        },
    )
    case.step(
        "r2_incoming_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step(
        "r2_occupants_drain",
        "preemption_wait",
        timeout_s=140,
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_incoming_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step(
        "r2_verdict",
        "preemption_decode_half",
        params={
            "occupants": output("r2_occupants", "requests"),
            "incoming": output("r2_incoming", "requests"),
            "phase": "owned",
        },
    )
    case.step("r2_master_clean", "balance_clean", timeout_s=30)
    case.step(
        "decode_verdict",
        "preemption_decode_final",
        params={
            "reserved_zero": output("r1_verdict", "zero_eviction"),
            "owned_zero": output("r2_verdict", "zero_eviction"),
            "incoming_rejected": output("r2_verdict", "incoming_rejected"),
            "survivors_ok": output("r2_verdict", "survivors_ok"),
        },
    )
    case.step(
        "clear_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 0,
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill0"), output("fleet", "prefill1")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown")


def error_code_family(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("s1_fleet", "balance_snapshot", params={"role": "prefill"})
    case.step(
        "s1_slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("s1_fleet", "first"), output("s1_fleet", "second")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("s1_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "s1_placeholder",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {
                    "tag": "placeholder0",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 2,
                },
                {
                    "tag": "placeholder1",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 2,
                },
            ],
        },
    )
    case.step(
        "s1_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("s1_placeholder", "requests")},
    )
    case.step(
        "s1_admitted",
        "check",
        params={
            "actual": output("s1_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "s1_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("s1_fleet", "first")},
    )
    case.step(
        "s1_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "wave0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave1", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "s1_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("s1_wave", "requests")},
    )
    case.step(
        "s1_placeholder_drain",
        "preemption_wait",
        timeout_s=70,
        params={"requests": output("s1_placeholder", "requests")},
    )
    case.step(
        "s1_wave_drain",
        "preemption_wait",
        timeout_s=70,
        params={"requests": output("s1_wave", "requests")},
    )
    case.step(
        "s1_segment",
        "preemption_error_segment",
        params={
            "placeholder": output("s1_placeholder", "requests"),
            "wave": output("s1_wave", "requests"),
            "segment": "outstanding",
        },
    )
    case.step(
        "s1_restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("s1_fleet", "first"), output("s1_fleet", "second")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "generate_payload": "match_schedule",
        },
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=60,
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step("s1_clean", "balance_clean", timeout_s=30)
    case.step(
        "s2_environment",
        "environment_reconfigure",
        timeout_s=300,
        params={
            "config_overrides": {
                "ordering": "priority",
                "decision": "single",
                "dispatcher": "non_batch",
                "queue_timeout_ms": 60000,
                "max_inflight_requests_per_worker": 1,
                "max_waiting_requests_per_prefill_worker": 8,
                "preemption": {"allowed_victim_stages": ["PREFILL_QUEUED"]},
            },
            "n_prefill": 1,
            "n_decode": 4,
        },
    )
    case.step("s2_fleet", "priority_fleet")
    case.step(
        "s2_slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("s2_fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("s2_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "s2_placeholder",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder0",
                    "priority": 70,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "s2_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("s2_placeholder", "requests")},
    )
    case.step(
        "s2_admitted",
        "check",
        params={
            "actual": output("s2_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "s2_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("s2_fleet", "prefill")},
    )
    case.step(
        "s2_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "wave0", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave1", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave2", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave3", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave4", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave5", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave6", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave7", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "wave8", "priority": 90, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "s2_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("s2_wave", "requests")},
    )
    case.step(
        "s2_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("s2_placeholder", "requests")},
    )
    case.step(
        "s2_wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("s2_wave", "requests")},
    )
    case.step(
        "s2_segment",
        "preemption_error_segment",
        params={
            "placeholder": output("s2_placeholder", "requests"),
            "wave": output("s2_wave", "requests"),
            "segment": "park",
        },
    )
    case.step("s2_clean", "balance_clean", timeout_s=30)
    case.step(
        "s2_restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("s2_fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "s3_environment",
        "environment_reconfigure",
        timeout_s=300,
        params={
            "config_overrides": {
                "ordering": "priority",
                "decision": "single",
                "dispatcher": "non_batch",
                "queue_timeout_ms": 7000,
                "max_inflight_requests_per_worker": 1,
                "max_waiting_requests_per_prefill_worker": 8,
                "preemption": {"allowed_victim_stages": ["PREFILL_QUEUED"]},
            },
            "n_prefill": 1,
            "n_decode": 4,
        },
    )
    case.step("s3_fleet", "priority_fleet")
    case.step(
        "s3_slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("s3_fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 10000},
        },
    )
    case.step("s3_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "s3_placeholder",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder0",
                    "priority": 70,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "s3_placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("s3_placeholder", "requests")},
    )
    case.step(
        "s3_admitted",
        "check",
        params={
            "actual": output("s3_placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "s3_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("s3_fleet", "prefill")},
    )
    case.step(
        "s3_wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "wave0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave2", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave3", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave4", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave5", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave6", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave7", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "wave8", "priority": 90, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "s3_wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("s3_wave", "requests")},
    )
    case.step(
        "s3_placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("s3_placeholder", "requests")},
    )
    case.step(
        "s3_wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("s3_wave", "requests")},
    )
    case.step(
        "s3_segment",
        "preemption_error_segment",
        params={
            "placeholder": output("s3_placeholder", "requests"),
            "wave": output("s3_wave", "requests"),
            "segment": "expiry",
        },
    )
    case.step("s3_clean", "balance_clean", timeout_s=30)
    case.step(
        "s3_restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("s3_fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "error_family_verdict",
        "preemption_error_final",
        params={
            "segments": [
                output("s1_segment", "passed"),
                output("s2_segment", "passed"),
                output("s3_segment", "passed"),
            ],
            "recovery": output("recovery_result", "passed"),
        },
    )
    case.step("teardown", "teardown")


def decode_reservation_priority(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "preemption_decode_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill0"), output("fleet", "prefill1")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_occupants",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {
                    "tag": "occupant0",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant1",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant2",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant3",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
            ],
        },
    )
    case.step(
        "r1_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_running",
        "preemption_decode_running",
        timeout_s=80,
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_baseline",
        "preemption_reservation_metric",
        params={"labels": {"victim_priority": "30", "incoming_priority": "70"}},
    )
    case.step(
        "r1_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 6291456,
        },
    )
    case.step("r1_pressure_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r1_incoming",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {"tag": "incoming", "priority": 70, "input_len": 2048, "output_len": 2}
            ],
        },
    )
    case.step(
        "r1_incoming_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_occupants_drain",
        "preemption_wait",
        timeout_s=140,
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_incoming_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_after",
        "preemption_reservation_metric",
        params={"labels": {"victim_priority": "30", "incoming_priority": "70"}},
    )
    case.step(
        "r1_verdict",
        "preemption_reservation_half",
        params={
            "occupants": output("r1_occupants", "requests"),
            "incoming": output("r1_incoming", "requests"),
            "wave": "lower",
            "baseline": output("r1_baseline", "snapshot"),
            "after": output("r1_after", "snapshot"),
        },
    )
    case.step("r1_clean", "balance_clean", timeout_s=30)
    case.step(
        "r2_release",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 0,
        },
    )
    case.step("r2_release_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r2_occupants",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {
                    "tag": "occupant0",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant1",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant2",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant3",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 500,
                },
            ],
        },
    )
    case.step(
        "r2_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_running",
        "preemption_decode_running",
        timeout_s=80,
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step("r2_baseline", "preemption_reservation_metric", params={"labels": {}})
    case.step(
        "r2_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 6291456,
        },
    )
    case.step("r2_pressure_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r2_incoming",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {"tag": "incoming", "priority": 50, "input_len": 2048, "output_len": 2}
            ],
        },
    )
    case.step(
        "r2_incoming_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step(
        "r2_occupants_drain",
        "preemption_wait",
        timeout_s=140,
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_incoming_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step("r2_after", "preemption_reservation_metric", params={"labels": {}})
    case.step(
        "r2_verdict",
        "preemption_reservation_half",
        params={
            "occupants": output("r2_occupants", "requests"),
            "incoming": output("r2_incoming", "requests"),
            "wave": "same",
            "baseline": output("r2_baseline", "snapshot"),
            "after": output("r2_after", "snapshot"),
        },
    )
    case.step("r2_clean", "balance_clean", timeout_s=30)
    case.step(
        "r3_release",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 0,
        },
    )
    case.step("r3_release_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r3_occupants",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {
                    "tag": "occupant0",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant1",
                    "priority": 30,
                    "input_len": 2048,
                    "output_len": 500,
                },
                {
                    "tag": "occupant2",
                    "priority": 30,
                    "input_len": 16384,
                    "output_len": 500,
                },
                {
                    "tag": "occupant3",
                    "priority": 30,
                    "input_len": 16384,
                    "output_len": 500,
                },
            ],
        },
    )
    case.step(
        "r3_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r3_occupants", "requests")},
    )
    case.step(
        "r3_running",
        "preemption_decode_running",
        timeout_s=80,
        params={"requests": output("r3_occupants", "requests")},
    )
    case.step(
        "r3_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 6291456,
        },
    )
    case.step("r3_pressure_sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "r3_incoming",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {"tag": "incoming", "priority": 70, "input_len": 8192, "output_len": 2}
            ],
        },
    )
    case.step(
        "r3_incoming_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("r3_incoming", "requests")},
    )
    case.step(
        "r3_occupants_drain",
        "preemption_wait",
        timeout_s=140,
        params={"requests": output("r3_occupants", "requests")},
    )
    case.step(
        "r3_incoming_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("r3_incoming", "requests")},
    )
    case.step(
        "r3_verdict",
        "preemption_reservation_half",
        params={
            "occupants": output("r3_occupants", "requests"),
            "incoming": output("r3_incoming", "requests"),
            "wave": "kvbucket",
        },
    )
    case.step("r3_clean", "balance_clean", timeout_s=30)
    case.step(
        "reservation_verdict",
        "preemption_reservation_final",
        params={
            "waves": [
                output("r1_verdict", "passed"),
                output("r2_verdict", "passed"),
                output("r3_verdict", "passed"),
            ]
        },
    )
    case.step(
        "clear_pressure",
        "preemption_decode_pressure",
        params={
            "targets": [
                output("fleet", "decode0"),
                output("fleet", "decode1"),
                output("fleet", "decode2"),
                output("fleet", "decode3"),
            ],
            "tokens": 0,
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill0"), output("fleet", "prefill1")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown")


def observability_integrity(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
        },
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params={
            "actual": output("placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "duplicate",
        "preemption_observability_duplicate",
        timeout_s=35,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "30a", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "30b", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "50a", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "50b", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "70a", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "70b", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "30c", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "30d", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "90", "priority": 90, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_wait",
        timeout_s=315,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "observe",
        "preemption_observability",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step("master_clean", "balance_clean", timeout_s=30)
    case.step(
        "observability_verdict",
        "preemption_observability_final",
        params={
            "client": output("observe", "client"),
            "planes": output("observe", "planes"),
            "duplicate": output("duplicate", "rejected"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def prefill_queued_live_single(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params={
            "actual": output("placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "victim_a", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "victim_b", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "incoming", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=50,
        params={"requests": output("placeholder", "requests"), "tags": ["placeholder"]},
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=100,
        params={
            "requests": output("wave", "requests"),
            "tags": ["victim_a", "incoming"],
        },
    )
    case.step(
        "live_verdict",
        "preemption_live_prefill",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step("master_clean", "balance_clean", timeout_s=30)
    case.step("engine_clean", "preemption_live_engine_clean", timeout_s=35)
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "generate_payload": "match_schedule",
        },
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=60,
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "live_final",
        "preemption_live_final",
        params={
            "pr10": output("live_verdict", "pr10"),
            "pr5": output("live_verdict", "pr5"),
            "pr6": output("live_verdict", "pr6"),
            "engine_clean": output("engine_clean", "passed"),
            "recovery": output("recovery_result", "passed"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def prefill_queued_live_window(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 50,
                    "input_len": 2048,
                    "output_len": 2,
                }
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params={
            "actual": output("placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "victim_a", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "victim_b", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "incoming", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=50,
        params={"requests": output("placeholder", "requests"), "tags": ["placeholder"]},
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=100,
        params={
            "requests": output("wave", "requests"),
            "tags": ["victim_a", "incoming"],
        },
    )
    case.step(
        "live_verdict",
        "preemption_live_prefill",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step("master_clean", "balance_clean", timeout_s=30)
    case.step("engine_clean", "preemption_live_engine_clean", timeout_s=35)
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "generate_payload": "match_schedule",
        },
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=60,
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "live_final",
        "preemption_live_final",
        params={
            "pr10": output("live_verdict", "pr10"),
            "pr5": output("live_verdict", "pr5"),
            "pr6": output("live_verdict", "pr6"),
            "engine_clean": output("engine_clean", "passed"),
            "recovery": output("recovery_result", "passed"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def decode_reserved_live_single(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 90,
                    "input_len": 512,
                    "output_len": 2,
                }
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params={
            "actual": output("placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    # Release the placeholder before waiting for incoming admission; otherwise
    # its deferred Fetch is behind the very Schedule waiting for its capacity.
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=50,
        params={"requests": output("placeholder", "requests"), "tags": ["placeholder"]},
    )
    case.step("placeholder_master_clean", "balance_clean", timeout_s=30)
    case.step("placeholder_engine_clean", "preemption_live_engine_clean", timeout_s=35)
    case.step(
        "placeholder_released",
        "check",
        params={
            "actual": output("placeholder_engine_clean", "passed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "victim", "priority": 30, "input_len": 512, "output_len": 2},
                {"tag": "incoming", "priority": 70, "input_len": 3500, "output_len": 2},
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=50,
        params={"requests": output("wave", "requests"), "tags": ["incoming"]},
    )
    case.step(
        "live_verdict",
        "preemption_live_reserved",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step("master_clean", "balance_clean", timeout_s=30)
    case.step("engine_clean", "preemption_live_engine_clean", timeout_s=35)
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "generate_payload": "match_schedule",
        },
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=60,
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "live_final",
        "preemption_live_final",
        params={
            "pr10": output("live_verdict", "pr10"),
            "pr5": output("live_verdict", "pr5"),
            "pr6": output("live_verdict", "pr6"),
            "engine_clean": output("engine_clean", "passed"),
            "recovery": output("recovery_result", "passed"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def decode_reserved_live_window(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {
                    "tag": "placeholder",
                    "priority": 90,
                    "input_len": 512,
                    "output_len": 2,
                }
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params={
            "actual": output("placeholder_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=6,
        params={"target": output("fleet", "prefill")},
    )
    # Release the placeholder before waiting for incoming admission; otherwise
    # its deferred Fetch is behind the very Schedule waiting for its capacity.
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=50,
        params={"requests": output("placeholder", "requests"), "tags": ["placeholder"]},
    )
    case.step("placeholder_master_clean", "balance_clean", timeout_s=30)
    case.step("placeholder_engine_clean", "preemption_live_engine_clean", timeout_s=35)
    case.step(
        "placeholder_released",
        "check",
        params={
            "actual": output("placeholder_engine_clean", "passed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "victim", "priority": 30, "input_len": 512, "output_len": 2},
                {"tag": "incoming", "priority": 70, "input_len": 3500, "output_len": 2},
            ],
            "defer_batch": True,
        },
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=50,
        params={"requests": output("wave", "requests"), "tags": ["incoming"]},
    )
    case.step(
        "live_verdict",
        "preemption_live_reserved",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step("master_clean", "balance_clean", timeout_s=30)
    case.step("engine_clean", "preemption_live_engine_clean", timeout_s=35)
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "generate_payload": "match_schedule",
        },
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=60,
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "live_final",
        "preemption_live_final",
        params={
            "pr10": output("live_verdict", "pr10"),
            "pr5": output("live_verdict", "pr5"),
            "pr6": output("live_verdict", "pr6"),
            "engine_clean": output("engine_clean", "passed"),
            "recovery": output("recovery_result", "passed"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def cancel_not_found(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "victim",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {"tag": "victim", "priority": 30, "input_len": 512, "output_len": 200}
            ],
        },
    )
    case.step(
        "victim_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_admitted",
        "check",
        params={
            "actual": output("victim_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "victim_running",
        "preemption_nf_state",
        timeout_s=10,
        params={"requests": output("victim", "requests"), "state": "running"},
    )
    case.step("before_freeze_pause", "balance_pause", params={"seconds": 0.6})
    case.step("baseline_cancel", "preemption_cancel_census")
    case.step(
        "freeze_status",
        "status_control",
        params={
            "role": "decode",
            "selection": "first",
            "fault": "status_no_respond",
            "enabled": True,
        },
    )
    case.step(
        "victim_engine_finished",
        "preemption_nf_state",
        timeout_s=3,
        params={"requests": output("victim", "requests"), "state": "finished"},
    )
    case.step(
        "incoming",
        "priority_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "requests": [
                {"tag": "incoming", "priority": 70, "input_len": 512, "output_len": 2}
            ],
        },
    )
    case.step(
        "incoming_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("incoming", "requests")},
    )
    case.step(
        "victim_drain",
        "preemption_wait",
        timeout_s=35,
        params={"requests": output("victim", "requests")},
    )
    case.step("after_cancel", "preemption_cancel_census")
    case.step(
        "nf_verdict",
        "preemption_nf_verdict",
        params={
            "victim": output("victim", "requests"),
            "incoming": output("incoming", "requests"),
            "before": output("baseline_cancel", "snapshot"),
            "after": output("after_cancel", "snapshot"),
        },
    )
    case.step(
        "clear_status",
        "status_control",
        params={
            "role": "decode",
            "selection": "first",
            "fault": "status_no_respond",
            "enabled": False,
        },
    )
    case.step("master_clean", "balance_clean", timeout_s=30)
    case.step("engine_clean", "preemption_nf_engine_clean", timeout_s=25)
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "generate_payload": "match_schedule",
        },
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=60,
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "nf_final",
        "preemption_nf_final",
        params={
            "settled": output("nf_verdict", "settled"),
            "cancel_seen": output("nf_verdict", "cancel_seen"),
            "engine_clean": output("engine_clean", "passed"),
            "recovery": output("recovery_result", "passed"),
        },
    )
    case.step("teardown", "teardown")


def cancel_tombstoned(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "victim",
        "preemption_live_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "defer_batch": True,
            "requests": [
                {"tag": "victim", "priority": 30, "input_len": 512, "output_len": 5000}
            ],
        },
    )
    case.step(
        "victim_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_admitted",
        "check",
        params={
            "actual": output("victim_settled", "admitted"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "first_output",
        "preemption_ts_first_output",
        timeout_s=15,
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "crash_fault",
        "engine_inject",
        params={"type": "crash_after", "targets": ["prefill-0"], "options": {"n": 1}},
    )
    case.step("restore_guard", "preemption_ts_restore_guard")
    case.step("crash_trigger", "preemption_ts_trigger", timeout_s=10)
    case.step(
        "prefill_dropped",
        "preemption_ts_health",
        timeout_s=45,
        params={"state": "dropped"},
    )
    case.step(
        "prefill_restart",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "prefill_restored",
        "preemption_ts_health",
        timeout_s=45,
        params={"state": "restored"},
    )
    case.step("reconnect", "balance_pause", params={"seconds": 3})
    case.step(
        "victim_cut",
        "preemption_ts_cut",
        timeout_s=10,
        params={"requests": output("victim", "requests")},
    )
    case.step("baseline_cancel", "preemption_cancel_census")
    case.step(
        "incoming",
        "preemption_live_start",
        timeout_s=5,
        params={
            "gap_s": 0,
            "defer_batch": True,
            "requests": [
                {"tag": "incoming", "priority": 70, "input_len": 512, "output_len": 2}
            ],
        },
    )
    case.step(
        "incoming_settled",
        "preemption_settled",
        timeout_s=90,
        params={"requests": output("incoming", "requests")},
    )
    case.step(
        "incoming_drain",
        "preemption_live_drain",
        timeout_s=50,
        params={"requests": output("incoming", "requests"), "tags": ["incoming"]},
    )
    case.step(
        "incoming_completed",
        "preemption_ts_incoming",
        params={"requests": output("incoming", "requests")},
    )
    case.step(
        "cancel_arrival",
        "preemption_ts_cancel",
        timeout_s=30,
        params={"before": output("baseline_cancel", "snapshot")},
    )
    case.step(
        "fence_probe",
        "preemption_ts_fence",
        timeout_s=15,
        params={"requests": output("victim", "requests")},
    )
    case.step("engine_clean", "preemption_ts_engine_clean", timeout_s=80)
    case.step("residue", "preemption_ts_residue", timeout_s=45)
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "generate_payload": "match_schedule",
        },
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=60,
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "ts_final",
        "preemption_ts_final",
        params={
            "cut": output("victim_cut", "cut"),
            "incoming": output("incoming_completed", "completed"),
            "delta_seen": output("cancel_arrival", "delta_seen"),
            "reached": output("cancel_arrival", "reached"),
            "fence": output("fence_probe", "passed"),
            "engine_clean": output("engine_clean", "passed"),
            "residue": output("residue", "passed"),
            "recovery": output("recovery_result", "passed"),
        },
    )
    case.step("teardown", "teardown")


VARIANTS = {
    "same_priority_zero_eviction": {
        "build": same_priority_zero_eviction,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "prefill_queued": {
        "build": prefill_queued,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "timeout_attribution": {
        "build": timeout_attribution,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "disabled_zero_eviction": {
        "build": disabled_zero_eviction,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "comparator_frozen_weak": {
        "build": comparator_frozen_weak,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "config_strict_reject": {
        "build": config_strict_reject,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "decode_engine_owned": {
        "build": decode_engine_owned,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "error_code_family": {
        "build": error_code_family,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "decode_reservation_priority": {
        "build": decode_reservation_priority,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "observability_integrity": {
        "build": observability_integrity,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "prefill_queued_live_single": {
        "build": prefill_queued_live_single,
        "profiles": ["single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "prefill_queued_live_window": {
        "build": prefill_queued_live_window,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "decode_reserved_live_single": {
        "build": decode_reserved_live_single,
        "profiles": ["single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "decode_reserved_live_window": {
        "build": decode_reserved_live_window,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "cancel_not_found": {
        "build": cancel_not_found,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "cancel_tombstoned": {
        "build": cancel_tombstoned,
        "profiles": ["single-batch", "batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
}
