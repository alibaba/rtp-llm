"""Explicit preemption cohorts and legacy design-final predicates."""

from ..case_config import output


def same_priority_zero_eviction(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("same_priority_zero_eviction.setup_timeout_s"),
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "same_priority_zero_eviction.slow",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "sync", "balance_pause", params=case.value("same_priority_zero_eviction.sync")
    )
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=case.value("same_priority_zero_eviction.placeholder_timeout_s"),
        params=case.value("same_priority_zero_eviction.placeholder"),
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=case.value(
            "same_priority_zero_eviction.placeholder_settled_timeout_s"
        ),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params=case.params(
            "same_priority_zero_eviction.placeholder_admitted",
            {"actual": output("placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=case.value(
            "same_priority_zero_eviction.placeholder_pending_timeout_s"
        ),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("same_priority_zero_eviction.wave_timeout_s"),
        params=case.value("same_priority_zero_eviction.wave"),
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=case.value("same_priority_zero_eviction.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("same_priority_zero_eviction.placeholder_drain_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_wait",
        timeout_s=case.value("same_priority_zero_eviction.wave_drain_timeout_s"),
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
    case.step(
        "master_clean",
        "balance_clean",
        timeout_s=case.value("same_priority_zero_eviction.master_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "same_priority_zero_eviction.restore",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("same_priority_zero_eviction.teardown_timeout_s"),
    )


def prefill_queued(case):
    case.step("setup", "setup", timeout_s=case.value("prefill_queued.setup_timeout_s"))
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "prefill_queued.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("prefill_queued.sync"))
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=case.value("prefill_queued.r1_placeholder_timeout_s"),
        params=case.value("prefill_queued.r1_placeholder"),
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("prefill_queued.r1_placeholder_settled_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params=case.params(
            "prefill_queued.r1_placeholder_admitted",
            {"actual": output("r1_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("prefill_queued.r1_placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=case.value("prefill_queued.r1_wave_timeout_s"),
        params=case.value("prefill_queued.r1_wave"),
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=case.value("prefill_queued.r1_wave_settled_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("prefill_queued.r1_placeholder_drain_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=case.value("prefill_queued.r1_wave_drain_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_queued_first",
        params=case.params(
            "prefill_queued.r1_same_priority",
            {
                "placeholder": output("r1_placeholder", "requests"),
                "wave": output("r1_wave", "requests"),
            },
        ),
    )
    case.step(
        "r1_master_clean",
        "balance_clean",
        timeout_s=case.value("prefill_queued.r1_master_clean_timeout_s"),
    )
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=case.value("prefill_queued.r2_placeholder_timeout_s"),
        params=case.value("prefill_queued.r2_placeholder"),
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("prefill_queued.r2_placeholder_settled_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params=case.params(
            "prefill_queued.r2_placeholder_admitted",
            {"actual": output("r2_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("prefill_queued.r2_placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=case.value("prefill_queued.r2_wave_timeout_s"),
        params=case.value("prefill_queued.r2_wave"),
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=case.value("prefill_queued.r2_wave_settled_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("prefill_queued.r2_placeholder_drain_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=case.value("prefill_queued.r2_wave_drain_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_queued_second",
        params=case.params(
            "prefill_queued.r2_same_priority",
            {
                "placeholder": output("r2_placeholder", "requests"),
                "wave": output("r2_wave", "requests"),
            },
        ),
    )
    case.step(
        "r2_master_clean",
        "balance_clean",
        timeout_s=case.value("prefill_queued.r2_master_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "prefill_queued.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("prefill_queued.teardown_timeout_s"),
    )


def timeout_attribution(case):
    case.step(
        "setup", "setup", timeout_s=case.value("timeout_attribution.setup_timeout_s")
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "timeout_attribution.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("timeout_attribution.sync"))
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=case.value("timeout_attribution.r1_placeholder_timeout_s"),
        params=case.value("timeout_attribution.r1_placeholder"),
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("timeout_attribution.r1_placeholder_settled_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params=case.params(
            "timeout_attribution.r1_placeholder_admitted",
            {"actual": output("r1_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("timeout_attribution.r1_placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=case.value("timeout_attribution.r1_wave_timeout_s"),
        params=case.value("timeout_attribution.r1_wave"),
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=case.value("timeout_attribution.r1_wave_settled_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("timeout_attribution.r1_placeholder_drain_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=case.value("timeout_attribution.r1_wave_drain_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_expiry",
        params=case.params(
            "timeout_attribution.r1_same_priority",
            {
                "placeholder": output("r1_placeholder", "requests"),
                "wave": output("r1_wave", "requests"),
            },
        ),
    )
    case.step(
        "r1_master_clean",
        "balance_clean",
        timeout_s=case.value("timeout_attribution.r1_master_clean_timeout_s"),
    )
    case.step(
        "round2_slow",
        "engine_control",
        params=case.params(
            "timeout_attribution.round2_slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "round2_sync",
        "balance_pause",
        params=case.value("timeout_attribution.round2_sync"),
    )
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=case.value("timeout_attribution.r2_placeholder_timeout_s"),
        params=case.value("timeout_attribution.r2_placeholder"),
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("timeout_attribution.r2_placeholder_settled_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params=case.params(
            "timeout_attribution.r2_placeholder_admitted",
            {"actual": output("r2_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("timeout_attribution.r2_placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=case.value("timeout_attribution.r2_wave_timeout_s"),
        params=case.value("timeout_attribution.r2_wave"),
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=case.value("timeout_attribution.r2_wave_settled_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("timeout_attribution.r2_placeholder_drain_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=case.value("timeout_attribution.r2_wave_drain_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_expiry",
        params=case.params(
            "timeout_attribution.r2_same_priority",
            {
                "placeholder": output("r2_placeholder", "requests"),
                "wave": output("r2_wave", "requests"),
            },
        ),
    )
    case.step(
        "r2_master_clean",
        "balance_clean",
        timeout_s=case.value("timeout_attribution.r2_master_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "timeout_attribution.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("timeout_attribution.teardown_timeout_s"),
    )


def disabled_zero_eviction(case):
    case.step(
        "setup", "setup", timeout_s=case.value("disabled_zero_eviction.setup_timeout_s")
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "disabled_zero_eviction.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("disabled_zero_eviction.sync"))
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=case.value("disabled_zero_eviction.r1_placeholder_timeout_s"),
        params=case.value("disabled_zero_eviction.r1_placeholder"),
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("disabled_zero_eviction.r1_placeholder_settled_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params=case.params(
            "disabled_zero_eviction.r1_placeholder_admitted",
            {"actual": output("r1_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("disabled_zero_eviction.r1_placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=case.value("disabled_zero_eviction.r1_wave_timeout_s"),
        params=case.value("disabled_zero_eviction.r1_wave"),
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=case.value("disabled_zero_eviction.r1_wave_settled_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("disabled_zero_eviction.r1_placeholder_drain_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=case.value("disabled_zero_eviction.r1_wave_drain_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_disabled",
        params=case.params(
            "disabled_zero_eviction.r1_same_priority",
            {
                "placeholder": output("r1_placeholder", "requests"),
                "wave": output("r1_wave", "requests"),
            },
        ),
    )
    case.step(
        "r1_master_clean",
        "balance_clean",
        timeout_s=case.value("disabled_zero_eviction.r1_master_clean_timeout_s"),
    )
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=case.value("disabled_zero_eviction.r2_placeholder_timeout_s"),
        params=case.value("disabled_zero_eviction.r2_placeholder"),
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("disabled_zero_eviction.r2_placeholder_settled_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params=case.params(
            "disabled_zero_eviction.r2_placeholder_admitted",
            {"actual": output("r2_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("disabled_zero_eviction.r2_placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=case.value("disabled_zero_eviction.r2_wave_timeout_s"),
        params=case.value("disabled_zero_eviction.r2_wave"),
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=case.value("disabled_zero_eviction.r2_wave_settled_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("disabled_zero_eviction.r2_placeholder_drain_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=case.value("disabled_zero_eviction.r2_wave_drain_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_disabled",
        params=case.params(
            "disabled_zero_eviction.r2_same_priority",
            {
                "placeholder": output("r2_placeholder", "requests"),
                "wave": output("r2_wave", "requests"),
            },
        ),
    )
    case.step(
        "r2_master_clean",
        "balance_clean",
        timeout_s=case.value("disabled_zero_eviction.r2_master_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "disabled_zero_eviction.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("disabled_zero_eviction.teardown_timeout_s"),
    )


def comparator_frozen_weak(case):
    case.step(
        "setup", "setup", timeout_s=case.value("comparator_frozen_weak.setup_timeout_s")
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "comparator_frozen_weak.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("comparator_frozen_weak.sync"))
    case.step(
        "r1_placeholder",
        "priority_start",
        timeout_s=case.value("comparator_frozen_weak.r1_placeholder_timeout_s"),
        params=case.value("comparator_frozen_weak.r1_placeholder"),
    )
    case.step(
        "r1_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("comparator_frozen_weak.r1_placeholder_settled_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_placeholder_admitted",
        "check",
        params=case.params(
            "comparator_frozen_weak.r1_placeholder_admitted",
            {"actual": output("r1_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r1_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("comparator_frozen_weak.r1_placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "r1_wave",
        "priority_start",
        timeout_s=case.value("comparator_frozen_weak.r1_wave_timeout_s"),
        params=case.value("comparator_frozen_weak.r1_wave"),
    )
    case.step(
        "r1_wave_settled",
        "preemption_settled",
        timeout_s=case.value("comparator_frozen_weak.r1_wave_settled_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("comparator_frozen_weak.r1_placeholder_drain_timeout_s"),
        params={"requests": output("r1_placeholder", "requests")},
    )
    case.step(
        "r1_wave_drain",
        "preemption_wait",
        timeout_s=case.value("comparator_frozen_weak.r1_wave_drain_timeout_s"),
        params={"requests": output("r1_wave", "requests")},
    )
    case.step(
        "r1_same_priority",
        "preemption_comparator_half",
        params=case.params(
            "comparator_frozen_weak.r1_same_priority",
            {
                "placeholder": output("r1_placeholder", "requests"),
                "wave": output("r1_wave", "requests"),
            },
        ),
    )
    case.step(
        "r1_master_clean",
        "balance_clean",
        timeout_s=case.value("comparator_frozen_weak.r1_master_clean_timeout_s"),
    )
    case.step(
        "fifo_environment",
        "environment_reconfigure",
        timeout_s=case.value("comparator_frozen_weak.fifo_environment_timeout_s"),
        params=case.value("comparator_frozen_weak.fifo_environment"),
    )
    case.step("fifo_fleet", "priority_fleet")
    case.step(
        "fifo_slow",
        "engine_control",
        params=case.params(
            "comparator_frozen_weak.fifo_slow",
            {"targets": [output("fifo_fleet", "prefill")]},
        ),
    )
    case.step(
        "fifo_sync",
        "balance_pause",
        params=case.value("comparator_frozen_weak.fifo_sync"),
    )
    case.step(
        "r2_placeholder",
        "priority_start",
        timeout_s=case.value("comparator_frozen_weak.r2_placeholder_timeout_s"),
        params=case.value("comparator_frozen_weak.r2_placeholder"),
    )
    case.step(
        "r2_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("comparator_frozen_weak.r2_placeholder_settled_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_placeholder_admitted",
        "check",
        params=case.params(
            "comparator_frozen_weak.r2_placeholder_admitted",
            {"actual": output("r2_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "r2_placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("comparator_frozen_weak.r2_placeholder_pending_timeout_s"),
        params={"target": output("fifo_fleet", "prefill")},
    )
    case.step(
        "r2_wave",
        "priority_start",
        timeout_s=case.value("comparator_frozen_weak.r2_wave_timeout_s"),
        params=case.value("comparator_frozen_weak.r2_wave"),
    )
    case.step(
        "r2_wave_settled",
        "preemption_settled",
        timeout_s=case.value("comparator_frozen_weak.r2_wave_settled_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("comparator_frozen_weak.r2_placeholder_drain_timeout_s"),
        params={"requests": output("r2_placeholder", "requests")},
    )
    case.step(
        "r2_wave_drain",
        "preemption_wait",
        timeout_s=case.value("comparator_frozen_weak.r2_wave_drain_timeout_s"),
        params={"requests": output("r2_wave", "requests")},
    )
    case.step(
        "r2_same_priority",
        "preemption_comparator_half",
        params=case.params(
            "comparator_frozen_weak.r2_same_priority",
            {
                "placeholder": output("r2_placeholder", "requests"),
                "wave": output("r2_wave", "requests"),
            },
        ),
    )
    case.step(
        "r2_master_clean",
        "balance_clean",
        timeout_s=case.value("comparator_frozen_weak.r2_master_clean_timeout_s"),
    )
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
        params=case.params(
            "comparator_frozen_weak.restore",
            {"targets": [output("fifo_fleet", "prefill")]},
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("comparator_frozen_weak.teardown_timeout_s"),
    )


def config_strict_reject(case):
    case.step(
        "setup", "setup", timeout_s=case.value("config_strict_reject.setup_timeout_s")
    )
    case.step(
        "removed_auto_tpm",
        "environment_startup_probe",
        timeout_s=case.value("config_strict_reject.removed_auto_tpm_timeout_s"),
        params=case.value("config_strict_reject.removed_auto_tpm"),
    )
    case.step(
        "fifo_default_priority",
        "environment_startup_probe",
        timeout_s=case.value("config_strict_reject.fifo_default_priority_timeout_s"),
        params=case.value("config_strict_reject.fifo_default_priority"),
    )
    case.step(
        "owned_without_cancellation",
        "environment_startup_probe",
        timeout_s=case.value(
            "config_strict_reject.owned_without_cancellation_timeout_s"
        ),
        params=case.value("config_strict_reject.owned_without_cancellation"),
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
    case.step(
        "setup", "setup", timeout_s=case.value("decode_engine_owned.setup_timeout_s")
    )
    case.step("fleet", "preemption_decode_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "decode_engine_owned.slow",
            {"targets": [output("fleet", "prefill0"), output("fleet", "prefill1")]},
        ),
    )
    case.step("sync", "balance_pause", params=case.value("decode_engine_owned.sync"))
    case.step(
        "r1_occupants",
        "priority_start",
        timeout_s=case.value("decode_engine_owned.r1_occupants_timeout_s"),
        params=case.value("decode_engine_owned.r1_occupants"),
    )
    case.step(
        "r1_settled",
        "preemption_settled",
        timeout_s=case.value("decode_engine_owned.r1_settled_timeout_s"),
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_pressure",
        "preemption_decode_pressure",
        params=case.params(
            "decode_engine_owned.r1_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "r1_pressure_sync",
        "balance_pause",
        params=case.value("decode_engine_owned.r1_pressure_sync"),
    )
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
        timeout_s=case.value("decode_engine_owned.r1_incoming_timeout_s"),
        params=case.value("decode_engine_owned.r1_incoming"),
    )
    case.step(
        "r1_incoming_settled",
        "preemption_settled",
        timeout_s=case.value("decode_engine_owned.r1_incoming_settled_timeout_s"),
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_occupants_drain",
        "preemption_wait",
        timeout_s=case.value("decode_engine_owned.r1_occupants_drain_timeout_s"),
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_incoming_drain",
        "preemption_wait",
        timeout_s=case.value("decode_engine_owned.r1_incoming_drain_timeout_s"),
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_verdict",
        "preemption_decode_half",
        params=case.params(
            "decode_engine_owned.r1_verdict",
            {
                "occupants": output("r1_occupants", "requests"),
                "incoming": output("r1_incoming", "requests"),
            },
        ),
    )
    case.step(
        "r1_master_clean",
        "balance_clean",
        timeout_s=case.value("decode_engine_owned.r1_master_clean_timeout_s"),
    )
    case.step(
        "release_pressure",
        "preemption_decode_pressure",
        params=case.params(
            "decode_engine_owned.release_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "release_sync",
        "balance_pause",
        params=case.value("decode_engine_owned.release_sync"),
    )
    case.step(
        "r2_occupants",
        "priority_start",
        timeout_s=case.value("decode_engine_owned.r2_occupants_timeout_s"),
        params=case.value("decode_engine_owned.r2_occupants"),
    )
    case.step(
        "r2_settled",
        "preemption_settled",
        timeout_s=case.value("decode_engine_owned.r2_settled_timeout_s"),
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_running",
        "preemption_decode_running",
        timeout_s=case.value("decode_engine_owned.r2_running_timeout_s"),
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_pressure",
        "preemption_decode_pressure",
        params=case.params(
            "decode_engine_owned.r2_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "r2_pressure_sync",
        "balance_pause",
        params=case.value("decode_engine_owned.r2_pressure_sync"),
    )
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
        timeout_s=case.value("decode_engine_owned.r2_incoming_timeout_s"),
        params=case.value("decode_engine_owned.r2_incoming"),
    )
    case.step(
        "r2_incoming_settled",
        "preemption_settled",
        timeout_s=case.value("decode_engine_owned.r2_incoming_settled_timeout_s"),
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step(
        "r2_occupants_drain",
        "preemption_wait",
        timeout_s=case.value("decode_engine_owned.r2_occupants_drain_timeout_s"),
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_incoming_drain",
        "preemption_wait",
        timeout_s=case.value("decode_engine_owned.r2_incoming_drain_timeout_s"),
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step(
        "r2_verdict",
        "preemption_decode_half",
        params=case.params(
            "decode_engine_owned.r2_verdict",
            {
                "occupants": output("r2_occupants", "requests"),
                "incoming": output("r2_incoming", "requests"),
            },
        ),
    )
    case.step(
        "r2_master_clean",
        "balance_clean",
        timeout_s=case.value("decode_engine_owned.r2_master_clean_timeout_s"),
    )
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
        params=case.params(
            "decode_engine_owned.clear_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "decode_engine_owned.restore",
            {"targets": [output("fleet", "prefill0"), output("fleet", "prefill1")]},
        ),
    )
    case.step("teardown", "teardown")


def error_code_family(case):
    case.step(
        "setup", "setup", timeout_s=case.value("error_code_family.setup_timeout_s")
    )
    case.step(
        "s1_fleet", "balance_snapshot", params=case.value("error_code_family.s1_fleet")
    )
    case.step(
        "s1_slow",
        "engine_control",
        params=case.params(
            "error_code_family.s1_slow",
            {"targets": [output("s1_fleet", "first"), output("s1_fleet", "second")]},
        ),
    )
    case.step(
        "s1_sync", "balance_pause", params=case.value("error_code_family.s1_sync")
    )
    case.step(
        "s1_placeholder",
        "priority_start",
        timeout_s=case.value("error_code_family.s1_placeholder_timeout_s"),
        params=case.value("error_code_family.s1_placeholder"),
    )
    case.step(
        "s1_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("error_code_family.s1_placeholder_settled_timeout_s"),
        params={"requests": output("s1_placeholder", "requests")},
    )
    case.step(
        "s1_admitted",
        "check",
        params=case.params(
            "error_code_family.s1_admitted",
            {"actual": output("s1_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "s1_pending",
        "preemption_pending",
        timeout_s=case.value("error_code_family.s1_pending_timeout_s"),
        params={"target": output("s1_fleet", "first")},
    )
    case.step(
        "s1_wave",
        "priority_start",
        timeout_s=case.value("error_code_family.s1_wave_timeout_s"),
        params=case.value("error_code_family.s1_wave"),
    )
    case.step(
        "s1_wave_settled",
        "preemption_settled",
        timeout_s=case.value("error_code_family.s1_wave_settled_timeout_s"),
        params={"requests": output("s1_wave", "requests")},
    )
    case.step(
        "s1_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("error_code_family.s1_placeholder_drain_timeout_s"),
        params={"requests": output("s1_placeholder", "requests")},
    )
    case.step(
        "s1_wave_drain",
        "preemption_wait",
        timeout_s=case.value("error_code_family.s1_wave_drain_timeout_s"),
        params={"requests": output("s1_wave", "requests")},
    )
    case.step(
        "s1_segment",
        "preemption_error_segment",
        params=case.params(
            "error_code_family.s1_segment",
            {
                "placeholder": output("s1_placeholder", "requests"),
                "wave": output("s1_wave", "requests"),
            },
        ),
    )
    case.step(
        "s1_restore",
        "engine_control",
        params=case.params(
            "error_code_family.s1_restore",
            {"targets": [output("s1_fleet", "first"), output("s1_fleet", "second")]},
        ),
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params=case.value("error_code_family.recovery_prepare"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("error_code_family.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "recovery_result",
        "preemption_error_recovery",
        params={"requests": output("recovery_prepare", "requests")},
    )
    case.step(
        "s1_clean",
        "balance_clean",
        timeout_s=case.value("error_code_family.s1_clean_timeout_s"),
    )
    case.step(
        "s2_environment",
        "environment_reconfigure",
        timeout_s=case.value("error_code_family.s2_environment_timeout_s"),
        params=case.value("error_code_family.s2_environment"),
    )
    case.step("s2_fleet", "priority_fleet")
    case.step(
        "s2_slow",
        "engine_control",
        params=case.params(
            "error_code_family.s2_slow", {"targets": [output("s2_fleet", "prefill")]}
        ),
    )
    case.step(
        "s2_sync", "balance_pause", params=case.value("error_code_family.s2_sync")
    )
    case.step(
        "s2_placeholder",
        "priority_start",
        timeout_s=case.value("error_code_family.s2_placeholder_timeout_s"),
        params=case.value("error_code_family.s2_placeholder"),
    )
    case.step(
        "s2_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("error_code_family.s2_placeholder_settled_timeout_s"),
        params={"requests": output("s2_placeholder", "requests")},
    )
    case.step(
        "s2_admitted",
        "check",
        params=case.params(
            "error_code_family.s2_admitted",
            {"actual": output("s2_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "s2_pending",
        "preemption_pending",
        timeout_s=case.value("error_code_family.s2_pending_timeout_s"),
        params={"target": output("s2_fleet", "prefill")},
    )
    case.step(
        "s2_wave",
        "priority_start",
        timeout_s=case.value("error_code_family.s2_wave_timeout_s"),
        params=case.value("error_code_family.s2_wave"),
    )
    case.step(
        "s2_wave_settled",
        "preemption_settled",
        timeout_s=case.value("error_code_family.s2_wave_settled_timeout_s"),
        params={"requests": output("s2_wave", "requests")},
    )
    case.step(
        "s2_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("error_code_family.s2_placeholder_drain_timeout_s"),
        params={"requests": output("s2_placeholder", "requests")},
    )
    case.step(
        "s2_wave_drain",
        "preemption_wait",
        timeout_s=case.value("error_code_family.s2_wave_drain_timeout_s"),
        params={"requests": output("s2_wave", "requests")},
    )
    case.step(
        "s2_segment",
        "preemption_error_segment",
        params=case.params(
            "error_code_family.s2_segment",
            {
                "placeholder": output("s2_placeholder", "requests"),
                "wave": output("s2_wave", "requests"),
            },
        ),
    )
    case.step(
        "s2_clean",
        "balance_clean",
        timeout_s=case.value("error_code_family.s2_clean_timeout_s"),
    )
    case.step(
        "s2_restore",
        "engine_control",
        params=case.params(
            "error_code_family.s2_restore", {"targets": [output("s2_fleet", "prefill")]}
        ),
    )
    case.step(
        "s3_environment",
        "environment_reconfigure",
        timeout_s=case.value("error_code_family.s3_environment_timeout_s"),
        params=case.value("error_code_family.s3_environment"),
    )
    case.step("s3_fleet", "priority_fleet")
    case.step(
        "s3_slow",
        "engine_control",
        params=case.params(
            "error_code_family.s3_slow", {"targets": [output("s3_fleet", "prefill")]}
        ),
    )
    case.step(
        "s3_sync", "balance_pause", params=case.value("error_code_family.s3_sync")
    )
    case.step(
        "s3_placeholder",
        "priority_start",
        timeout_s=case.value("error_code_family.s3_placeholder_timeout_s"),
        params=case.value("error_code_family.s3_placeholder"),
    )
    case.step(
        "s3_placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("error_code_family.s3_placeholder_settled_timeout_s"),
        params={"requests": output("s3_placeholder", "requests")},
    )
    case.step(
        "s3_admitted",
        "check",
        params=case.params(
            "error_code_family.s3_admitted",
            {"actual": output("s3_placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "s3_pending",
        "preemption_pending",
        timeout_s=case.value("error_code_family.s3_pending_timeout_s"),
        params={"target": output("s3_fleet", "prefill")},
    )
    case.step(
        "s3_wave",
        "priority_start",
        timeout_s=case.value("error_code_family.s3_wave_timeout_s"),
        params=case.value("error_code_family.s3_wave"),
    )
    case.step(
        "s3_wave_settled",
        "preemption_settled",
        timeout_s=case.value("error_code_family.s3_wave_settled_timeout_s"),
        params={"requests": output("s3_wave", "requests")},
    )
    case.step(
        "s3_placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("error_code_family.s3_placeholder_drain_timeout_s"),
        params={"requests": output("s3_placeholder", "requests")},
    )
    case.step(
        "s3_wave_drain",
        "preemption_wait",
        timeout_s=case.value("error_code_family.s3_wave_drain_timeout_s"),
        params={"requests": output("s3_wave", "requests")},
    )
    case.step(
        "s3_segment",
        "preemption_error_segment",
        params=case.params(
            "error_code_family.s3_segment",
            {
                "placeholder": output("s3_placeholder", "requests"),
                "wave": output("s3_wave", "requests"),
            },
        ),
    )
    case.step(
        "s3_clean",
        "balance_clean",
        timeout_s=case.value("error_code_family.s3_clean_timeout_s"),
    )
    case.step(
        "s3_restore",
        "engine_control",
        params=case.params(
            "error_code_family.s3_restore", {"targets": [output("s3_fleet", "prefill")]}
        ),
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
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_reservation_priority.setup_timeout_s"),
    )
    case.step("fleet", "preemption_decode_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "decode_reservation_priority.slow",
            {"targets": [output("fleet", "prefill0"), output("fleet", "prefill1")]},
        ),
    )
    case.step(
        "sync", "balance_pause", params=case.value("decode_reservation_priority.sync")
    )
    case.step(
        "r1_occupants",
        "priority_start",
        timeout_s=case.value("decode_reservation_priority.r1_occupants_timeout_s"),
        params=case.value("decode_reservation_priority.r1_occupants"),
    )
    case.step(
        "r1_settled",
        "preemption_settled",
        timeout_s=case.value("decode_reservation_priority.r1_settled_timeout_s"),
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_running",
        "preemption_decode_running",
        timeout_s=case.value("decode_reservation_priority.r1_running_timeout_s"),
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_baseline",
        "preemption_reservation_metric",
        params=case.value("decode_reservation_priority.r1_baseline"),
    )
    case.step(
        "r1_pressure",
        "preemption_decode_pressure",
        params=case.params(
            "decode_reservation_priority.r1_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "r1_pressure_sync",
        "balance_pause",
        params=case.value("decode_reservation_priority.r1_pressure_sync"),
    )
    case.step(
        "r1_incoming",
        "priority_start",
        timeout_s=case.value("decode_reservation_priority.r1_incoming_timeout_s"),
        params=case.value("decode_reservation_priority.r1_incoming"),
    )
    case.step(
        "r1_incoming_settled",
        "preemption_settled",
        timeout_s=case.value(
            "decode_reservation_priority.r1_incoming_settled_timeout_s"
        ),
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_occupants_drain",
        "preemption_wait",
        timeout_s=case.value(
            "decode_reservation_priority.r1_occupants_drain_timeout_s"
        ),
        params={"requests": output("r1_occupants", "requests")},
    )
    case.step(
        "r1_incoming_drain",
        "preemption_wait",
        timeout_s=case.value("decode_reservation_priority.r1_incoming_drain_timeout_s"),
        params={"requests": output("r1_incoming", "requests")},
    )
    case.step(
        "r1_after",
        "preemption_reservation_metric",
        params=case.value("decode_reservation_priority.r1_after"),
    )
    case.step(
        "r1_verdict",
        "preemption_reservation_half",
        params=case.params(
            "decode_reservation_priority.r1_verdict",
            {
                "occupants": output("r1_occupants", "requests"),
                "incoming": output("r1_incoming", "requests"),
                "baseline": output("r1_baseline", "snapshot"),
                "after": output("r1_after", "snapshot"),
            },
        ),
    )
    case.step(
        "r1_clean",
        "balance_clean",
        timeout_s=case.value("decode_reservation_priority.r1_clean_timeout_s"),
    )
    case.step(
        "r2_release",
        "preemption_decode_pressure",
        params=case.params(
            "decode_reservation_priority.r2_release",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "r2_release_sync",
        "balance_pause",
        params=case.value("decode_reservation_priority.r2_release_sync"),
    )
    case.step(
        "r2_occupants",
        "priority_start",
        timeout_s=case.value("decode_reservation_priority.r2_occupants_timeout_s"),
        params=case.value("decode_reservation_priority.r2_occupants"),
    )
    case.step(
        "r2_settled",
        "preemption_settled",
        timeout_s=case.value("decode_reservation_priority.r2_settled_timeout_s"),
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_running",
        "preemption_decode_running",
        timeout_s=case.value("decode_reservation_priority.r2_running_timeout_s"),
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_baseline",
        "preemption_reservation_metric",
        params=case.value("decode_reservation_priority.r2_baseline"),
    )
    case.step(
        "r2_pressure",
        "preemption_decode_pressure",
        params=case.params(
            "decode_reservation_priority.r2_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "r2_pressure_sync",
        "balance_pause",
        params=case.value("decode_reservation_priority.r2_pressure_sync"),
    )
    case.step(
        "r2_incoming",
        "priority_start",
        timeout_s=case.value("decode_reservation_priority.r2_incoming_timeout_s"),
        params=case.value("decode_reservation_priority.r2_incoming"),
    )
    case.step(
        "r2_incoming_settled",
        "preemption_settled",
        timeout_s=case.value(
            "decode_reservation_priority.r2_incoming_settled_timeout_s"
        ),
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step(
        "r2_occupants_drain",
        "preemption_wait",
        timeout_s=case.value(
            "decode_reservation_priority.r2_occupants_drain_timeout_s"
        ),
        params={"requests": output("r2_occupants", "requests")},
    )
    case.step(
        "r2_incoming_drain",
        "preemption_wait",
        timeout_s=case.value("decode_reservation_priority.r2_incoming_drain_timeout_s"),
        params={"requests": output("r2_incoming", "requests")},
    )
    case.step(
        "r2_after",
        "preemption_reservation_metric",
        params=case.value("decode_reservation_priority.r2_after"),
    )
    case.step(
        "r2_verdict",
        "preemption_reservation_half",
        params=case.params(
            "decode_reservation_priority.r2_verdict",
            {
                "occupants": output("r2_occupants", "requests"),
                "incoming": output("r2_incoming", "requests"),
                "baseline": output("r2_baseline", "snapshot"),
                "after": output("r2_after", "snapshot"),
            },
        ),
    )
    case.step(
        "r2_clean",
        "balance_clean",
        timeout_s=case.value("decode_reservation_priority.r2_clean_timeout_s"),
    )
    case.step(
        "r3_release",
        "preemption_decode_pressure",
        params=case.params(
            "decode_reservation_priority.r3_release",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "r3_release_sync",
        "balance_pause",
        params=case.value("decode_reservation_priority.r3_release_sync"),
    )
    case.step(
        "r3_occupants",
        "priority_start",
        timeout_s=case.value("decode_reservation_priority.r3_occupants_timeout_s"),
        params=case.value("decode_reservation_priority.r3_occupants"),
    )
    case.step(
        "r3_settled",
        "preemption_settled",
        timeout_s=case.value("decode_reservation_priority.r3_settled_timeout_s"),
        params={"requests": output("r3_occupants", "requests")},
    )
    case.step(
        "r3_running",
        "preemption_decode_running",
        timeout_s=case.value("decode_reservation_priority.r3_running_timeout_s"),
        params={"requests": output("r3_occupants", "requests")},
    )
    case.step(
        "r3_pressure",
        "preemption_decode_pressure",
        params=case.params(
            "decode_reservation_priority.r3_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "r3_pressure_sync",
        "balance_pause",
        params=case.value("decode_reservation_priority.r3_pressure_sync"),
    )
    case.step(
        "r3_incoming",
        "priority_start",
        timeout_s=case.value("decode_reservation_priority.r3_incoming_timeout_s"),
        params=case.value("decode_reservation_priority.r3_incoming"),
    )
    case.step(
        "r3_incoming_settled",
        "preemption_settled",
        timeout_s=case.value(
            "decode_reservation_priority.r3_incoming_settled_timeout_s"
        ),
        params={"requests": output("r3_incoming", "requests")},
    )
    case.step(
        "r3_occupants_drain",
        "preemption_wait",
        timeout_s=case.value(
            "decode_reservation_priority.r3_occupants_drain_timeout_s"
        ),
        params={"requests": output("r3_occupants", "requests")},
    )
    case.step(
        "r3_incoming_drain",
        "preemption_wait",
        timeout_s=case.value("decode_reservation_priority.r3_incoming_drain_timeout_s"),
        params={"requests": output("r3_incoming", "requests")},
    )
    case.step(
        "r3_verdict",
        "preemption_reservation_half",
        params=case.params(
            "decode_reservation_priority.r3_verdict",
            {
                "occupants": output("r3_occupants", "requests"),
                "incoming": output("r3_incoming", "requests"),
            },
        ),
    )
    case.step(
        "r3_clean",
        "balance_clean",
        timeout_s=case.value("decode_reservation_priority.r3_clean_timeout_s"),
    )
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
        params=case.params(
            "decode_reservation_priority.clear_pressure",
            {
                "targets": [
                    output("fleet", "decode0"),
                    output("fleet", "decode1"),
                    output("fleet", "decode2"),
                    output("fleet", "decode3"),
                ]
            },
        ),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "decode_reservation_priority.restore",
            {"targets": [output("fleet", "prefill0"), output("fleet", "prefill1")]},
        ),
    )
    case.step("teardown", "teardown")


def observability_integrity(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("observability_integrity.setup_timeout_s"),
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "observability_integrity.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "sync", "balance_pause", params=case.value("observability_integrity.sync")
    )
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=case.value("observability_integrity.placeholder_timeout_s"),
        params=case.value("observability_integrity.placeholder"),
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=case.value("observability_integrity.placeholder_settled_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params=case.params(
            "observability_integrity.placeholder_admitted",
            {"actual": output("placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=case.value("observability_integrity.placeholder_pending_timeout_s"),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "duplicate",
        "preemption_observability_duplicate",
        timeout_s=case.value("observability_integrity.duplicate_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("observability_integrity.wave_timeout_s"),
        params=case.value("observability_integrity.wave"),
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=case.value("observability_integrity.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_wait",
        timeout_s=case.value("observability_integrity.placeholder_drain_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_wait",
        timeout_s=case.value("observability_integrity.wave_drain_timeout_s"),
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
    case.step(
        "master_clean",
        "balance_clean",
        timeout_s=case.value("observability_integrity.master_clean_timeout_s"),
    )
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
        params=case.params(
            "observability_integrity.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("observability_integrity.teardown_timeout_s"),
    )


def prefill_queued_live_single(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("prefill_queued_live_single.setup_timeout_s"),
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "prefill_queued_live_single.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "sync", "balance_pause", params=case.value("prefill_queued_live_single.sync")
    )
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=case.value("prefill_queued_live_single.placeholder_timeout_s"),
        params=case.value("prefill_queued_live_single.placeholder"),
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=case.value(
            "prefill_queued_live_single.placeholder_settled_timeout_s"
        ),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params=case.params(
            "prefill_queued_live_single.placeholder_admitted",
            {"actual": output("placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=case.value(
            "prefill_queued_live_single.placeholder_pending_timeout_s"
        ),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=case.value("prefill_queued_live_single.wave_timeout_s"),
        params=case.value("prefill_queued_live_single.wave"),
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=case.value("prefill_queued_live_single.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=case.value("prefill_queued_live_single.placeholder_drain_timeout_s"),
        params=case.params(
            "prefill_queued_live_single.placeholder_drain",
            {"requests": output("placeholder", "requests")},
        ),
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=case.value("prefill_queued_live_single.wave_drain_timeout_s"),
        params=case.params(
            "prefill_queued_live_single.wave_drain",
            {"requests": output("wave", "requests")},
        ),
    )
    case.step(
        "live_verdict",
        "preemption_live_prefill",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step(
        "master_clean",
        "balance_clean",
        timeout_s=case.value("prefill_queued_live_single.master_clean_timeout_s"),
    )
    case.step(
        "engine_clean",
        "preemption_live_engine_clean",
        timeout_s=case.value("prefill_queued_live_single.engine_clean_timeout_s"),
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params=case.value("prefill_queued_live_single.recovery_prepare"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("prefill_queued_live_single.recovery_dispatch_timeout_s"),
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
        params=case.params(
            "prefill_queued_live_single.restore",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("prefill_queued_live_single.teardown_timeout_s"),
    )


def prefill_queued_live_window(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("prefill_queued_live_window.setup_timeout_s"),
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "prefill_queued_live_window.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "sync", "balance_pause", params=case.value("prefill_queued_live_window.sync")
    )
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=case.value("prefill_queued_live_window.placeholder_timeout_s"),
        params=case.value("prefill_queued_live_window.placeholder"),
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=case.value(
            "prefill_queued_live_window.placeholder_settled_timeout_s"
        ),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params=case.params(
            "prefill_queued_live_window.placeholder_admitted",
            {"actual": output("placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=case.value(
            "prefill_queued_live_window.placeholder_pending_timeout_s"
        ),
        params={"target": output("fleet", "prefill")},
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=case.value("prefill_queued_live_window.wave_timeout_s"),
        params=case.value("prefill_queued_live_window.wave"),
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=case.value("prefill_queued_live_window.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=case.value("prefill_queued_live_window.placeholder_drain_timeout_s"),
        params=case.params(
            "prefill_queued_live_window.placeholder_drain",
            {"requests": output("placeholder", "requests")},
        ),
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=case.value("prefill_queued_live_window.wave_drain_timeout_s"),
        params=case.params(
            "prefill_queued_live_window.wave_drain",
            {"requests": output("wave", "requests")},
        ),
    )
    case.step(
        "live_verdict",
        "preemption_live_prefill",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step(
        "master_clean",
        "balance_clean",
        timeout_s=case.value("prefill_queued_live_window.master_clean_timeout_s"),
    )
    case.step(
        "engine_clean",
        "preemption_live_engine_clean",
        timeout_s=case.value("prefill_queued_live_window.engine_clean_timeout_s"),
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params=case.value("prefill_queued_live_window.recovery_prepare"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("prefill_queued_live_window.recovery_dispatch_timeout_s"),
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
        params=case.params(
            "prefill_queued_live_window.restore",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("prefill_queued_live_window.teardown_timeout_s"),
    )


def decode_reserved_live_single(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_reserved_live_single.setup_timeout_s"),
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "decode_reserved_live_single.slow",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "sync", "balance_pause", params=case.value("decode_reserved_live_single.sync")
    )
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=case.value("decode_reserved_live_single.placeholder_timeout_s"),
        params=case.value("decode_reserved_live_single.placeholder"),
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=case.value(
            "decode_reserved_live_single.placeholder_settled_timeout_s"
        ),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params=case.params(
            "decode_reserved_live_single.placeholder_admitted",
            {"actual": output("placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=case.value(
            "decode_reserved_live_single.placeholder_pending_timeout_s"
        ),
        params={"target": output("fleet", "prefill")},
    )
    # Release the placeholder before waiting for incoming admission; otherwise
    # its deferred Fetch is behind the very Schedule waiting for its capacity.
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=case.value("decode_reserved_live_single.placeholder_drain_timeout_s"),
        params=case.params(
            "decode_reserved_live_single.placeholder_drain",
            {"requests": output("placeholder", "requests")},
        ),
    )
    case.step(
        "placeholder_master_clean",
        "balance_clean",
        timeout_s=case.value(
            "decode_reserved_live_single.placeholder_master_clean_timeout_s"
        ),
    )
    case.step(
        "placeholder_engine_clean",
        "preemption_live_engine_clean",
        timeout_s=case.value(
            "decode_reserved_live_single.placeholder_engine_clean_timeout_s"
        ),
    )
    case.step(
        "placeholder_released",
        "check",
        params=case.params(
            "decode_reserved_live_single.placeholder_released",
            {"actual": output("placeholder_engine_clean", "passed")},
        ),
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=case.value("decode_reserved_live_single.wave_timeout_s"),
        params=case.value("decode_reserved_live_single.wave"),
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=case.value("decode_reserved_live_single.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=case.value("decode_reserved_live_single.wave_drain_timeout_s"),
        params=case.params(
            "decode_reserved_live_single.wave_drain",
            {"requests": output("wave", "requests")},
        ),
    )
    case.step(
        "live_verdict",
        "preemption_live_reserved",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step(
        "master_clean",
        "balance_clean",
        timeout_s=case.value("decode_reserved_live_single.master_clean_timeout_s"),
    )
    case.step(
        "engine_clean",
        "preemption_live_engine_clean",
        timeout_s=case.value("decode_reserved_live_single.engine_clean_timeout_s"),
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params=case.value("decode_reserved_live_single.recovery_prepare"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("decode_reserved_live_single.recovery_dispatch_timeout_s"),
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
        params=case.params(
            "decode_reserved_live_single.restore",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("decode_reserved_live_single.teardown_timeout_s"),
    )


def decode_reserved_live_window(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_reserved_live_window.setup_timeout_s"),
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "decode_reserved_live_window.slow",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "sync", "balance_pause", params=case.value("decode_reserved_live_window.sync")
    )
    case.step(
        "placeholder",
        "preemption_live_start",
        timeout_s=case.value("decode_reserved_live_window.placeholder_timeout_s"),
        params=case.value("decode_reserved_live_window.placeholder"),
    )
    case.step(
        "placeholder_settled",
        "preemption_settled",
        timeout_s=case.value(
            "decode_reserved_live_window.placeholder_settled_timeout_s"
        ),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "placeholder_admitted",
        "check",
        params=case.params(
            "decode_reserved_live_window.placeholder_admitted",
            {"actual": output("placeholder_settled", "admitted")},
        ),
    )
    case.step(
        "placeholder_pending",
        "preemption_pending",
        timeout_s=case.value(
            "decode_reserved_live_window.placeholder_pending_timeout_s"
        ),
        params={"target": output("fleet", "prefill")},
    )
    # Release the placeholder before waiting for incoming admission; otherwise
    # its deferred Fetch is behind the very Schedule waiting for its capacity.
    case.step(
        "placeholder_drain",
        "preemption_live_drain",
        timeout_s=case.value("decode_reserved_live_window.placeholder_drain_timeout_s"),
        params=case.params(
            "decode_reserved_live_window.placeholder_drain",
            {"requests": output("placeholder", "requests")},
        ),
    )
    case.step(
        "placeholder_master_clean",
        "balance_clean",
        timeout_s=case.value(
            "decode_reserved_live_window.placeholder_master_clean_timeout_s"
        ),
    )
    case.step(
        "placeholder_engine_clean",
        "preemption_live_engine_clean",
        timeout_s=case.value(
            "decode_reserved_live_window.placeholder_engine_clean_timeout_s"
        ),
    )
    case.step(
        "placeholder_released",
        "check",
        params=case.params(
            "decode_reserved_live_window.placeholder_released",
            {"actual": output("placeholder_engine_clean", "passed")},
        ),
    )
    case.step(
        "wave",
        "preemption_live_start",
        timeout_s=case.value("decode_reserved_live_window.wave_timeout_s"),
        params=case.value("decode_reserved_live_window.wave"),
    )
    case.step(
        "wave_settled",
        "preemption_settled",
        timeout_s=case.value("decode_reserved_live_window.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "wave_drain",
        "preemption_live_drain",
        timeout_s=case.value("decode_reserved_live_window.wave_drain_timeout_s"),
        params=case.params(
            "decode_reserved_live_window.wave_drain",
            {"requests": output("wave", "requests")},
        ),
    )
    case.step(
        "live_verdict",
        "preemption_live_reserved",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step(
        "master_clean",
        "balance_clean",
        timeout_s=case.value("decode_reserved_live_window.master_clean_timeout_s"),
    )
    case.step(
        "engine_clean",
        "preemption_live_engine_clean",
        timeout_s=case.value("decode_reserved_live_window.engine_clean_timeout_s"),
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params=case.value("decode_reserved_live_window.recovery_prepare"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("decode_reserved_live_window.recovery_dispatch_timeout_s"),
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
        params=case.params(
            "decode_reserved_live_window.restore",
            {"targets": [output("fleet", "prefill")]},
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("decode_reserved_live_window.teardown_timeout_s"),
    )


def cancel_not_found(case):
    case.step(
        "setup", "setup", timeout_s=case.value("cancel_not_found.setup_timeout_s")
    )
    case.step(
        "victim",
        "priority_start",
        timeout_s=case.value("cancel_not_found.victim_timeout_s"),
        params=case.value("cancel_not_found.victim"),
    )
    case.step(
        "victim_settled",
        "preemption_settled",
        timeout_s=case.value("cancel_not_found.victim_settled_timeout_s"),
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_admitted",
        "check",
        params=case.params(
            "cancel_not_found.victim_admitted",
            {"actual": output("victim_settled", "admitted")},
        ),
    )
    case.step(
        "victim_running",
        "preemption_nf_state",
        timeout_s=case.value("cancel_not_found.victim_running_timeout_s"),
        params=case.params(
            "cancel_not_found.victim_running",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "before_freeze_pause",
        "balance_pause",
        params=case.value("cancel_not_found.before_freeze_pause"),
    )
    case.step("baseline_cancel", "preemption_cancel_census")
    case.step(
        "freeze_status",
        "status_control",
        params=case.value("cancel_not_found.freeze_status"),
    )
    case.step(
        "victim_engine_finished",
        "preemption_nf_state",
        timeout_s=case.value("cancel_not_found.victim_engine_finished_timeout_s"),
        params=case.params(
            "cancel_not_found.victim_engine_finished",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "incoming",
        "priority_start",
        timeout_s=case.value("cancel_not_found.incoming_timeout_s"),
        params=case.value("cancel_not_found.incoming"),
    )
    case.step(
        "incoming_settled",
        "preemption_settled",
        timeout_s=case.value("cancel_not_found.incoming_settled_timeout_s"),
        params={"requests": output("incoming", "requests")},
    )
    case.step(
        "victim_drain",
        "preemption_wait",
        timeout_s=case.value("cancel_not_found.victim_drain_timeout_s"),
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
        params=case.value("cancel_not_found.clear_status"),
    )
    case.step(
        "master_clean",
        "balance_clean",
        timeout_s=case.value("cancel_not_found.master_clean_timeout_s"),
    )
    case.step(
        "engine_clean",
        "preemption_nf_engine_clean",
        timeout_s=case.value("cancel_not_found.engine_clean_timeout_s"),
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params=case.value("cancel_not_found.recovery_prepare"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("cancel_not_found.recovery_dispatch_timeout_s"),
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
    case.step(
        "setup", "setup", timeout_s=case.value("cancel_tombstoned.setup_timeout_s")
    )
    case.step(
        "victim",
        "preemption_live_start",
        timeout_s=case.value("cancel_tombstoned.victim_timeout_s"),
        params=case.value("cancel_tombstoned.victim"),
    )
    case.step(
        "victim_settled",
        "preemption_settled",
        timeout_s=case.value("cancel_tombstoned.victim_settled_timeout_s"),
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_admitted",
        "check",
        params=case.params(
            "cancel_tombstoned.victim_admitted",
            {"actual": output("victim_settled", "admitted")},
        ),
    )
    case.step(
        "first_output",
        "preemption_ts_first_output",
        timeout_s=case.value("cancel_tombstoned.first_output_timeout_s"),
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "crash_fault",
        "engine_inject",
        params=case.value("cancel_tombstoned.crash_fault"),
    )
    case.step("restore_guard", "preemption_ts_restore_guard")
    case.step(
        "crash_trigger",
        "preemption_ts_trigger",
        timeout_s=case.value("cancel_tombstoned.crash_trigger_timeout_s"),
    )
    case.step(
        "prefill_dropped",
        "preemption_ts_health",
        timeout_s=case.value("cancel_tombstoned.prefill_dropped_timeout_s"),
        params=case.value("cancel_tombstoned.prefill_dropped"),
    )
    case.step(
        "prefill_restart",
        "engine_control",
        params=case.value("cancel_tombstoned.prefill_restart"),
    )
    case.step(
        "prefill_restored",
        "preemption_ts_health",
        timeout_s=case.value("cancel_tombstoned.prefill_restored_timeout_s"),
        params=case.value("cancel_tombstoned.prefill_restored"),
    )
    case.step(
        "reconnect", "balance_pause", params=case.value("cancel_tombstoned.reconnect")
    )
    case.step(
        "victim_cut",
        "preemption_ts_cut",
        timeout_s=case.value("cancel_tombstoned.victim_cut_timeout_s"),
        params={"requests": output("victim", "requests")},
    )
    case.step("baseline_cancel", "preemption_cancel_census")
    case.step(
        "incoming",
        "preemption_live_start",
        timeout_s=case.value("cancel_tombstoned.incoming_timeout_s"),
        params=case.value("cancel_tombstoned.incoming"),
    )
    case.step(
        "incoming_settled",
        "preemption_settled",
        timeout_s=case.value("cancel_tombstoned.incoming_settled_timeout_s"),
        params={"requests": output("incoming", "requests")},
    )
    case.step(
        "incoming_drain",
        "preemption_live_drain",
        timeout_s=case.value("cancel_tombstoned.incoming_drain_timeout_s"),
        params=case.params(
            "cancel_tombstoned.incoming_drain",
            {"requests": output("incoming", "requests")},
        ),
    )
    case.step(
        "incoming_completed",
        "preemption_ts_incoming",
        params={"requests": output("incoming", "requests")},
    )
    case.step(
        "cancel_arrival",
        "preemption_ts_cancel",
        timeout_s=case.value("cancel_tombstoned.cancel_arrival_timeout_s"),
        params={"before": output("baseline_cancel", "snapshot")},
    )
    case.step(
        "fence_probe",
        "preemption_ts_fence",
        timeout_s=case.value("cancel_tombstoned.fence_probe_timeout_s"),
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "engine_clean",
        "preemption_ts_engine_clean",
        timeout_s=case.value("cancel_tombstoned.engine_clean_timeout_s"),
    )
    case.step(
        "residue",
        "preemption_ts_residue",
        timeout_s=case.value("cancel_tombstoned.residue_timeout_s"),
    )
    case.step(
        "recovery_prepare",
        "recovery_prepare",
        params=case.value("cancel_tombstoned.recovery_prepare"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("cancel_tombstoned.recovery_dispatch_timeout_s"),
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
