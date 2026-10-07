"""Engine generation bump with transport retirement and request recovery evidence."""

from cases.config import output


def generation_bump(case):
    case.step("setup", "setup", timeout_s=case.value("generation_bump.setup_timeout_s"))
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("generation_bump.prior_drain_observed_timeout_s"),
        params=case.value("generation_bump.prior_drain_observed"),
    )
    case.step("target", "recovery_select", params=case.value("generation_bump.target"))
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=case.value("generation_bump.generation_before_timeout_s"),
        params=case.params(
            "generation_bump.generation_before",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "baseline",
        "recovery_prepare",
        params=case.value("generation_bump.baseline"),
    )
    case.step(
        "baseline_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("generation_bump.baseline_dispatch_timeout_s"),
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_state",
        "recovery_observe",
        timeout_s=case.value("generation_bump.baseline_state_timeout_s"),
        params=case.params(
            "generation_bump.baseline_state",
            {"requests": output("baseline", "requests")},
        ),
    )
    case.step(
        "baseline_succeeds",
        "recovery_check",
        params=case.params(
            "generation_bump.baseline_succeeds",
            {"snapshot": output("baseline_state", "snapshot")},
        ),
    )
    case.step(
        "outage",
        "engine_control",
        params=case.value("generation_bump.outage"),
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=case.value("generation_bump.retirement_timeout_s"),
        params=case.params(
            "generation_bump.retirement", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "transport_retired",
        "recovery_check",
        params=case.params(
            "generation_bump.transport_retired",
            {"snapshot": output("retirement", "snapshot")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.value("generation_bump.restart"),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("generation_bump.alive_restored_timeout_s"),
        params=case.value("generation_bump.alive_restored"),
    )
    case.step(
        "alive_back",
        "recovery_check",
        params=case.params(
            "generation_bump.alive_back",
            {"snapshot": output("alive_restored", "snapshot")},
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("generation_bump.reconnect_timeout_s"),
        params=case.value("generation_bump.reconnect"),
    )
    case.step(
        "recovered_generation",
        "recovery_observe",
        timeout_s=case.value("generation_bump.recovered_generation_timeout_s"),
        params=case.params(
            "generation_bump.recovered_generation",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "generation_is_new",
        "recovery_check",
        params=case.params(
            "generation_bump.generation_is_new",
            {
                "snapshot": output("recovered_generation", "snapshot"),
                "baseline": output("generation_before", "snapshot"),
            },
        ),
    )
    case.step(
        "recovered_prefill_batch_ledger_zero",
        "recovery_check",
        params=case.params(
            "generation_bump.recovered_prefill_batch_ledger_zero",
            {"snapshot": output("recovered_generation", "snapshot")},
        ),
    )
    case.step(
        "recovered_prefill_member_ledger_zero",
        "recovery_check",
        params=case.params(
            "generation_bump.recovered_prefill_member_ledger_zero",
            {"snapshot": output("recovered_generation", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params=case.value("generation_bump.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("generation_bump.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=case.value("generation_bump.recovery_state_timeout_s"),
        params=case.params(
            "generation_bump.recovery_state",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params=case.params(
            "generation_bump.recovery_succeeds",
            {"snapshot": output("recovery_state", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")
