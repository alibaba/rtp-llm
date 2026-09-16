"""No preemption: after the victim reaches decode running, its acceptance permit is free and a higher-priority incomer is admitted alongside it."""

from ..case_config import output


def permit_released_without_preemption(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("permit_released_without_preemption.setup_timeout_s"),
    )
    case.step(
        "victim",
        "admission_fire",
        params=case.value("permit_released_without_preemption.victim"),
    )
    case.step(
        "victim_admitted",
        "admission_check",
        params=case.params(
            "permit_released_without_preemption.victim_admitted",
            {"rows": output("victim", "rows")},
        ),
    )
    case.step(
        "running",
        "admission_observe",
        timeout_s=case.value("permit_released_without_preemption.running_timeout_s"),
        params=case.value("permit_released_without_preemption.running"),
    )
    case.step(
        "victim_running",
        "admission_gauge_check",
        params=case.params(
            "permit_released_without_preemption.victim_running",
            {"snapshot": output("running", "snapshot")},
        ),
    )
    case.step(
        "incomer",
        "admission_fire",
        params=case.value("permit_released_without_preemption.incomer"),
    )
    case.step(
        "incomer_admitted",
        "admission_check",
        params=case.params(
            "permit_released_without_preemption.incomer_admitted",
            {"rows": output("incomer", "rows")},
        ),
    )
    case.step(
        "incomer_code",
        "admission_check",
        params=case.params(
            "permit_released_without_preemption.incomer_code",
            {"rows": output("incomer", "rows")},
        ),
    )
    case.step(
        "accepted_fast",
        "admission_check",
        params=case.params(
            "permit_released_without_preemption.accepted_fast",
            {"rows": output("incomer", "rows")},
        ),
    )
    case.step(
        "victim_done",
        "admission_wait",
        timeout_s=case.value(
            "permit_released_without_preemption.victim_done_timeout_s"
        ),
        params={"wave": output("victim", "wave")},
    )
    case.step(
        "victim_unmolested",
        "admission_check",
        params=case.params(
            "permit_released_without_preemption.victim_unmolested",
            {"rows": output("victim_done", "rows")},
        ),
    )
    case.step(
        "incomer_done",
        "admission_wait",
        timeout_s=case.value(
            "permit_released_without_preemption.incomer_done_timeout_s"
        ),
        params={"wave": output("incomer", "wave")},
    )
    case.step(
        "incomer_completed",
        "admission_check",
        params=case.params(
            "permit_released_without_preemption.incomer_completed",
            {"rows": output("incomer_done", "rows")},
        ),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("permit_released_without_preemption.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value(
            "permit_released_without_preemption.recovery_done_timeout_s"
        ),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "permit_released_without_preemption.recovered",
            {"rows": output("recovery_done", "rows")},
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value(
            "permit_released_without_preemption.master_clean_timeout_s"
        ),
        params=case.value("permit_released_without_preemption.master_clean"),
    )
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=case.value(
            "permit_released_without_preemption.engine_clean_timeout_s"
        ),
        params=case.value("permit_released_without_preemption.engine_clean"),
    )
    case.step("cleanup", "teardown")
