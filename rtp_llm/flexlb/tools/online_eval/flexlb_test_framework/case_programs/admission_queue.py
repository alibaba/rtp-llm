"""Distinct queue-depth, KV wait/deadline, and global outstanding-capacity admission contracts."""

from ..case_config import output


def queue_depth(case):
    case.step("setup", "setup", timeout_s=case.value("queue_depth.setup_timeout_s"))
    case.step(
        "slow",
        "engine_control",
        params=case.value("queue_depth.slow"),
    )
    case.step(
        "depth_gate",
        "engine_inject",
        params=case.value("queue_depth.depth_gate"),
    )
    case.step(
        "occupants",
        "admission_occupy",
        timeout_s=case.value("queue_depth.occupants_timeout_s"),
        params=case.value("queue_depth.occupants"),
    )
    case.step(
        "probe",
        "admission_wave",
        params=case.value("queue_depth.probe"),
    )
    case.step(
        "probe_done",
        "admission_wait",
        timeout_s=case.value("queue_depth.probe_done_timeout_s"),
        params={"wave": output("probe", "wave")},
    )
    case.step(
        "depth_error",
        "admission_check",
        params=case.params(
            "queue_depth.depth_error", {"rows": output("probe_done", "rows")}
        ),
    )
    case.step(
        "fast_reject",
        "admission_check",
        params=case.params(
            "queue_depth.fast_reject", {"rows": output("probe_done", "rows")}
        ),
    )
    case.step(
        "clear_depth", "engine_clear", params={"fault": output("depth_gate", "fault")}
    )
    case.step(
        "occupants_done",
        "admission_wait",
        timeout_s=case.value("queue_depth.occupants_done_timeout_s"),
        params={"wave": output("occupants", "wave")},
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("queue_depth.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("queue_depth.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "queue_depth.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("queue_depth.master_clean_timeout_s"),
        params=case.value("queue_depth.master_clean"),
    )
    case.step(
        "engine_clean",
        "master_direct_clean",
        timeout_s=case.value("queue_depth.engine_clean_timeout_s"),
    )
    case.step(
        "normal_perf",
        "engine_control",
        params=case.value("queue_depth.normal_perf"),
    )
    case.step("cleanup", "teardown")


def slo_deadline(case):
    case.step("setup", "setup", timeout_s=case.value("slo_deadline.setup_timeout_s"))
    case.step(
        "kv_squeeze",
        "engine_inject",
        params=case.value("slo_deadline.kv_squeeze"),
    )
    case.step(
        "poll_squeeze", "master_mark", params=case.value("slo_deadline.poll_squeeze")
    )
    case.step(
        "probe",
        "admission_wave",
        params=case.value("slo_deadline.probe"),
    )
    case.step(
        "probe_done",
        "admission_wait",
        timeout_s=case.value("slo_deadline.probe_done_timeout_s"),
        params={"wave": output("probe", "wave")},
    )
    case.step(
        "deadline_error_family",
        "admission_check",
        params=case.params(
            "slo_deadline.deadline_error_family", {"rows": output("probe_done", "rows")}
        ),
    )
    case.step(
        "waited",
        "admission_check",
        params=case.params(
            "slo_deadline.waited", {"rows": output("probe_done", "rows")}
        ),
    )
    case.step(
        "bounded_deadline",
        "admission_check",
        params=case.params(
            "slo_deadline.bounded_deadline", {"rows": output("probe_done", "rows")}
        ),
    )
    case.step(
        "clear_pressure",
        "engine_clear",
        params={"fault": output("kv_squeeze", "fault")},
    )
    case.step(
        "poll_recovery", "master_mark", params=case.value("slo_deadline.poll_recovery")
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("slo_deadline.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("slo_deadline.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "slo_deadline.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("slo_deadline.master_clean_timeout_s"),
        params=case.value("slo_deadline.master_clean"),
    )
    case.step("cleanup", "teardown")
