"""Separate prefill concurrency, decode hard gate, prefill waiting cap and KV pool capacity programs."""

from ..case_config import output


def prefill_concurrency(case):
    case.step(
        "setup", "setup", timeout_s=case.value("prefill_concurrency.setup_timeout_s")
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("prefill_concurrency.slow"),
    )
    case.step(
        "fired",
        "admission_fire",
        timeout_s=case.value("prefill_concurrency.fired_timeout_s"),
        params=case.value("prefill_concurrency.fired"),
    )
    case.step(
        "all_admitted",
        "admission_check",
        params=case.params(
            "prefill_concurrency.all_admitted", {"rows": output("fired", "rows")}
        ),
    )
    case.step(
        "drained",
        "admission_wait",
        timeout_s=case.value("prefill_concurrency.drained_timeout_s"),
        params={"wave": output("fired", "wave")},
    )
    case.step(
        "all_completed",
        "admission_check",
        params=case.params(
            "prefill_concurrency.all_completed", {"rows": output("drained", "rows")}
        ),
    )
    case.step(
        "execution_capacity",
        "admission_execution_capacity",
        params=case.params(
            "prefill_concurrency.execution_capacity",
            {"rows": output("drained", "rows")},
        ),
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("prefill_concurrency.empty_timeout_s"),
        params=case.value("prefill_concurrency.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "prefill_concurrency.park_empty", {"snapshot": output("empty", "snapshot")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("prefill_concurrency.master_clean_timeout_s"),
        params=case.value("prefill_concurrency.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("prefill_concurrency.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("prefill_concurrency.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "prefill_concurrency.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step(
        "normal_perf",
        "engine_control",
        params=case.value("prefill_concurrency.normal_perf"),
    )
    case.step("cleanup", "teardown")


def decode_hard_gate(case):
    case.step(
        "setup", "setup", timeout_s=case.value("decode_hard_gate.setup_timeout_s")
    )
    case.step(
        "slow_decode",
        "engine_control",
        params=case.value("decode_hard_gate.slow_decode"),
    )
    case.step(
        "fired",
        "admission_fire",
        timeout_s=case.value("decode_hard_gate.fired_timeout_s"),
        params=case.value("decode_hard_gate.fired"),
    )
    case.step(
        "all_admitted",
        "admission_check",
        params=case.params(
            "decode_hard_gate.all_admitted", {"rows": output("fired", "rows")}
        ),
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=case.value("decode_hard_gate.park_timeout_s"),
        params=case.value("decode_hard_gate.park"),
    )
    case.step(
        "conditional_park",
        "admission_decode_park_check",
        params=case.params(
            "decode_hard_gate.conditional_park",
            {"snapshot": output("park", "snapshot")},
        ),
    )
    case.step(
        "drained",
        "admission_wait",
        timeout_s=case.value("decode_hard_gate.drained_timeout_s"),
        params={"wave": output("fired", "wave")},
    )
    case.step(
        "completed_95pct",
        "admission_check",
        params=case.params(
            "decode_hard_gate.completed_95pct", {"rows": output("drained", "rows")}
        ),
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("decode_hard_gate.empty_timeout_s"),
        params=case.value("decode_hard_gate.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "decode_hard_gate.park_empty", {"snapshot": output("empty", "snapshot")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("decode_hard_gate.master_clean_timeout_s"),
        params=case.value("decode_hard_gate.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("decode_hard_gate.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("decode_hard_gate.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "decode_hard_gate.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step(
        "normal_perf",
        "engine_control",
        params=case.value("decode_hard_gate.normal_perf"),
    )
    case.step("cleanup", "teardown")


def prefill_waiting_cap(case):
    case.step(
        "setup", "setup", timeout_s=case.value("prefill_waiting_cap.setup_timeout_s")
    )
    case.step(
        "limited",
        "engine_control",
        params=case.value("prefill_waiting_cap.limited"),
    )
    case.step(
        "occupants",
        "admission_fire",
        timeout_s=case.value("prefill_waiting_cap.occupants_timeout_s"),
        params=case.value("prefill_waiting_cap.occupants"),
    )
    case.step(
        "occupants_admitted",
        "admission_check",
        params=case.params(
            "prefill_waiting_cap.occupants_admitted",
            {"rows": output("occupants", "rows")},
        ),
    )
    case.step(
        "saturated",
        "admission_observe",
        timeout_s=case.value("prefill_waiting_cap.saturated_timeout_s"),
        params=case.value("prefill_waiting_cap.saturated"),
    )
    case.step(
        "cap_seen",
        "admission_gauge_check",
        params=case.params(
            "prefill_waiting_cap.cap_seen",
            {"snapshot": output("saturated", "snapshot")},
        ),
    )
    case.step(
        "probe",
        "admission_engine_probe",
        timeout_s=case.value("prefill_waiting_cap.probe_done_timeout_s"),
        params=case.value("prefill_waiting_cap.probe"),
    )
    case.step(
        "backpressure_error",
        "admission_check",
        params=case.params(
            "prefill_waiting_cap.backpressure_error",
            {"rows": output("probe", "rows")},
        ),
    )
    case.step(
        "fast_reject",
        "admission_check",
        params=case.params(
            "prefill_waiting_cap.fast_reject", {"rows": output("probe", "rows")}
        ),
    )
    case.step(
        "unbounded",
        "engine_control",
        params=case.value("prefill_waiting_cap.unbounded"),
    )
    case.step(
        "before_fourth",
        "admission_observe",
        timeout_s=case.value("prefill_waiting_cap.before_fourth_timeout_s"),
        params=case.value("prefill_waiting_cap.before_fourth"),
    )
    case.step(
        "fourth",
        "admission_fire",
        timeout_s=case.value("prefill_waiting_cap.fourth_timeout_s"),
        params=case.value("prefill_waiting_cap.fourth"),
    )
    case.step(
        "fourth_admitted",
        "admission_check",
        params=case.params(
            "prefill_waiting_cap.fourth_admitted", {"rows": output("fourth", "rows")}
        ),
    )
    case.step(
        "occupants_done",
        "admission_wait",
        timeout_s=case.value("prefill_waiting_cap.occupants_done_timeout_s"),
        params={"wave": output("occupants", "wave")},
    )
    case.step(
        "occupants_complete",
        "admission_check",
        params=case.params(
            "prefill_waiting_cap.occupants_complete",
            {"rows": output("occupants_done", "rows")},
        ),
    )
    case.step(
        "fourth_done",
        "admission_wait",
        timeout_s=case.value("prefill_waiting_cap.fourth_done_timeout_s"),
        params={"wave": output("fourth", "wave")},
    )
    case.step(
        "pressure_recovery",
        "admission_check",
        params=case.params(
            "prefill_waiting_cap.pressure_recovery",
            {"rows": output("fourth_done", "rows")},
        ),
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("prefill_waiting_cap.empty_timeout_s"),
        params=case.value("prefill_waiting_cap.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "prefill_waiting_cap.park_empty", {"snapshot": output("empty", "snapshot")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("prefill_waiting_cap.master_clean_timeout_s"),
        params=case.value("prefill_waiting_cap.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("prefill_waiting_cap.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("prefill_waiting_cap.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "prefill_waiting_cap.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step(
        "normal_perf",
        "engine_control",
        params=case.value("prefill_waiting_cap.normal_perf"),
    )
    case.step("cleanup", "teardown")


def kv_pool_capacity(case):
    case.step(
        "setup", "setup", timeout_s=case.value("kv_pool_capacity.setup_timeout_s")
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("kv_pool_capacity.slow"),
    )
    case.step(
        "occupants",
        "admission_fire",
        timeout_s=case.value("kv_pool_capacity.occupants_timeout_s"),
        params=case.value("kv_pool_capacity.occupants"),
    )
    case.step(
        "occupants_admitted",
        "admission_check",
        params=case.params(
            "kv_pool_capacity.occupants_admitted", {"rows": output("occupants", "rows")}
        ),
    )
    case.step(
        "saturated",
        "admission_observe",
        timeout_s=case.value("kv_pool_capacity.saturated_timeout_s"),
        params=case.value("kv_pool_capacity.saturated"),
    )
    case.step(
        "pool_full",
        "admission_gauge_check",
        params=case.params(
            "kv_pool_capacity.pool_full", {"snapshot": output("saturated", "snapshot")}
        ),
    )
    case.step(
        "probe",
        "admission_engine_probe",
        timeout_s=case.value("kv_pool_capacity.probe_done_timeout_s"),
        params=case.value("kv_pool_capacity.probe"),
    )
    case.step(
        "lack_mem_error",
        "admission_check",
        params=case.params(
            "kv_pool_capacity.lack_mem_error", {"rows": output("probe", "rows")}
        ),
    )
    case.step(
        "fast_reject",
        "admission_check",
        params=case.params(
            "kv_pool_capacity.fast_reject", {"rows": output("probe", "rows")}
        ),
    )
    case.step(
        "occupants_done",
        "admission_wait",
        timeout_s=case.value("kv_pool_capacity.occupants_done_timeout_s"),
        params={"wave": output("occupants", "wave")},
    )
    case.step(
        "occupants_complete",
        "admission_check",
        params=case.params(
            "kv_pool_capacity.occupants_complete",
            {"rows": output("occupants_done", "rows")},
        ),
    )
    case.step(
        "available",
        "admission_observe",
        timeout_s=case.value("kv_pool_capacity.available_timeout_s"),
        params=case.value("kv_pool_capacity.available"),
    )
    case.step(
        "pool_recovered",
        "admission_gauge_check",
        params=case.params(
            "kv_pool_capacity.pool_recovered",
            {"snapshot": output("available", "snapshot")},
        ),
    )
    case.step(
        "fresh_lease",
        "admission_fire",
        timeout_s=case.value("kv_pool_capacity.fresh_lease_timeout_s"),
        params=case.value("kv_pool_capacity.fresh_lease"),
    )
    case.step(
        "fresh_done",
        "admission_wait",
        timeout_s=case.value("kv_pool_capacity.fresh_done_timeout_s"),
        params={"wave": output("fresh_lease", "wave")},
    )
    case.step(
        "fresh_succeeded",
        "admission_check",
        params=case.params(
            "kv_pool_capacity.fresh_succeeded", {"rows": output("fresh_done", "rows")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("kv_pool_capacity.master_clean_timeout_s"),
        params=case.value("kv_pool_capacity.master_clean"),
    )
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=case.value("kv_pool_capacity.engine_clean_timeout_s"),
        params=case.value("kv_pool_capacity.engine_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("kv_pool_capacity.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("kv_pool_capacity.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "kv_pool_capacity.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step(
        "normal_perf",
        "engine_control",
        params=case.value("kv_pool_capacity.normal_perf"),
    )
    case.step("cleanup", "teardown")
