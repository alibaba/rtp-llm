"""Master queue and placement backlog, sampled during Schedule; explicit old deadline, FIFO wait-return and recovery contracts."""

from ..case_config import output


def batcher_queue_capacity_park(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("batcher_queue_capacity_park.setup_timeout_s"),
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("batcher_queue_capacity_park.slow"),
    )
    case.step(
        "wave",
        "admission_tracked_fire",
        timeout_s=case.value("batcher_queue_capacity_park.wave_timeout_s"),
        params=case.value("batcher_queue_capacity_park.wave"),
    )
    case.step(
        "all_admitted",
        "admission_check",
        params=case.params(
            "batcher_queue_capacity_park.all_admitted", {"rows": output("wave", "rows")}
        ),
    )
    case.step(
        "done",
        "admission_drain",
        timeout_s=case.value("batcher_queue_capacity_park.done_timeout_s"),
        params={"waves": [output("wave", "wave")]},
    )
    case.step(
        "all_completed",
        "admission_check",
        params=case.params(
            "batcher_queue_capacity_park.all_completed",
            {"rows": output("done", "rows")},
        ),
    )
    case.step(
        "fifo",
        "admission_check",
        params=case.params(
            "batcher_queue_capacity_park.fifo", {"rows": output("done", "rows")}
        ),
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("batcher_queue_capacity_park.empty_timeout_s"),
        params=case.value("batcher_queue_capacity_park.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "batcher_queue_capacity_park.park_empty",
            {"snapshot": output("empty", "snapshot")},
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("batcher_queue_capacity_park.master_clean_timeout_s"),
        params=case.value("batcher_queue_capacity_park.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("batcher_queue_capacity_park.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("batcher_queue_capacity_park.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "batcher_queue_capacity_park.recovered",
            {"rows": output("recovery_done", "rows")},
        ),
    )
    case.step(
        "relieve",
        "engine_control",
        params=case.value("batcher_queue_capacity_park.relieve"),
    )
    case.step("cleanup", "teardown")


def batcher_queue_deadline(case):
    case.step(
        "setup", "setup", timeout_s=case.value("batcher_queue_deadline.setup_timeout_s")
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("batcher_queue_deadline.slow"),
    )
    case.step(
        "wave",
        "admission_tracked_fire",
        timeout_s=case.value("batcher_queue_deadline.wave_timeout_s"),
        params=case.value("batcher_queue_deadline.wave"),
    )
    case.step(
        "six_admitted",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.six_admitted", {"rows": output("wave", "rows")}
        ),
    )
    case.step(
        "two_rejected",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.two_rejected", {"rows": output("wave", "rows")}
        ),
    )
    case.step(
        "deadline_family",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.deadline_family", {"rows": output("wave", "rows")}
        ),
    )
    case.step(
        "deadline_min",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.deadline_min", {"rows": output("wave", "rows")}
        ),
    )
    case.step(
        "deadline_max",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.deadline_max", {"rows": output("wave", "rows")}
        ),
    )
    case.step(
        "done",
        "admission_drain",
        timeout_s=case.value("batcher_queue_deadline.done_timeout_s"),
        params={"waves": [output("wave", "wave")]},
    )
    case.step(
        "six_completed",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.six_completed", {"rows": output("done", "rows")}
        ),
    )
    case.step(
        "no_serving_errors",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.no_serving_errors", {"rows": output("done", "rows")}
        ),
    )
    case.step(
        "relieve",
        "engine_control",
        params=case.value("batcher_queue_deadline.relieve"),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("batcher_queue_deadline.master_clean_timeout_s"),
        params=case.value("batcher_queue_deadline.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("batcher_queue_deadline.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("batcher_queue_deadline.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "batcher_queue_deadline.recovered",
            {"rows": output("recovery_done", "rows")},
        ),
    )
    case.step("cleanup", "teardown")


def placement_pool_wait(case):
    case.step(
        "setup", "setup", timeout_s=case.value("placement_pool_wait.setup_timeout_s")
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("placement_pool_wait.slow"),
    )
    case.step(
        "a",
        "admission_tracked_fire",
        timeout_s=case.value("placement_pool_wait.a_timeout_s"),
        params=case.value("placement_pool_wait.a"),
    )
    case.step(
        "a_admitted",
        "admission_check",
        params=case.params(
            "placement_pool_wait.a_admitted", {"rows": output("a", "rows")}
        ),
    )
    case.step(
        "running",
        "admission_observe",
        timeout_s=case.value("placement_pool_wait.running_timeout_s"),
        params=case.value("placement_pool_wait.running"),
    )
    case.step(
        "a_running",
        "admission_gauge_check",
        params=case.params(
            "placement_pool_wait.a_running", {"snapshot": output("running", "snapshot")}
        ),
    )
    case.step(
        "lease_before_b",
        "admission_lease_precondition",
        timeout_s=case.value("placement_pool_wait.lease_before_b_timeout_s"),
        params=case.value("placement_pool_wait.lease_before_b"),
    )
    case.step(
        "b",
        "admission_tracked_fire",
        timeout_s=case.value("placement_pool_wait.b_timeout_s"),
        params=case.value("placement_pool_wait.b"),
    )
    case.step(
        "b_admitted",
        "admission_check",
        params=case.params(
            "placement_pool_wait.b_admitted", {"rows": output("b", "rows")}
        ),
    )
    case.step(
        "b_rpc_parked",
        "admission_check",
        params=case.params(
            "placement_pool_wait.b_rpc_parked", {"rows": output("b", "rows")}
        ),
    )
    case.step(
        "done",
        "admission_drain",
        timeout_s=case.value("placement_pool_wait.done_timeout_s"),
        params={"waves": [output("a", "wave"), output("b", "wave")]},
    )
    case.step(
        "both_completed",
        "admission_check",
        params=case.params(
            "placement_pool_wait.both_completed", {"rows": output("done", "rows")}
        ),
    )
    case.step(
        "b_after_a",
        "admission_check",
        params=case.params(
            "placement_pool_wait.b_after_a", {"rows": output("done", "rows")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("placement_pool_wait.master_clean_timeout_s"),
        params=case.value("placement_pool_wait.master_clean"),
    )
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=case.value("placement_pool_wait.engine_clean_timeout_s"),
        params=case.value("placement_pool_wait.engine_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("placement_pool_wait.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("placement_pool_wait.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "placement_pool_wait.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step(
        "relieve",
        "engine_control",
        params=case.value("placement_pool_wait.relieve"),
    )
    case.step("cleanup", "teardown")
