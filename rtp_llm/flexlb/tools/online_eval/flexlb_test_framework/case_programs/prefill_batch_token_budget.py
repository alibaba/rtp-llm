"""One master batch, explicit startup engine token budget and observed Master batch/member settlement; construction shape is diagnostic."""

from ..case_config import output


def split(case):
    case.step("setup", "setup", timeout_s=case.value("split.setup_timeout_s"))
    case.step("before", "admission_budget_snapshot", params=case.value("split.before"))
    case.step(
        "wave",
        "admission_burst",
        timeout_s=case.value("split.wave_timeout_s"),
        params=case.value("split.wave"),
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=case.value("split.park_timeout_s"),
        params=case.value("split.park"),
    )
    case.step(
        "park_contract",
        "admission_gauge_check",
        params=case.params(
            "split.park_contract", {"snapshot": output("park", "snapshot")}
        ),
    )
    case.step(
        "ledger",
        "admission_ledger_observe",
        timeout_s=case.value("split.ledger_timeout_s"),
        params=case.value("split.ledger"),
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params=case.params("split.linkage", {"samples": output("ledger", "samples")}),
    )
    case.step(
        "done",
        "admission_wait",
        timeout_s=case.value("split.done_timeout_s"),
        params={"wave": output("wave", "wave")},
    )
    case.step(
        "all_completed",
        "admission_check",
        params=case.params("split.all_completed", {"rows": output("done", "rows")}),
    )
    case.step("after", "admission_budget_snapshot", params=case.value("split.after"))
    case.step(
        "shape",
        "admission_shape_observe",
        params=case.params(
            "split.shape",
            {
                "before": output("before", "snapshot"),
                "after": output("after", "snapshot"),
            },
        ),
    )
    case.step(
        "batch_identity",
        "admission_budget_check",
        params=case.params(
            "split.batch_identity",
            {"rows": output("done", "rows"), "snapshot": output("after", "snapshot")},
        ),
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("split.empty_timeout_s"),
        params=case.value("split.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "split.park_empty", {"snapshot": output("empty", "snapshot")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("split.master_clean_timeout_s"),
        params=case.value("split.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("split.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("split.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "split.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step("cleanup", "teardown")


def split_fifo(case):
    case.step("setup", "setup", timeout_s=case.value("split_fifo.setup_timeout_s"))
    case.step(
        "before", "admission_budget_snapshot", params=case.value("split_fifo.before")
    )
    case.step(
        "wave",
        "admission_burst",
        timeout_s=case.value("split_fifo.wave_timeout_s"),
        params=case.value("split_fifo.wave"),
    )
    case.step(
        "consumers",
        "admission_drain_start",
        timeout_s=case.value("split_fifo.consumers_timeout_s"),
        params={"waves": [output("wave", "wave")]},
    )
    case.step(
        "ledger",
        "admission_ledger_observe",
        timeout_s=case.value("split_fifo.ledger_timeout_s"),
        params=case.value("split_fifo.ledger"),
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params=case.params(
            "split_fifo.linkage", {"samples": output("ledger", "samples")}
        ),
    )
    case.step(
        "done",
        "admission_drain_collect",
        timeout_s=case.value("split_fifo.done_timeout_s"),
        params={"drain": output("consumers", "drain")},
    )
    case.step(
        "all_completed",
        "admission_check",
        params=case.params(
            "split_fifo.all_completed", {"rows": output("done", "rows")}
        ),
    )
    case.step(
        "client_fifo",
        "admission_budget_check",
        params=case.params("split_fifo.client_fifo", {"rows": output("done", "rows")}),
    )
    case.step(
        "after", "admission_budget_snapshot", params=case.value("split_fifo.after")
    )
    case.step(
        "shape",
        "admission_shape_observe",
        params=case.params(
            "split_fifo.shape",
            {
                "before": output("before", "snapshot"),
                "after": output("after", "snapshot"),
            },
        ),
    )
    case.step(
        "engine_fifo_shape",
        "admission_engine_clusters",
        params={
            "rows": output("done", "rows"),
            "snapshot": output("after", "snapshot"),
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("split_fifo.empty_timeout_s"),
        params=case.value("split_fifo.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "split_fifo.park_empty", {"snapshot": output("empty", "snapshot")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("split_fifo.master_clean_timeout_s"),
        params=case.value("split_fifo.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("split_fifo.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("split_fifo.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "split_fifo.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step("cleanup", "teardown")


def boundary(case):
    case.step("setup", "setup", timeout_s=case.value("boundary.setup_timeout_s"))
    case.step(
        "baseline",
        "admission_wave",
        params=case.value("boundary.baseline"),
    )
    case.step(
        "baseline_done",
        "admission_wait",
        timeout_s=case.value("boundary.baseline_done_timeout_s"),
        params={"wave": output("baseline", "wave")},
    )
    case.step(
        "baseline_success",
        "admission_check",
        params=case.params(
            "boundary.baseline_success", {"rows": output("baseline_done", "rows")}
        ),
    )
    case.step(
        "before", "admission_budget_snapshot", params=case.value("boundary.before")
    )
    case.step(
        "wave",
        "admission_burst",
        timeout_s=case.value("boundary.wave_timeout_s"),
        params=case.value("boundary.wave"),
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=case.value("boundary.park_timeout_s"),
        params=case.value("boundary.park"),
    )
    case.step(
        "park_contract",
        "admission_gauge_check",
        params=case.params(
            "boundary.park_contract", {"snapshot": output("park", "snapshot")}
        ),
    )
    case.step(
        "ledger",
        "admission_ledger_observe",
        timeout_s=case.value("boundary.ledger_timeout_s"),
        params=case.value("boundary.ledger"),
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params=case.params(
            "boundary.linkage", {"samples": output("ledger", "samples")}
        ),
    )
    case.step(
        "done",
        "admission_wait",
        timeout_s=case.value("boundary.done_timeout_s"),
        params={"wave": output("wave", "wave")},
    )
    case.step(
        "all_completed",
        "admission_check",
        params=case.params("boundary.all_completed", {"rows": output("done", "rows")}),
    )
    case.step("after", "admission_budget_snapshot", params=case.value("boundary.after"))
    case.step(
        "shape",
        "admission_shape_observe",
        params=case.params(
            "boundary.shape",
            {
                "before": output("before", "snapshot"),
                "after": output("after", "snapshot"),
            },
        ),
    )
    case.step(
        "ttft_neutral",
        "admission_budget_check",
        params=case.params(
            "boundary.ttft_neutral",
            {
                "rows": output("done", "rows"),
                "baseline_rows": output("baseline_done", "rows"),
            },
        ),
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("boundary.empty_timeout_s"),
        params=case.value("boundary.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "boundary.park_empty", {"snapshot": output("empty", "snapshot")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("boundary.master_clean_timeout_s"),
        params=case.value("boundary.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("boundary.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("boundary.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "boundary.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step("cleanup", "teardown")


def regroup_disabled(case):
    case.step(
        "setup", "setup", timeout_s=case.value("regroup_disabled.setup_timeout_s")
    )
    case.step(
        "before",
        "admission_budget_snapshot",
        params=case.value("regroup_disabled.before"),
    )
    case.step(
        "wave",
        "admission_burst",
        timeout_s=case.value("regroup_disabled.wave_timeout_s"),
        params=case.value("regroup_disabled.wave"),
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=case.value("regroup_disabled.park_timeout_s"),
        params=case.value("regroup_disabled.park"),
    )
    case.step(
        "park_contract",
        "admission_gauge_check",
        params=case.params(
            "regroup_disabled.park_contract", {"snapshot": output("park", "snapshot")}
        ),
    )
    case.step(
        "ledger",
        "admission_ledger_observe",
        timeout_s=case.value("regroup_disabled.ledger_timeout_s"),
        params=case.value("regroup_disabled.ledger"),
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params=case.params(
            "regroup_disabled.linkage", {"samples": output("ledger", "samples")}
        ),
    )
    case.step(
        "done",
        "admission_wait",
        timeout_s=case.value("regroup_disabled.done_timeout_s"),
        params={"wave": output("wave", "wave")},
    )
    case.step(
        "all_completed",
        "admission_check",
        params=case.params(
            "regroup_disabled.all_completed", {"rows": output("done", "rows")}
        ),
    )
    case.step(
        "after",
        "admission_budget_snapshot",
        params=case.value("regroup_disabled.after"),
    )
    case.step(
        "shape",
        "admission_shape_observe",
        params=case.params(
            "regroup_disabled.shape",
            {
                "before": output("before", "snapshot"),
                "after": output("after", "snapshot"),
            },
        ),
    )
    case.step(
        "batch_identity",
        "admission_budget_check",
        params=case.params(
            "regroup_disabled.batch_identity",
            {"rows": output("done", "rows"), "snapshot": output("after", "snapshot")},
        ),
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=case.value("regroup_disabled.empty_timeout_s"),
        params=case.value("regroup_disabled.empty"),
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params=case.params(
            "regroup_disabled.park_empty", {"snapshot": output("empty", "snapshot")}
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("regroup_disabled.master_clean_timeout_s"),
        params=case.value("regroup_disabled.master_clean"),
    )
    case.step(
        "recovery",
        "admission_wave",
        params=case.value("regroup_disabled.recovery"),
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=case.value("regroup_disabled.recovery_done_timeout_s"),
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params=case.params(
            "regroup_disabled.recovered", {"rows": output("recovery_done", "rows")}
        ),
    )
    case.step("cleanup", "teardown")
