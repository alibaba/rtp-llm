"""One master batch, explicit startup engine token budget and observed Master batch/member settlement; construction shape is diagnostic."""

from ..case_config import output

METADATA = {
    "id": "prefill_batch_token_budget",
    "description": "One master batch, explicit startup engine token budget and observed Master "
    "batch/member settlement; construction shape is diagnostic.",
    "category": "admission",
    "requires": ["enqueue_batch"],
}

PROFILES = ["batch-window"]


def split(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("before", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "wave",
        "admission_burst",
        timeout_s=60,
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 512,
            "output_len": 2,
            "consume": "deferred",
            "spacing_s": 0.01,
            "request_timeout_s": 45,
            "await_submissions": True,
        },
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["waiting", "prefill_waiting_batches"],
            "duration_s": 10,
            "until_op": "ge",
            "until_value": 1,
            "reduce": "any",
        },
    )
    case.step(
        "park_contract",
        "admission_gauge_check",
        params={
            "snapshot": output("park", "snapshot"),
            "fields": ["waiting", "prefill_waiting_batches"],
            "stat": "max_seen",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "ledger", "admission_ledger_observe", timeout_s=15, params={"duration_s": 8}
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params={
            "samples": output("ledger", "samples"),
            "n_requests": 4,
            "expect_intermediate": True,
        },
    )
    case.step(
        "done", "admission_wait", timeout_s=180, params={"wave": output("wave", "wave")}
    )
    case.step(
        "all_completed",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "success_count",
            "expected": 4,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("after", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "shape",
        "admission_shape_observe",
        params={
            "before": output("before", "snapshot"),
            "after": output("after", "snapshot"),
            "expected": [2, 4, 2],
        },
    )
    case.step(
        "batch_identity",
        "admission_budget_check",
        params={
            "rows": output("done", "rows"),
            "snapshot": output("after", "snapshot"),
            "metric": "batch_identity",
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["waiting", "prefill_waiting_batches"],
            "duration_s": 10,
            "until_op": "eq",
            "until_value": 0,
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["waiting", "prefill_waiting_batches"],
            "stat": "max_latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=40,
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params={
            "rows": output("recovery_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("cleanup", "teardown")


def split_fifo(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("before", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "wave",
        "admission_burst",
        timeout_s=60,
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 512,
            "output_len": 2,
            "consume": "deferred",
            "spacing_s": 0.01,
            "request_timeout_s": 45,
            "await_submissions": True,
        },
    )
    case.step(
        "consumers",
        "admission_drain_start",
        timeout_s=45,
        params={"waves": [output("wave", "wave")]},
    )
    case.step(
        "ledger", "admission_ledger_observe", timeout_s=15, params={"duration_s": 8}
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params={
            "samples": output("ledger", "samples"),
            "n_requests": 4,
            "expect_intermediate": True,
        },
    )
    case.step(
        "done",
        "admission_drain_collect",
        timeout_s=45,
        params={"drain": output("consumers", "drain")},
    )
    case.step(
        "all_completed",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "success_count",
            "expected": 4,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "client_fifo",
        "admission_budget_check",
        params={"rows": output("done", "rows"), "metric": "client_two_clusters"},
    )
    case.step("after", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "shape",
        "admission_shape_observe",
        params={
            "before": output("before", "snapshot"),
            "after": output("after", "snapshot"),
            "expected": [2, 4, 2],
        },
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
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["waiting", "prefill_waiting_batches"],
            "duration_s": 10,
            "until_op": "eq",
            "until_value": 0,
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["waiting", "prefill_waiting_batches"],
            "stat": "max_latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=40,
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params={
            "rows": output("recovery_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("cleanup", "teardown")


def boundary(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "baseline",
        "admission_wave",
        params={"count": 1, "input_len": 512, "output_len": 2, "request_timeout_s": 15},
    )
    case.step(
        "baseline_done",
        "admission_wait",
        timeout_s=45,
        params={"wave": output("baseline", "wave")},
    )
    case.step(
        "baseline_success",
        "admission_check",
        params={
            "rows": output("baseline_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("before", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "wave",
        "admission_burst",
        timeout_s=60,
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 512,
            "output_len": 2,
            "consume": "immediate",
            "spacing_s": 0.01,
            "request_timeout_s": 15,
            "await_submissions": False,
        },
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["prefill_waiting_batches"],
            "duration_s": 2.5,
        },
    )
    case.step(
        "park_contract",
        "admission_gauge_check",
        params={
            "snapshot": output("park", "snapshot"),
            "fields": ["prefill_waiting_batches"],
            "stat": "max_seen",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "ledger", "admission_ledger_observe", timeout_s=15, params={"duration_s": 5}
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params={
            "samples": output("ledger", "samples"),
            "n_requests": 4,
            "expect_intermediate": False,
        },
    )
    case.step(
        "done", "admission_wait", timeout_s=45, params={"wave": output("wave", "wave")}
    )
    case.step(
        "all_completed",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "success_count",
            "expected": 4,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("after", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "shape",
        "admission_shape_observe",
        params={
            "before": output("before", "snapshot"),
            "after": output("after", "snapshot"),
            "expected": [1, 4, 4],
        },
    )
    case.step(
        "ttft_neutral",
        "admission_budget_check",
        params={
            "rows": output("done", "rows"),
            "baseline_rows": output("baseline_done", "rows"),
            "metric": "ttft_degradation",
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["waiting", "prefill_waiting_batches"],
            "duration_s": 10,
            "until_op": "eq",
            "until_value": 0,
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["waiting", "prefill_waiting_batches"],
            "stat": "max_latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=40,
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params={
            "rows": output("recovery_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("cleanup", "teardown")


def regroup_disabled(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("before", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "wave",
        "admission_burst",
        timeout_s=60,
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 512,
            "output_len": 2,
            "consume": "deferred",
            "spacing_s": 0.01,
            "request_timeout_s": 45,
            "await_submissions": True,
        },
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["prefill_waiting_batches"],
            "duration_s": 2.5,
        },
    )
    case.step(
        "park_contract",
        "admission_gauge_check",
        params={
            "snapshot": output("park", "snapshot"),
            "fields": ["prefill_waiting_batches"],
            "stat": "max_seen",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "ledger", "admission_ledger_observe", timeout_s=15, params={"duration_s": 5}
    )
    case.step(
        "linkage",
        "admission_ledger_check",
        params={
            "samples": output("ledger", "samples"),
            "n_requests": 4,
            "expect_intermediate": False,
        },
    )
    case.step(
        "done", "admission_wait", timeout_s=180, params={"wave": output("wave", "wave")}
    )
    case.step(
        "all_completed",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "success_count",
            "expected": 4,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("after", "admission_budget_snapshot", params={"targets": ["prefill-0"]})
    case.step(
        "shape",
        "admission_shape_observe",
        params={
            "before": output("before", "snapshot"),
            "after": output("after", "snapshot"),
            "expected": [1, 4, 4],
        },
    )
    case.step(
        "batch_identity",
        "admission_budget_check",
        params={
            "rows": output("done", "rows"),
            "snapshot": output("after", "snapshot"),
            "metric": "batch_identity",
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["waiting", "prefill_waiting_batches"],
            "duration_s": 10,
            "until_op": "eq",
            "until_value": 0,
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["waiting", "prefill_waiting_batches"],
            "stat": "max_latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=40,
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params={
            "rows": output("recovery_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "split": {
        "build": split,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "split_fifo": {
        "build": split_fifo,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "boundary": {
        "build": boundary,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "regroup_disabled": {
        "build": regroup_disabled,
        "profiles": ["batch-window"],
        "metadata": {},
    },
}
