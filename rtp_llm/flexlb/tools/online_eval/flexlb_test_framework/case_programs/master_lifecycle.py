"""Cold restart of the owned master, clean scheduler state, topology and request recovery."""

from ..case_config import output

METADATA = {
    "id": "master_lifecycle",
    "description": "Cold restart of the owned master, clean scheduler state, topology and request "
    "recovery.",
    "category": "master",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def kill_single(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("baseline", "request", params={"count": 1, "output_len": 2})
    case.step(
        "baseline_done", "wait", params={"requests": output("baseline", "requests")}
    )
    case.step(
        "baseline_complete",
        "check",
        params={
            "actual": output("baseline_done", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "baseline_no_errors",
        "check",
        params={
            "actual": output("baseline_done", "error_count"),
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("kill", "master_fault", params={"mode": "kill", "target": "single"})
    case.step(
        "restart",
        "master_restore",
        timeout_s=180,
        params={"fault": output("kill", "fault")},
    )
    case.step(
        "restored_topology",
        "master_ready",
        timeout_s=60,
        params={"target": "single", "inflight_zero": False},
    )
    case.step(
        "restored_inflight",
        "master_ready",
        timeout_s=10,
        params={"target": "single", "inflight_zero": True},
    )
    case.step("recovery", "request", params={"count": 1, "output_len": 2})
    case.step(
        "recovery_done", "wait", params={"requests": output("recovery", "requests")}
    )
    case.step(
        "recovery_complete",
        "check",
        params={
            "actual": output("recovery_done", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "recovery_no_errors",
        "check",
        params={
            "actual": output("recovery_done", "error_count"),
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


def freeze_short_long(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "flow", "master_client_start", params={"targets": ["B", "A"], "duration_s": 100}
    )
    case.step("short_begin", "master_mark", params={"wait_s": 12})
    case.step("short_freeze", "master_fault", params={"mode": "freeze", "target": "B"})
    case.step("short_end", "master_mark", params={"wait_s": 6})
    case.step(
        "short_restore",
        "master_restore",
        params={"fault": output("short_freeze", "fault")},
    )
    case.step("post_short_end", "master_mark", params={"wait_s": 8})
    case.step("before", "master_state", params={"target": "B"})
    case.step("long_begin", "master_mark", params={"wait_s": 0})
    case.step("long_freeze", "master_fault", params={"mode": "freeze", "target": "B"})
    # The HA flow may succeed through A; use a no-fallback request to frozen B
    # to exercise the deadline path deterministically.
    case.step(
        "deadline_probe",
        "master_request_batch",
        timeout_s=10,
        params={
            "target": "B",
            "count": 1,
            "concurrency": 1,
            "request_timeout_s": 2,
            "sample_topology": False,
        },
    )
    case.step("long_end", "master_mark", params={"wait_s": 46})
    case.step(
        "long_restore",
        "master_restore",
        params={"fault": output("long_freeze", "fault")},
    )
    case.step("after", "master_scheduler_state", params={"target": "B"})
    case.step("post_long_end", "master_mark", params={"wait_s": 10})
    case.step("ready_b", "master_topology_state", params={"target": "B"})
    case.step(
        "continuity",
        "master_continuity",
        params={
            "before": output("before", "state"),
            "after": output("after", "state"),
            "settled": output("ready_b", "state"),
        },
    )
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=120,
        params={"client": output("flow", "client")},
    )
    case.step(
        "short_hang",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("short_begin", "epoch_s"),
            "until": output("short_end", "epoch_s"),
        },
    )
    case.step(
        "short_post",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("short_end", "epoch_s"),
            "until": output("post_short_end", "epoch_s"),
            # Match the burst guard: tail requests can straddle the long freeze.
            "until_offset_s": -0.5,
        },
    )
    case.step(
        "short_burst",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("short_end", "epoch_s"),
            "until": output("long_begin", "epoch_s"),
            "until_offset_s": -0.5,
        },
    )
    case.step(
        "short_verdict",
        "master_short_hang_check",
        params={
            "hang": output("short_hang", "rows"),
            "burst": output("short_burst", "rows"),
            "post": output("short_post", "rows"),
            "target": "B",
        },
    )
    case.step(
        "judged",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("long_begin", "epoch_s"),
            "from_offset_s": 29,
            "until": output("long_end", "epoch_s"),
        },
    )
    case.step(
        "pre_freeze",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("long_begin", "epoch_s"),
            "from_offset_s": -2,
            "until": output("long_begin", "epoch_s"),
        },
    )
    case.step(
        "deadline_straddle",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("long_begin", "epoch_s"),
            "from_offset_s": -4,
            "until": output("long_end", "epoch_s"),
        },
    )
    case.step(
        "post_long",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("long_end", "epoch_s"),
            "until": output("post_long_end", "epoch_s"),
        },
    )
    case.step(
        "retry_seen",
        "master_client_check",
        params={
            "rows": output("deadline_straddle", "rows"),
            "metric": "failover_count",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "switch_to_a",
        "master_client_check",
        params={
            "rows": output("judged", "rows"),
            "metric": "target_count",
            "op": "ge",
            "expected": 1,
            "target": "A",
        },
    )
    case.step(
        "visible_terminal",
        "master_client_check",
        params={
            "rows": output("pre_freeze", "rows"),
            "metric": "visible_terminal_share",
            "op": "eq",
            "expected": 1,
        },
    )
    case.step(
        "deadline_visible",
        "master_deadline_probe_check",
        params={"snapshot": output("deadline_probe", "snapshot")},
    )
    case.step(
        "post_success",
        "master_client_check",
        params={
            "rows": output("post_long", "rows"),
            "metric": "success_rate",
            "op": "ge",
            "expected": 0.9,
        },
    )
    case.step(
        "post_on_a",
        "master_client_check",
        params={
            "rows": output("post_long", "rows"),
            "metric": "target_share",
            "op": "ge",
            "expected": 0.8,
            "target": "A",
        },
    )
    case.step(
        "unique_requests",
        "master_client_check",
        params={
            "rows": output("finish", "rows"),
            "metric": "duplicate_ids",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


def kill_dual_b_to_a(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "flow", "master_client_start", params={"targets": ["B", "A"], "duration_s": 90}
    )
    case.step("kill_time", "master_mark", params={"wait_s": 12})
    case.step("kill_b", "master_fault", params={"target": "B", "mode": "kill"})
    case.step("switched", "master_mark", params={"wait_s": 10})
    case.step(
        "restart_b",
        "master_restore",
        timeout_s=180,
        params={"fault": output("kill_b", "fault")},
    )
    case.step(
        "ready_b",
        "master_ready",
        timeout_s=60,
        params={"target": "B", "inflight_zero": False},
    )
    case.step(
        "clean_b",
        "master_ready",
        timeout_s=10,
        params={"target": "B", "inflight_zero": True},
    )
    case.step(
        "recovery_a",
        "master_request_batch",
        timeout_s=630,
        params={"target": "A", "count": 20, "concurrency": 1, "request_timeout_s": 30},
    )
    case.step(
        "recovery_rate",
        "check",
        params={
            "actual": output("recovery_a", "success_rate"),
            "op": "ge",
            "expected": 0.95,
        },
    )
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=120,
        params={"client": output("flow", "client")},
    )
    case.step(
        "steady_plain",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "until": output("kill_time", "epoch_s"),
            "failover": False,
        },
    )
    case.step(
        "straddle",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("kill_time", "epoch_s"),
            "from_offset_s": -10,
            "until": output("switched", "epoch_s"),
        },
    )
    case.step(
        "switch",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("kill_time", "epoch_s"),
            "until": output("switched", "epoch_s"),
        },
    )
    case.step(
        "after",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("switched", "epoch_s"),
        },
    )
    case.step(
        "steady_b",
        "master_client_check",
        params={
            "rows": output("steady_plain", "rows"),
            "metric": "target_share",
            "op": "eq",
            "expected": 1,
            "target": "B",
            "min_samples": 10,
        },
    )
    case.step(
        "retry_seen",
        "master_client_check",
        params={
            "rows": output("straddle", "rows"),
            "metric": "failover_count",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "switch_to_a",
        "master_client_check",
        params={
            "rows": output("switch", "rows"),
            "metric": "target_count",
            "op": "ge",
            "expected": 1,
            "target": "A",
        },
    )
    case.step(
        "switch_errors",
        "master_client_check",
        params={
            "rows": output("switch", "rows"),
            "metric": "failed_rate_above_one",
            "op": "le",
            "expected": 0.05,
        },
    )
    case.step(
        "after_a",
        "master_client_check",
        params={
            "rows": output("after", "rows"),
            "metric": "target_share",
            "op": "ge",
            "expected": 0.95,
            "target": "A",
        },
    )
    case.step(
        "after_success",
        "master_client_check",
        params={
            "rows": output("after", "rows"),
            "metric": "success_rate",
            "op": "ge",
            "expected": 0.9,
        },
    )
    case.step(
        "unique_requests",
        "master_client_check",
        params={
            "rows": output("finish", "rows"),
            "metric": "duplicate_ids",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "kill_single": {
        "build": kill_single,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "freeze_short_long": {
        "build": freeze_short_long,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "kill_dual_b_to_a": {
        "build": kill_dual_b_to_a,
        "profiles": ["batch-window"],
        "metadata": {},
    },
}
