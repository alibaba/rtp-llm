"""Actual dual standalone masters: sticky A to B through same-request failover, with per-request route evidence."""

from ..case_config import output

METADATA = {
    "id": "master_ha_failover",
    "description": "Actual dual standalone masters: sticky A to B through same-request failover, with "
    "per-request route evidence.",
    "category": "master",
}

PROFILES = ["batch-window"]


def standalone_a_to_b(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "flow", "master_client_start", params={"targets": ["A", "B"], "duration_s": 60}
    )
    case.step("lookback", "master_mark", params={"wait_s": 2})
    case.step("kill_time", "master_mark", params={"wait_s": 10})
    case.step("kill_a", "master_fault", params={"mode": "kill", "target": "A"})
    case.step("switched", "master_mark", params={"wait_s": 10})
    case.step("b_ready", "master_ready", params={"target": "B", "inflight_zero": False})
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=90,
        params={"client": output("flow", "client")},
    )
    case.step(
        "steady",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "until": output("kill_time", "epoch_s"),
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
        "straddle",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("lookback", "epoch_s"),
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
        "steady_a",
        "master_client_check",
        params={
            "rows": output("steady", "rows"),
            "metric": "target_share",
            "op": "eq",
            "expected": 1,
            "target": "A",
            "min_samples": 10,
        },
    )
    case.step(
        "failover_seen",
        "master_client_check",
        params={
            "rows": output("straddle", "rows"),
            "metric": "failover_count",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "switch_to_b",
        "master_client_check",
        params={
            "rows": output("switch", "rows"),
            "metric": "target_count",
            "op": "ge",
            "expected": 1,
            "target": "B",
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
        "after_b",
        "master_client_check",
        params={
            "rows": output("after", "rows"),
            "metric": "target_share",
            "op": "ge",
            "expected": 0.95,
            "target": "B",
            "min_samples": 20,
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
            "min_samples": 20,
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
    "standalone_a_to_b": {
        "build": standalone_a_to_b,
        "profiles": ["batch-window"],
        "metadata": {},
    },
}
