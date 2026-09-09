"""A newly added worker serves traffic, stops, and serves fresh traffic after restart."""

from ..case_config import output

METADATA = {
    "id": "elastic_added_worker_fault",
    "description": "A newly added worker serves traffic, stops, and serves fresh traffic after "
    "restart.",
    "category": "elastic",
    "requires": ["queue"],
    "estimated_duration_s": 100,
}

PROFILES = ["batch-window", "single-batch", "single-nonbatch", "window-nonbatch"]


def default(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params={"operation": "stop", "targets": [output("add", "engine")]},
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 2,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "survivor_request",
        "request",
        params={
            "input_len": 2048,
            "output_len": 2,
            "count": 1,
            "block_keys": [7],
            "schedule_timeout_s": 30,
            "stream_timeout_s": 10,
        },
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params={
            "actual": output("survivor_terminal", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "survivor_no_errors",
        "check",
        params={
            "actual": output("survivor_terminal", "error_count"),
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "restart",
        "engine_control",
        params={"operation": "start", "targets": [output("add", "engine")]},
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=65,
        params={
            "engine": output("add", "engine"),
            "window_s": 20,
            "method": "FetchResponse",
        },
    )
    case.step(
        "after_restart",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "cross_restart_growth",
        "elastic_added_growth",
        params={
            "before": output("before_stop", "accepted"),
            "after": output("after_restart", "accepted"),
        },
    )
    case.step("teardown", "teardown")


def single_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params={"operation": "stop", "targets": [output("add", "engine")]},
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 2,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "survivor_request",
        "request",
        params={
            "input_len": 2048,
            "output_len": 2,
            "count": 1,
            "block_keys": [7],
            "schedule_timeout_s": 30,
            "stream_timeout_s": 10,
        },
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params={
            "actual": output("survivor_terminal", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "survivor_no_errors",
        "check",
        params={
            "actual": output("survivor_terminal", "error_count"),
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "restart",
        "engine_control",
        params={"operation": "start", "targets": [output("add", "engine")]},
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=65,
        params={
            "engine": output("add", "engine"),
            "window_s": 20,
            "method": "FetchResponse",
        },
    )
    case.step(
        "after_restart",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "cross_restart_growth",
        "elastic_added_growth",
        params={
            "before": output("before_stop", "accepted"),
            "after": output("after_restart", "accepted"),
        },
    )
    case.step("teardown", "teardown")


def single_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params={"operation": "stop", "targets": [output("add", "engine")]},
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 2,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "survivor_request",
        "request",
        params={
            "input_len": 2048,
            "output_len": 2,
            "count": 1,
            "block_keys": [7],
            "schedule_timeout_s": 30,
            "stream_timeout_s": 10,
        },
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params={
            "actual": output("survivor_terminal", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "survivor_no_errors",
        "check",
        params={
            "actual": output("survivor_terminal", "error_count"),
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "restart",
        "engine_control",
        params={"operation": "start", "targets": [output("add", "engine")]},
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=65,
        params={
            "engine": output("add", "engine"),
            "window_s": 20,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "after_restart",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "cross_restart_growth",
        "elastic_added_growth",
        params={
            "before": output("before_stop", "accepted"),
            "after": output("after_restart", "accepted"),
        },
    )
    case.step("teardown", "teardown")


def window_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params={"operation": "stop", "targets": [output("add", "engine")]},
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 2,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "survivor_request",
        "request",
        params={
            "input_len": 2048,
            "output_len": 2,
            "count": 1,
            "block_keys": [7],
            "schedule_timeout_s": 30,
            "stream_timeout_s": 10,
        },
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params={
            "actual": output("survivor_terminal", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "survivor_no_errors",
        "check",
        params={
            "actual": output("survivor_terminal", "error_count"),
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "restart",
        "engine_control",
        params={"operation": "start", "targets": [output("add", "engine")]},
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=65,
        params={
            "engine": output("add", "engine"),
            "window_s": 20,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "after_restart",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "cross_restart_growth",
        "elastic_added_growth",
        params={
            "before": output("before_stop", "accepted"),
            "after": output("after_restart", "accepted"),
        },
    )
    case.step("teardown", "teardown")


VARIANTS = {
    "default": {"build": default, "profiles": ["batch-window"], "metadata": {}},
    "single_batch": {
        "build": single_batch,
        "profiles": ["single-batch"],
        "metadata": {},
    },
    "single_nonbatch": {
        "build": single_nonbatch,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "window_nonbatch": {
        "build": window_nonbatch,
        "profiles": ["window-nonbatch"],
        "metadata": {},
    },
}
