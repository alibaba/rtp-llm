"""Private pending-wave scale-in; legacy visible terminals and an explicit stronger all-issued zero-error variant."""

from ..case_config import output

METADATA = {
    "id": "elastic_pending_drain",
    "description": "Private pending-wave scale-in; legacy visible terminals and an explicit stronger "
    "all-issued zero-error variant.",
    "category": "elastic",
    "requires": ["enqueue_batch"],
    "estimated_duration_s": 120,
}

PROFILES = ["batch-window", "single-batch"]


def legacy_terminal(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_both_prefills",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 8000},
        },
    )
    case.step("perf_sync", "elastic_pause", params={"seconds": 1.5})
    case.step("wave", "elastic_pending_wave", timeout_s=450)
    case.step("remove", "elastic_pending_remove", timeout_s=110)
    case.step(
        "collect",
        "elastic_pending_collect",
        timeout_s=100,
        params={
            "requests": output("wave", "requests"),
            "mutation": output("remove", "mutation"),
        },
    )
    case.step("accounting", "elastic_pending_accounting", timeout_s=60)
    case.step(
        "restore_survivor",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("recovery", "elastic_pending_recovery", timeout_s=100)
    case.step(
        "topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 1,
            "alive": 1,
            "port": output("remove", "port"),
            "present": False,
        },
    )
    case.step(
        "visible_terminal",
        "elastic_pending_visible",
        params={"result": output("collect", "result")},
    )
    case.step("teardown", "teardown")


def zero_errors(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_both_prefills",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 8000},
        },
    )
    case.step("perf_sync", "elastic_pause", params={"seconds": 1.5})
    case.step("wave", "elastic_pending_wave", timeout_s=450)
    case.step("remove", "elastic_pending_remove", timeout_s=110)
    case.step(
        "collect",
        "elastic_pending_collect",
        timeout_s=100,
        params={
            "requests": output("wave", "requests"),
            "mutation": output("remove", "mutation"),
        },
    )
    case.step("accounting", "elastic_pending_accounting", timeout_s=60)
    case.step(
        "restore_survivor",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("recovery", "elastic_pending_recovery", timeout_s=100)
    case.step(
        "topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 1,
            "alive": 1,
            "port": output("remove", "port"),
            "present": False,
        },
    )
    case.step(
        "visible_terminal",
        "elastic_pending_visible",
        params={"result": output("collect", "result")},
    )
    case.step(
        "all_issued_zero_errors",
        "elastic_flow_assert",
        params={"result": output("collect", "summary"), "min_success_rate": 1.0},
    )
    case.step("teardown", "teardown")


def single_batch_terminal(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_both_prefills",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 8000},
        },
    )
    case.step("perf_sync", "elastic_pause", params={"seconds": 1.5})
    case.step("wave", "elastic_pending_wave", timeout_s=450)
    case.step(
        "batch_path",
        "elastic_pending_batch_path",
        params={"requests": output("wave", "requests")},
    )
    case.step("remove", "elastic_pending_remove", timeout_s=110)
    case.step(
        "collect",
        "elastic_pending_collect",
        timeout_s=100,
        params={
            "requests": output("wave", "requests"),
            "mutation": output("remove", "mutation"),
        },
    )
    case.step("accounting", "elastic_pending_accounting", timeout_s=60)
    case.step(
        "restore_survivor",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("recovery", "elastic_pending_recovery", timeout_s=100)
    case.step(
        "topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 1,
            "alive": 1,
            "port": output("remove", "port"),
            "present": False,
        },
    )
    case.step(
        "visible_terminal",
        "elastic_pending_visible",
        params={"result": output("collect", "result")},
    )
    case.step("teardown", "teardown")


VARIANTS = {
    "legacy_terminal": {
        "build": legacy_terminal,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "zero_errors": {
        "build": zero_errors,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "single_batch_terminal": {
        "build": single_batch_terminal,
        "profiles": ["single-batch"],
        "metadata": {},
    },
}
