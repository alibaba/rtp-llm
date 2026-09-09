"""Explicit request baselines, controlled RPC faults, consumer-complete outcomes, recovery and applicable scheduler-owner cleanup."""

from ..case_config import output


def enqueue_delay_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("enqueue_delay_batch.setup_timeout_s")
    )
    case.step(
        "baseline",
        "rpc_latency_sample",
        timeout_s=case.value("enqueue_delay_batch.baseline_timeout_s"),
        params=case.value("enqueue_delay_batch.baseline"),
    )
    case.step(
        "baseline_ready",
        "check",
        params=case.params(
            "enqueue_delay_batch.baseline_ready",
            {"actual": output("baseline", "legacy_success")},
        ),
    )
    case.step(
        "inject",
        "engine_inject",
        params=case.value("enqueue_delay_batch.inject"),
    )
    case.step(
        "delayed",
        "rpc_latency_sample",
        timeout_s=case.value("enqueue_delay_batch.delayed_timeout_s"),
        params=case.value("enqueue_delay_batch.delayed"),
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "recovery",
        "rpc_latency_sample",
        timeout_s=case.value("enqueue_delay_batch.recovery_timeout_s"),
        params=case.value("enqueue_delay_batch.recovery"),
    )
    case.step(
        "latency",
        "rpc_latency_check",
        params=case.params(
            "enqueue_delay_batch.latency",
            {
                "baseline": output("baseline", "requests"),
                "delayed": output("delayed", "requests"),
                "recovery": output("recovery", "requests"),
            },
        ),
    )
    case.step(
        "owner_clean",
        "rpc_owner_clean",
        timeout_s=case.value("enqueue_delay_batch.owner_clean_timeout_s"),
    )
    case.step("cleanup", "teardown")


def generate_delay_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("generate_delay_batch.setup_timeout_s")
    )
    case.step(
        "baseline",
        "rpc_latency_sample",
        timeout_s=case.value("generate_delay_batch.baseline_timeout_s"),
        params=case.value("generate_delay_batch.baseline"),
    )
    case.step(
        "baseline_ready",
        "check",
        params=case.params(
            "generate_delay_batch.baseline_ready",
            {"actual": output("baseline", "legacy_success")},
        ),
    )
    case.step(
        "inject",
        "engine_inject",
        params=case.value("generate_delay_batch.inject"),
    )
    case.step(
        "delayed",
        "rpc_latency_sample",
        timeout_s=case.value("generate_delay_batch.delayed_timeout_s"),
        params=case.value("generate_delay_batch.delayed"),
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "recovery",
        "rpc_latency_sample",
        timeout_s=case.value("generate_delay_batch.recovery_timeout_s"),
        params=case.value("generate_delay_batch.recovery"),
    )
    case.step(
        "latency",
        "rpc_latency_check",
        params=case.params(
            "generate_delay_batch.latency",
            {
                "baseline": output("baseline", "requests"),
                "delayed": output("delayed", "requests"),
                "recovery": output("recovery", "requests"),
            },
        ),
    )
    case.step(
        "owner_clean",
        "rpc_owner_clean",
        timeout_s=case.value("generate_delay_batch.owner_clean_timeout_s"),
    )
    case.step("cleanup", "teardown")


def generate_delay_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("generate_delay_nonbatch.setup_timeout_s"),
    )
    case.step(
        "baseline",
        "rpc_latency_sample",
        timeout_s=case.value("generate_delay_nonbatch.baseline_timeout_s"),
        params=case.value("generate_delay_nonbatch.baseline"),
    )
    case.step(
        "baseline_ready",
        "check",
        params=case.params(
            "generate_delay_nonbatch.baseline_ready",
            {"actual": output("baseline", "legacy_success")},
        ),
    )
    case.step(
        "inject",
        "engine_inject",
        params=case.value("generate_delay_nonbatch.inject"),
    )
    case.step(
        "delayed",
        "rpc_latency_sample",
        timeout_s=case.value("generate_delay_nonbatch.delayed_timeout_s"),
        params=case.value("generate_delay_nonbatch.delayed"),
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "recovery",
        "rpc_latency_sample",
        timeout_s=case.value("generate_delay_nonbatch.recovery_timeout_s"),
        params=case.value("generate_delay_nonbatch.recovery"),
    )
    case.step(
        "latency",
        "rpc_latency_check",
        params=case.params(
            "generate_delay_nonbatch.latency",
            {
                "baseline": output("baseline", "requests"),
                "delayed": output("delayed", "requests"),
                "recovery": output("recovery", "requests"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def enqueue_error(case):
    case.step("setup", "setup", timeout_s=case.value("enqueue_error.setup_timeout_s"))
    case.step(
        "inject",
        "engine_inject",
        params=case.value("enqueue_error.inject"),
    )
    case.step(
        "probe",
        "rpc_fault_probe",
        timeout_s=case.value("enqueue_error.probe_timeout_s"),
        params=case.value("enqueue_error.probe"),
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "cancel_failed_delivery",
        "rpc_probe_cancel",
        params={"requests": output("probe", "requests")},
    )
    case.step(
        "recovery_settle",
        "balance_pause",
        params=case.value("enqueue_error.recovery_settle"),
    )
    case.step(
        "recovery",
        "request",
        params=case.value("enqueue_error.recovery"),
    )
    case.step(
        "recovery_terminal",
        "wait",
        timeout_s=case.value("enqueue_error.recovery_terminal_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "check",
        params=case.params(
            "enqueue_error.recovery_success",
            {"actual": output("recovery_terminal", "completed")},
        ),
    )
    case.step(
        "owner_clean",
        "rpc_probe_owner_clean",
        timeout_s=case.value("enqueue_error.owner_clean_timeout_s"),
        params={"requests": output("probe", "requests")},
    )
    case.step("cleanup", "teardown")
