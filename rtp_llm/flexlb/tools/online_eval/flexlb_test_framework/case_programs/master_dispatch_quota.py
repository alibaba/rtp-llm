"""Fill one batch quota, stop its prefill, observe blocked traffic and independent scheduler TTL cleanup, then recover."""

from ..case_config import output


def single_prefill_ttl(case):
    case.step(
        "setup", "setup", timeout_s=case.value("single_prefill_ttl.setup_timeout_s")
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("single_prefill_ttl.slow"),
    )
    case.step("fill", "request", params=case.value("single_prefill_ttl.fill"))
    case.step(
        "all_fill_admitted",
        "master_admission_check",
        params=case.params(
            "single_prefill_ttl.all_fill_admitted",
            {"requests": output("fill", "requests")},
        ),
    )
    case.step(
        "quota_held",
        "master_wait_inflight",
        timeout_s=case.value("single_prefill_ttl.quota_held_timeout_s"),
        params=case.value("single_prefill_ttl.quota_held"),
    )
    case.step("stop", "engine_control", params=case.value("single_prefill_ttl.stop"))
    case.step(
        "eviction_begin",
        "master_mark",
        params=case.value("single_prefill_ttl.eviction_begin"),
    )
    case.step(
        "blocked",
        "master_request_batch",
        params=case.value("single_prefill_ttl.blocked"),
    )
    case.step(
        "block_verdict",
        "check",
        params=case.params(
            "single_prefill_ttl.block_verdict",
            {"actual": output("blocked", "success_rate")},
        ),
    )
    case.step(
        "ttl_empty",
        "master_wait_inflight",
        timeout_s=case.value("single_prefill_ttl.ttl_empty_timeout_s"),
        params=case.value("single_prefill_ttl.ttl_empty"),
    )
    case.step(
        "start",
        "engine_control",
        params=case.value("single_prefill_ttl.start"),
    )
    case.step(
        "normal_perf",
        "engine_control",
        params=case.value("single_prefill_ttl.normal_perf"),
    )
    case.step(
        "ready",
        "master_prefill_alive",
        timeout_s=case.value("single_prefill_ttl.ready_timeout_s"),
    )
    case.step("settle", "master_mark", params=case.value("single_prefill_ttl.settle"))
    case.step(
        "recovery",
        "master_request_batch",
        timeout_s=case.value("single_prefill_ttl.recovery_timeout_s"),
        params=case.value("single_prefill_ttl.recovery"),
    )
    case.step(
        "recovered",
        "check",
        params=case.params(
            "single_prefill_ttl.recovered",
            {"actual": output("recovery", "success_rate")},
        ),
    )
    case.step("cleanup", "teardown")
