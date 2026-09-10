"""Private pending-wave scale-in; legacy visible terminals and an explicit stronger all-issued zero-error variant."""

from ..case_config import output


def legacy_terminal(case):
    case.step("setup", "setup", timeout_s=case.value("legacy_terminal.setup_timeout_s"))
    case.step(
        "slow_both_prefills",
        "engine_control",
        params=case.value("legacy_terminal.slow_both_prefills"),
    )
    case.step(
        "perf_sync", "elastic_pause", params=case.value("legacy_terminal.perf_sync")
    )
    case.step(
        "wave",
        "elastic_pending_wave",
        timeout_s=case.value("legacy_terminal.wave_timeout_s"),
        params=case.value("legacy_terminal.wave"),
    )
    case.step(
        "remove",
        "elastic_pending_remove",
        timeout_s=case.value("legacy_terminal.remove_timeout_s"),
    )
    case.step(
        "collect",
        "elastic_pending_collect",
        timeout_s=case.value("legacy_terminal.collect_timeout_s"),
        params={
            "requests": output("wave", "requests"),
            "mutation": output("remove", "mutation"),
        },
    )
    case.step(
        "accounting",
        "elastic_pending_accounting",
        timeout_s=case.value("legacy_terminal.accounting_timeout_s"),
    )
    case.step(
        "restore_survivor",
        "engine_control",
        params=case.value("legacy_terminal.restore_survivor"),
    )
    case.step(
        "recovery",
        "elastic_pending_recovery",
        timeout_s=case.value("legacy_terminal.recovery_timeout_s"),
    )
    case.step(
        "topology",
        "elastic_topology",
        timeout_s=case.value("legacy_terminal.topology_timeout_s"),
        params=case.params(
            "legacy_terminal.topology", {"port": output("remove", "port")}
        ),
    )
    case.step(
        "visible_terminal",
        "elastic_pending_visible",
        params={"result": output("collect", "result")},
    )
    case.step("teardown", "teardown")


def zero_errors(case):
    case.step("setup", "setup", timeout_s=case.value("zero_errors.setup_timeout_s"))
    case.step(
        "slow_both_prefills",
        "engine_control",
        params=case.value("zero_errors.slow_both_prefills"),
    )
    case.step("perf_sync", "elastic_pause", params=case.value("zero_errors.perf_sync"))
    case.step(
        "wave",
        "elastic_pending_wave",
        timeout_s=case.value("zero_errors.wave_timeout_s"),
        params=case.value("zero_errors.wave"),
    )
    case.step(
        "remove",
        "elastic_pending_remove",
        timeout_s=case.value("zero_errors.remove_timeout_s"),
    )
    case.step(
        "collect",
        "elastic_pending_collect",
        timeout_s=case.value("zero_errors.collect_timeout_s"),
        params={
            "requests": output("wave", "requests"),
            "mutation": output("remove", "mutation"),
        },
    )
    case.step(
        "accounting",
        "elastic_pending_accounting",
        timeout_s=case.value("zero_errors.accounting_timeout_s"),
    )
    case.step(
        "restore_survivor",
        "engine_control",
        params=case.value("zero_errors.restore_survivor"),
    )
    case.step(
        "recovery",
        "elastic_pending_recovery",
        timeout_s=case.value("zero_errors.recovery_timeout_s"),
    )
    case.step(
        "topology",
        "elastic_topology",
        timeout_s=case.value("zero_errors.topology_timeout_s"),
        params=case.params("zero_errors.topology", {"port": output("remove", "port")}),
    )
    case.step(
        "visible_terminal",
        "elastic_pending_visible",
        params={"result": output("collect", "result")},
    )
    case.step(
        "all_issued_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "zero_errors.all_issued_zero_errors",
            {"result": output("collect", "summary")},
        ),
    )
    case.step("teardown", "teardown")


def single_batch_terminal(case):
    case.step(
        "setup", "setup", timeout_s=case.value("single_batch_terminal.setup_timeout_s")
    )
    case.step(
        "slow_both_prefills",
        "engine_control",
        params=case.value("single_batch_terminal.slow_both_prefills"),
    )
    case.step(
        "perf_sync",
        "elastic_pause",
        params=case.value("single_batch_terminal.perf_sync"),
    )
    case.step(
        "wave",
        "elastic_pending_wave",
        timeout_s=case.value("single_batch_terminal.wave_timeout_s"),
        params=case.value("single_batch_terminal.wave"),
    )
    case.step(
        "batch_path",
        "elastic_pending_batch_path",
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "remove",
        "elastic_pending_remove",
        timeout_s=case.value("single_batch_terminal.remove_timeout_s"),
    )
    case.step(
        "collect",
        "elastic_pending_collect",
        timeout_s=case.value("single_batch_terminal.collect_timeout_s"),
        params={
            "requests": output("wave", "requests"),
            "mutation": output("remove", "mutation"),
        },
    )
    case.step(
        "accounting",
        "elastic_pending_accounting",
        timeout_s=case.value("single_batch_terminal.accounting_timeout_s"),
    )
    case.step(
        "restore_survivor",
        "engine_control",
        params=case.value("single_batch_terminal.restore_survivor"),
    )
    case.step(
        "recovery",
        "elastic_pending_recovery",
        timeout_s=case.value("single_batch_terminal.recovery_timeout_s"),
    )
    case.step(
        "topology",
        "elastic_topology",
        timeout_s=case.value("single_batch_terminal.topology_timeout_s"),
        params=case.params(
            "single_batch_terminal.topology", {"port": output("remove", "port")}
        ),
    )
    case.step(
        "visible_terminal",
        "elastic_pending_visible",
        params={"result": output("collect", "result")},
    )
    case.step("teardown", "teardown")
