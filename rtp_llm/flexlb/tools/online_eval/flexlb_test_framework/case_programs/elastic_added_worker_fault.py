"""A newly added worker serves traffic, stops, and serves fresh traffic after restart."""

from ..case_config import output


def default(case):
    case.step("setup", "setup", timeout_s=case.value("default.setup_timeout_s"))
    case.step("add", "elastic_add", params=case.value("default.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        params=case.params("default.added_topology", {"port": output("add", "port")}),
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=case.value("default.first_traffic_timeout_s"),
        params=case.params(
            "default.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params=case.params("default.stop", {"targets": [output("add", "engine")]}),
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params=case.params("default.stopped_topology", {"port": output("add", "port")}),
    )
    case.step(
        "survivor_request",
        "request",
        params=case.value("default.survivor_request"),
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params=case.params(
            "default.survivor_completed",
            {"actual": output("survivor_terminal", "completed")},
        ),
    )
    case.step(
        "survivor_no_errors",
        "check",
        params=case.params(
            "default.survivor_no_errors",
            {"actual": output("survivor_terminal", "error_count")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.params("default.restart", {"targets": [output("add", "engine")]}),
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params=case.params(
            "default.restored_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=case.value("default.resumed_traffic_timeout_s"),
        params=case.params(
            "default.resumed_traffic", {"engine": output("add", "engine")}
        ),
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
    case.step("setup", "setup", timeout_s=case.value("single_batch.setup_timeout_s"))
    case.step("add", "elastic_add", params=case.value("single_batch.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        params=case.params(
            "single_batch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=case.value("single_batch.first_traffic_timeout_s"),
        params=case.params(
            "single_batch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params=case.params("single_batch.stop", {"targets": [output("add", "engine")]}),
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params=case.params(
            "single_batch.stopped_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "survivor_request",
        "request",
        params=case.value("single_batch.survivor_request"),
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params=case.params(
            "single_batch.survivor_completed",
            {"actual": output("survivor_terminal", "completed")},
        ),
    )
    case.step(
        "survivor_no_errors",
        "check",
        params=case.params(
            "single_batch.survivor_no_errors",
            {"actual": output("survivor_terminal", "error_count")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.params(
            "single_batch.restart", {"targets": [output("add", "engine")]}
        ),
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params=case.params(
            "single_batch.restored_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=case.value("single_batch.resumed_traffic_timeout_s"),
        params=case.params(
            "single_batch.resumed_traffic", {"engine": output("add", "engine")}
        ),
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
    case.step("setup", "setup", timeout_s=case.value("single_nonbatch.setup_timeout_s"))
    case.step("add", "elastic_add", params=case.value("single_nonbatch.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        params=case.params(
            "single_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=case.value("single_nonbatch.first_traffic_timeout_s"),
        params=case.params(
            "single_nonbatch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params=case.params(
            "single_nonbatch.stop", {"targets": [output("add", "engine")]}
        ),
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params=case.params(
            "single_nonbatch.stopped_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "survivor_request",
        "request",
        params=case.value("single_nonbatch.survivor_request"),
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params=case.params(
            "single_nonbatch.survivor_completed",
            {"actual": output("survivor_terminal", "completed")},
        ),
    )
    case.step(
        "survivor_no_errors",
        "check",
        params=case.params(
            "single_nonbatch.survivor_no_errors",
            {"actual": output("survivor_terminal", "error_count")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.params(
            "single_nonbatch.restart", {"targets": [output("add", "engine")]}
        ),
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params=case.params(
            "single_nonbatch.restored_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=case.value("single_nonbatch.resumed_traffic_timeout_s"),
        params=case.params(
            "single_nonbatch.resumed_traffic", {"engine": output("add", "engine")}
        ),
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
    case.step("setup", "setup", timeout_s=case.value("window_nonbatch.setup_timeout_s"))
    case.step("add", "elastic_add", params=case.value("window_nonbatch.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        params=case.params(
            "window_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "first_traffic",
        "elastic_added_probe",
        timeout_s=case.value("window_nonbatch.first_traffic_timeout_s"),
        params=case.params(
            "window_nonbatch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "before_stop",
        "elastic_accepted_snapshot",
        params={"engine": output("add", "engine")},
    )
    case.step(
        "stop",
        "engine_control",
        params=case.params(
            "window_nonbatch.stop", {"targets": [output("add", "engine")]}
        ),
    )
    case.step(
        "stopped_topology",
        "elastic_topology",
        params=case.params(
            "window_nonbatch.stopped_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "survivor_request",
        "request",
        params=case.value("window_nonbatch.survivor_request"),
    )
    case.step(
        "survivor_terminal",
        "wait",
        params={"requests": output("survivor_request", "requests")},
    )
    case.step(
        "survivor_completed",
        "check",
        params=case.params(
            "window_nonbatch.survivor_completed",
            {"actual": output("survivor_terminal", "completed")},
        ),
    )
    case.step(
        "survivor_no_errors",
        "check",
        params=case.params(
            "window_nonbatch.survivor_no_errors",
            {"actual": output("survivor_terminal", "error_count")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.params(
            "window_nonbatch.restart", {"targets": [output("add", "engine")]}
        ),
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        params=case.params(
            "window_nonbatch.restored_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "resumed_traffic",
        "elastic_added_probe",
        timeout_s=case.value("window_nonbatch.resumed_traffic_timeout_s"),
        params=case.params(
            "window_nonbatch.resumed_traffic", {"engine": output("add", "engine")}
        ),
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
