"""Four concurrent add/remove workers under bounded concurrent request load, followed by quiescent discovery validation."""

from ..case_config import output


def default(case):
    case.step("setup", "setup", timeout_s=case.value("default.setup_timeout_s"))
    case.step(
        "crossfire",
        "elastic_crossfire",
        timeout_s=case.value("default.crossfire_timeout_s"),
        params=case.value("default.crossfire"),
    )
    case.step(
        "health",
        "elastic_flow_assert",
        params=case.params("default.health", {"result": output("crossfire", "result")}),
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")


def single_batch(case):
    case.step("setup", "setup", timeout_s=case.value("single_batch.setup_timeout_s"))
    case.step(
        "crossfire",
        "elastic_crossfire",
        timeout_s=case.value("single_batch.crossfire_timeout_s"),
        params=case.value("single_batch.crossfire"),
    )
    case.step(
        "protocol",
        "elastic_crossfire_protocol",
        params=case.params(
            "single_batch.protocol", {"requests": output("crossfire", "requests")}
        ),
    )
    case.step(
        "health",
        "elastic_flow_assert",
        params=case.params(
            "single_batch.health", {"result": output("crossfire", "result")}
        ),
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")


def single_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("single_nonbatch.setup_timeout_s"))
    case.step(
        "crossfire",
        "elastic_crossfire",
        timeout_s=case.value("single_nonbatch.crossfire_timeout_s"),
        params=case.value("single_nonbatch.crossfire"),
    )
    case.step(
        "protocol",
        "elastic_crossfire_protocol",
        params=case.params(
            "single_nonbatch.protocol", {"requests": output("crossfire", "requests")}
        ),
    )
    case.step(
        "health",
        "elastic_flow_assert",
        params=case.params(
            "single_nonbatch.health", {"result": output("crossfire", "result")}
        ),
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")


def window_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("window_nonbatch.setup_timeout_s"))
    case.step(
        "crossfire",
        "elastic_crossfire",
        timeout_s=case.value("window_nonbatch.crossfire_timeout_s"),
        params=case.value("window_nonbatch.crossfire"),
    )
    case.step(
        "protocol",
        "elastic_crossfire_protocol",
        params=case.params(
            "window_nonbatch.protocol", {"requests": output("crossfire", "requests")}
        ),
    )
    case.step(
        "health",
        "elastic_flow_assert",
        params=case.params(
            "window_nonbatch.health", {"result": output("crossfire", "result")}
        ),
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")


def convergence(case):
    case.step("setup", "setup", timeout_s=case.value("convergence.setup_timeout_s"))
    case.step(
        "crossfire",
        "elastic_crossfire",
        timeout_s=case.value("convergence.crossfire_timeout_s"),
        params=case.value("convergence.crossfire"),
    )
    case.step(
        "convergence",
        "elastic_convergence_verdict",
        params={"result": output("crossfire", "result")},
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")
