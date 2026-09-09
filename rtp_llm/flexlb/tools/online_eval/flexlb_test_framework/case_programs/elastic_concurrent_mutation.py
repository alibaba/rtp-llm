"""Four concurrent add/remove workers and serial health probes, followed by quiescent discovery validation."""

from ..case_config import output

METADATA = {
    "id": "elastic_concurrent_mutation",
    "description": "Four concurrent add/remove workers and serial health probes, followed by "
    "quiescent discovery validation.",
    "category": "elastic",
    "requires": ["queue"],
    "estimated_duration_s": 90,
}

PROFILES = ["batch-window", "single-batch", "single-nonbatch", "window-nonbatch"]


def default(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("crossfire", "elastic_crossfire", timeout_s=120)
    case.step(
        "health",
        "elastic_flow_assert",
        params={"result": output("crossfire", "result"), "min_success_rate": 0.5},
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")


def single_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("crossfire", "elastic_crossfire", timeout_s=120)
    case.step(
        "protocol",
        "elastic_crossfire_protocol",
        params={"requests": output("crossfire", "requests"), "method": "FetchResponse"},
    )
    case.step(
        "health",
        "elastic_flow_assert",
        params={"result": output("crossfire", "result"), "min_success_rate": 0.5},
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")


def single_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("crossfire", "elastic_crossfire", timeout_s=120)
    case.step(
        "protocol",
        "elastic_crossfire_protocol",
        params={
            "requests": output("crossfire", "requests"),
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "health",
        "elastic_flow_assert",
        params={"result": output("crossfire", "result"), "min_success_rate": 0.5},
    )
    case.step("discovery", "elastic_discovery_consistency")
    case.step("teardown", "teardown")


def window_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("crossfire", "elastic_crossfire", timeout_s=120)
    case.step(
        "protocol",
        "elastic_crossfire_protocol",
        params={
            "requests": output("crossfire", "requests"),
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "health",
        "elastic_flow_assert",
        params={"result": output("crossfire", "result"), "min_success_rate": 0.5},
    )
    case.step("discovery", "elastic_discovery_consistency")
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
