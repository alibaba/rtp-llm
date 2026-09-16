"""Prepare real background load, add Decode capacity, and verify Java traffic."""

from ..case_config import output


def prepared_background(case):
    case.step("setup", "setup", timeout_s=case.value("setup_timeout_s"))
    case.step("background", "java_flow_start", params=case.value("background"))
    case.step(
        "prepared",
        "java_flow_checkpoint",
        params=case.params("prepared", {"flow": output("background", "flow")}),
    )
    case.step("formal", "java_flow_start", params=case.value("formal"))
    case.step(
        "formal_active",
        "java_flow_checkpoint",
        params=case.params("formal_active", {"flow": output("formal", "flow")}),
    )
    case.step("add_decode", "elastic_add", params=case.value("add_decode"))
    case.step(
        "new_decode_serving",
        "java_flow_checkpoint",
        params=case.params(
            "new_decode_serving",
            {
                "flow": output("formal", "flow"),
                "engine": output("add_decode", "engine"),
            },
        ),
    )
    case.step(
        "background_persisted",
        "java_flow_checkpoint",
        params=case.params(
            "background_persisted", {"flow": output("background", "flow")}
        ),
    )
    case.step(
        "formal_drain", "java_flow_drain", params={"flow": output("formal", "flow")}
    )
    case.step(
        "background_stop",
        "java_flow_stop",
        params={"flow": output("background", "flow")},
    )
    case.step(
        "background_drain",
        "java_flow_drain",
        params={"flow": output("background", "flow")},
    )
    case.observe(
        "formal_check",
        "java_flow_check",
        params=case.params("check", {"flow": output("formal", "flow")}),
    )
    case.observe(
        "background_check",
        "java_flow_check",
        params=case.params("check", {"flow": output("background", "flow")}),
    )
    case.step("teardown", "teardown")
