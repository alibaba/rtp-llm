"""Submitted requests reach a business terminal without stream errors."""

from ..case_config import output

METADATA = {
    "id": "request_completion",
    "description": "Submitted requests reach a business terminal without stream errors.",
    "category": "status",
    "tags": ["smoke", "lifecycle"],
}
PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def completion(case, *, deferred=False):
    """Both consumption modes share the same terminal and zero-error contract."""
    request = {
        "input_len": case.number("input_len", 2048, maximum=2**31 - 1),
        "output_len": case.number("output_len", 2, maximum=2**31 - 1),
        "count": case.number("count", 1, maximum=10000),
    }
    if deferred:
        request["consume"] = "deferred"
    case.step("setup", "setup", timeout_s=180)
    case.step("submit", "request", params=request)
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    for name, field, expected in (
        ("completed", "completed", True),
        ("no_errors", "error_count", 0),
    ):
        case.step(
            name,
            "check",
            params={
                "actual": output("terminal", field),
                "op": "eq",
                "expected": expected,
            },
        )
    case.step("cleanup", "teardown")


def immediate(case):
    completion(case)


def deferred_fetch(case):
    completion(case, deferred=True)


def client_no_fetch(case):
    shape = dict(
        input_len=case.number("input_len", 2048, maximum=2**31 - 1),
        output_len=case.number("output_len", 8, maximum=2**31 - 1),
        observe_s=case.number(
            "observe_s", 0.2, minimum=0.01, maximum=10, integer=False
        ),
    )
    case.step("setup", "setup", timeout_s=180)
    for mode in ("late", "missing"):
        case.step(
            mode, "client_fetch_probe", params=dict(shape, mode=mode), timeout_s=30
        )
    case.step(
        "automatic_environment",
        "environment_reconfigure",
        timeout_s=240,
        params={"config_overrides": {}, "mock_auto_fetch": True},
    )
    case.step(
        "automatic",
        "client_auto_fetch_probe",
        params=dict(shape, mode="automatic"),
        timeout_s=30,
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "client_no_fetch": {
        "build": client_no_fetch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "immediate": {"build": immediate, "profiles": PROFILES, "metadata": {}},
    "deferred_fetch": {
        "build": deferred_fetch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
}
