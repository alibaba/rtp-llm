"""A real completion delivered after a deliberately incomplete Decode report."""

METADATA = {
    "id": "late_completion",
    "category": "status",
    "tags": ["protocol", "regression"],
    "description": "Late terminal delivery must retire scheduler ownership after an incomplete status report.",
}
PROFILES = ["single-batch"]


def after_missing(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "late_terminal",
        "late_completion_probe",
        timeout_s=90,
        params={
            "missing_rounds": case.number("missing_rounds", 1, maximum=10),
            "delay_ms": case.number("delay_ms", 300, minimum=1, maximum=10000),
            "clean_window_s": case.number("clean_window_s", 30, minimum=30, maximum=30),
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "after_missing": {"build": after_missing, "profiles": PROFILES, "metadata": {}}
}
