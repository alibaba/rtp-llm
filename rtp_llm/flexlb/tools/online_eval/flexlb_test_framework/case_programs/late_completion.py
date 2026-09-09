"""A real completion delivered after a deliberately incomplete Decode report."""


def after_missing(case):
    case.step("setup", "setup", timeout_s=case.value("after_missing.setup_timeout_s"))
    case.step(
        "late_terminal",
        "late_completion_probe",
        timeout_s=case.value("after_missing.late_terminal_timeout_s"),
        params={
            "missing_rounds": case.number("missing_rounds"),
            "delay_ms": case.number("delay_ms"),
            "clean_window_s": case.number("clean_window_s"),
        },
    )
    case.step("cleanup", "teardown")
