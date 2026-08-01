py_test(
    name = "test_tracing",
    srcs = ["test_tracing.py"],
    # unittest style (repo convention, no pytest dependency). opentelemetry
    # comes from the execution image interpreter (not in the public pip locks;
    # rules_python here does not isolate system site-packages). An unskipped
    # dependency-contract test makes a missing runtime an explicit failure.
    deps = [
        "//rtp_llm:telemetry",
    ],
)
