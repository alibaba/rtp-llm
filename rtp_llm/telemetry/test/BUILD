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

py_test(
    name = "trace_attributes_parity_test",
    srcs = ["trace_attributes_parity_test.py"],
    data = [
        "//rtp_llm:telemetry",
        "//rtp_llm/cpp/telemetry:TraceAttributes.h",
    ],
)
