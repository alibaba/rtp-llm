load("@arch_config//:arch_select.bzl", "telemetry_test_deps")

py_test(
    name = "test_tracing",
    srcs = ["test_tracing.py"],
    # unittest style (repo convention, no pytest dependency). telemetry_test_deps()
    # declares the opentelemetry runtime for the locks that carry it, so the
    # unskipped dependency-contract test asserts a Bazel-declared runtime instead
    # of whatever the execution image interpreter happens to have installed.
    deps = [
        "//rtp_llm:telemetry",
    ] + telemetry_test_deps(),
)

py_test(
    name = "trace_attributes_parity_test",
    srcs = ["trace_attributes_parity_test.py"],
    data = [
        "//rtp_llm:telemetry",
        "//rtp_llm/cpp/telemetry:TraceAttributes.h",
    ],
)
