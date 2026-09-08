load("@arch_config//:arch_select.bzl", "telemetry_test_deps")

py_test(
    name = "test_tracing",
    srcs = ["test_tracing.py"],
    # unittest style (repo convention, no pytest dependency). The tracing SDK is
    # optional and absent from the public Bazel locks, so SDK-dependent methods
    # explicitly skip when telemetry_test_deps() cannot provide it.
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
