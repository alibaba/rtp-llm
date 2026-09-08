load("//:def.bzl", "copts")

cc_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/telemetry",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@io_opentelemetry_cpp//exporters/memory:in_memory_span_exporter",
    ],
)

# Real in-process gRPC transport roundtrip for traceparent metadata;
# proto-only dep, no engine closure.
cc_test(
    name = "grpc_propagation_test",
    srcs = ["grpc_propagation_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/telemetry",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@io_opentelemetry_cpp//exporters/memory:in_memory_span_exporter",
    ],
)

# Post-hoc phase child span synthesis: verifies
# wait/prefill/decode INTERNAL spans carry correct timestamps and parentage.
cc_test(
    name = "phase_span_synthesizer_test",
    srcs = ["phase_span_synthesizer_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/telemetry",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@io_opentelemetry_cpp//exporters/memory:in_memory_span_exporter",
    ],
)
