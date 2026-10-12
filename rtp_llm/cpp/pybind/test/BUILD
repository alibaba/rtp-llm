load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "rtp_llm_op_shutdown_test",
    srcs = ["RtpLLMOpShutdownTest.cc"],
    copts = copts(),
    # th_transformer_lib resolves runtime and autil symbols from this shared
    # library. Keep it after the test objects so --as-needed does not drop it,
    # and declare it as a linker input for remote execution.
    additional_linker_inputs = ["//:rtp_compute_ops"],
    # The wrapper runs this binary from its package directory, whereas
    # librtp_compute_ops.so lives at the runfiles repository root.
    data = ["//:rtp_compute_ops"],
    linkopts = [
        "-Wl,-rpath='$$ORIGIN/../../../..'",
        "-Wl,--no-as-needed",
        "$(location //:rtp_compute_ops)",
        "-Wl,--as-needed",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/pybind:th_transformer_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
