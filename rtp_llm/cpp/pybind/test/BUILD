load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "rtp_llm_op_shutdown_test",
    srcs = ["RtpLLMOpShutdownTest.cc"],
    copts = copts(),
    # th_transformer_lib resolves runtime and autil symbols from this shared
    # library. Keep it after the test objects so --as-needed does not drop it.
    linkopts = [
        "-Wl,--no-as-needed",
        "-lrtp_compute_ops",
        "-Wl,--as-needed",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/pybind:th_transformer_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
