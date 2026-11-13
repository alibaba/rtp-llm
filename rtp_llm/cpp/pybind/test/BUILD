load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "rtp_llm_op_shutdown_test",
    srcs = ["RtpLLMOpShutdownTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/pybind:th_transformer_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
