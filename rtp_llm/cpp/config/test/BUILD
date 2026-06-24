load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "model_config_test",
    srcs = ["ModelConfigTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/config:model_config",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
