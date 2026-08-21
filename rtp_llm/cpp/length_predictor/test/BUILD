load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_test(
    name = "length_predictor_test",
    srcs = [
        "LengthPredictorTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/length_predictor:length_predictor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)
