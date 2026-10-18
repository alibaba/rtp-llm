load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "dsv4_moe_strategy_test",
    srcs = ["Dsv4MoeStrategyTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/model_utils:dsv4_moe_strategy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)
