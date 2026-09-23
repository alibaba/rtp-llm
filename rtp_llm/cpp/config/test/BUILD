load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "rank_layout_test",
    srcs = ["RankLayoutTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/config:rank_layout",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)

cc_test(
    name = "local_cp_policy_test",
    srcs = ["LocalCPPolicyTest.cc"],
    copts = test_copts,
    # The standalone config test links alog without GTest's pthread dependency.
    linkopts = ["-pthread"],
    deps = ["//rtp_llm/cpp/config:config_modules"],
    env = {},
)
