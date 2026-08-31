load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "pp_layout_test",
    srcs = ["PPLayoutTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/config:pp_layout",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)
