load("//:def.bzl", "copts")

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob(["*.cc"]),
    deps = [
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/devices:device_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

