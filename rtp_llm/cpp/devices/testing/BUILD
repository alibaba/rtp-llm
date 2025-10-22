load("//:def.bzl", "copts")

cc_library(
    name = "device_test_utils",
    srcs = glob(["*.cc"]),
    hdrs = glob([
        "*.h",
    ]),
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/cache2",
        "@com_google_googletest//:gtest",
    ],
)
