load("//:def.bzl", "copts")

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob(["*.cc"]),
    deps = [
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/cache_new:cache_new",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

