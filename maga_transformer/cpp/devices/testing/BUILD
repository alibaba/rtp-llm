load("//:def.bzl", "copts")

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob(["*.cc"]),
    deps = [
        "@com_google_googletest//:gtest",
        "//maga_transformer/cpp:cache",
        "//maga_transformer/cpp:dataclass",
        "//:gpt_init_params",
        "//maga_transformer/cpp/devices:device_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

