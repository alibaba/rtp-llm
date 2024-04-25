load("//:def.bzl", "copts")

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "@com_google_googletest//:gtest",
        "//maga_transformer/cpp:cache",
        "//src/fastertransformer/devices:device_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

