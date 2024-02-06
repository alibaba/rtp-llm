load("//:def.bzl", "copts")

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

