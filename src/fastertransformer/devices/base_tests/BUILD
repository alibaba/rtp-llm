load("//:def.bzl", "copts")

cc_library(
    name = "base_tests",
    hdrs = glob([
        "*.hpp",
    ]),
    deps = [
        "//src/fastertransformer/devices/testing:device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
