load("//:def.bzl", "copts")

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "@com_google_googletest//:gtest",
        "//src/fastertransformer/devices:device_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "ffn_layer_test",
    hdrs = [
        "FfnLayerTest.hpp",
    ],
    deps = [
        ":device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
