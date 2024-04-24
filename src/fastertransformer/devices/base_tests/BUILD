load("//:def.bzl", "copts")

cc_library(
    name = "base_tests",
    hdrs = glob([
        "*.hpp",
    ]),
    deps = [
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/torch_impl:torch_reference_impl",
        "//maga_transformer/cpp:cache",
        "//maga_transformer/cpp:models",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
