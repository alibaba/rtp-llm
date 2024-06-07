load("//:def.bzl", "copts", "rocm_copts")

cc_library(
    name = "rocm_impl",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/core:allocator",
    ],
    visibility = ["//visibility:public"],
    copts = rocm_copts(),
)
