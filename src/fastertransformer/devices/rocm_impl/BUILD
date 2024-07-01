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
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
    ],
    visibility = ["//visibility:public"],
    copts = rocm_copts(),
)
