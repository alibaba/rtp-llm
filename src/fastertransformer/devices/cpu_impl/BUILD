load("//:def.bzl", "copts")

cc_library(
    name = "cpu_impl",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/core:cpu_allocator",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)
