load("//:def.bzl", "copts")

cc_library(
    name = "arm_cpu_impl",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/core:cpu_allocator",
        "@arm_compute//arm_compute:core_headers",
        "@arm_compute//arm_compute:runtime_headers",
        "@arm_compute//support",
    ],
    visibility = ["//visibility:public"],
    copts = copts() + ["-fopenmp"],
    alwayslink = 1,
)
