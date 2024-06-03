load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "devices_base",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/core:allocator",
        "//src/fastertransformer/core:memory_tracker",
        "//src/fastertransformer/core:buffer",
        "//src/fastertransformer/utils:utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "device_utils",
    hdrs = glob([
        "utils/*.h",
    ]),
    srcs = glob([
        "utils/*.cc",
    ]),
    deps = [
        "devices_base",
        "//src/fastertransformer/core:buffer_torch"
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "devices_base_impl",
    srcs = glob([
        "base_impl/*.cc",
    ]),
    deps = [
        ":devices_base",
        ":device_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

