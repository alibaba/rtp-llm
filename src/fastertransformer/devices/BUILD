load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "devices_base",
    hdrs = glob([
        "*.h",
    ], exclude = [
        "DeviceFactory.h"
    ]),
    srcs = glob([
        "*.cc",
    ], exclude = [
        "DeviceFactory.cc"
    ]),
    deps = [
        "//src/fastertransformer/core:allocator",
        "//src/fastertransformer/core:buffer",
        "//src/fastertransformer/utils:utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "device_factory",
    hdrs = glob([
        "DeviceFactory.h",
    ]),
    srcs = glob([
        "DeviceFactory.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices/cpu_impl:cpu_impl",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

