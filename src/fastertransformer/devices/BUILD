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
        "//src/fastertransformer/core:tensor_hdr",
        "//src/fastertransformer/utils:utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "cpu_impl",
    hdrs = glob([
        "cpu_impl/*.h",
    ]),
    srcs = glob([
        "cpu_impl/*.cc",
    ]),
    deps = [
        ":devices_base",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "cuda_impl",
    hdrs = glob([
        "cuda_impl/*.h",
    ]),
    srcs = glob([
        "cuda_impl/*.cc",
    ]),
    deps = [
        ":devices_base",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/core:allocator_cuda",
    ],
    visibility = ["//visibility:public"],
    copts = cuda_copts(),
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
        ":cpu_impl",
        ":cuda_device_impl",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

