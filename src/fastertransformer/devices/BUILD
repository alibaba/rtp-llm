load("//:def.bzl", "copts")

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
        "//src/fastertransformer/core:buffer_torch",
        "//src/fastertransformer/core:buffer",
        "//maga_transformer/cpp:utils",
        "//src/fastertransformer/utils",
        "//:gpt_init_params_hdr",
        "//src/fastertransformer/models:weights_define",
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
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
    alwayslink = 1,
)

cc_library(
    name = "device_py_export",
    hdrs = [
        "DeviceExport.h",
    ],
    srcs = [
        "DeviceExport.cc",
    ],
    deps = [
        ":devices_base_impl",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
