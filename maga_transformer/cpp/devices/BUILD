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
        "//maga_transformer/cpp/core:allocator",
        "//maga_transformer/cpp/core:memory_tracker",
        "//maga_transformer/cpp/core:buffer_torch",
        "//maga_transformer/cpp/core:buffer",
        "//maga_transformer/cpp/core:event",
        "//maga_transformer/cpp:utils",
        "//:gpt_init_params",
        "//maga_transformer/cpp/models_weight:weights_define",
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store_interface",
        "//maga_transformer/cpp/stats:stats",
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
        "//maga_transformer/cpp/core:buffer_torch"
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
