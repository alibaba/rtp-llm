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
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/core:memory_tracker",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:event",
        "//rtp_llm/cpp:utils",
        "//:gpt_init_params",
        "//rtp_llm/cpp/models_weight:weights_define",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/stats:stats",
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
        "//rtp_llm/cpp/core:buffer_torch"
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
