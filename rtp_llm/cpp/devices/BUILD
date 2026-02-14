load("//:def.bzl", "copts")


# Shape checking utilities - standalone functions
cc_library(
    name = "shape_check",
    hdrs = [
        "ShapeCheck.h",
    ],
    srcs = [
        "ShapeCheck.cc",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)


cc_library(
    name = "devices_base",
    hdrs = glob([
        "*.h",
    ], exclude = []),
    srcs = glob([
        "*.cc",
    ], exclude = []),
    deps = [
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/core:memory_tracker",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:event",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/models_py/bindings:op_defs",
        "@havenask//aios/autil:scope",
        "//rtp_llm/cpp/utils:signal_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
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
        ":devices_base",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",
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
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
