load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//:__subpackages__"])



# KVCache utilities - simple inline functions, no external dependencies
cc_library(
    name = "kv_cache_utils",
    hdrs = [
        "KVCacheUtils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# New modular targets
cc_library(
    name = "core_utils",
    hdrs = [
        "AssertUtils.h",
        "Logger.h",
        "StringUtil.h",
        "StatusUtil.h",
        "Exception.h",
        "utils.h"
    ],
    srcs = [
        "AssertUtils.cc",
        "Logger.cc",
        "Exception.cc"
    ],
    deps = [
        "@havenask//aios/alog:alog",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:time",
        "@havenask//aios/autil:net",
    ],
    implementation_deps = [
        "//rtp_llm/cpp/config:static_config",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "error_code",
    hdrs = [
        "ErrorCode.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Signal handling utilities
cc_library(
    name = "signal_utils",
    hdrs = [
        "SignalUtils.h",
    ],
    srcs = [
        "SignalUtils.cc",
    ],
    deps = [
        ":core_utils",
        ":stack_trace",
        "@havenask//aios/autil:env_util",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

# Stack trace utilities
cc_library(
    name = "stack_trace",
    hdrs = [
        "StackTrace.h",
    ],
    srcs = [
        "StackTrace.cc",
    ],
    deps = [
        ":core_utils",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
        "@local_config_python//:python_headers",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Time utilities
cc_library(
    name = "time_util",
    hdrs = [
        "TimeUtil.h",
    ],
    deps = [
        "@havenask//aios/autil:time",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# LRU Cache implementation
cc_library(
    name = "lru_cache",
    hdrs = [
        "LRUCache.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Hash utilities
cc_library(
    name = "hash_util",
    hdrs = [
        "HashUtil.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Atomic utilities
cc_library(
    name = "atomic_util",
    hdrs = [
        "AtomicUtil.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Math utilities
cc_library(
    name = "math_utils",
    hdrs = [
        "math_utils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
