load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//:__subpackages__"])


cc_library(
    name = "rpc_error_code",
    hdrs = [
        "RpcErrorCode.h",
    ],
    deps = [
        "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
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
        "utils.h",
    ],
    srcs = [
        "AssertUtils.cc",
        "Logger.cc",
        "Exception.cc",
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

cc_library(
    name = "system_utils",
    hdrs = [
        "SignalUtils.h",
        "StackTrace.h",
        "TimeUtil.h",
        "ScopeGuard.h",
    ],
    srcs = [
        "SignalUtils.cc",
        "StackTrace.cc",
    ],
    deps = [
        ":core_utils",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:net",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "data_structures",
    hdrs = [
        "LRUCache.h",
        "HashUtil.h",
        "AtomicUtil.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "domain_utils",
    hdrs = [
        "KVCacheUtils.h",
        "ShapeCheck.h",
        "DFAUtil.h",
        "PrefixToCandidateTokens.h",
    ],
    srcs = [
        "ShapeCheck.cc",
    ],
    deps = [
        ":core_utils",
        "//rtp_llm/cpp/config:config_modules",
    ],
    implementation_deps = [
        "//rtp_llm/cpp/config:static_config",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "py_utils",
    hdrs = [
        "PyUtils.h",
    ],
    srcs = [
        "PyUtils.cc",
    ],
    deps = torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
