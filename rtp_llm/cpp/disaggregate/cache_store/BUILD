load("@//bazel:arch_select.bzl", "cache_store_deps")

cache_store_deps()

package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

cc_library(
    name = "cache_store",
    deps = [":cache_store_arch_select_impl", ":cache_store_base"],
    visibility=['//visibility:public'],
)

cc_library(
    name = "cache_store_base_impl",
    deps = [":cache_store_base"],
    srcs = ["Impl.cpp"],
    alwayslink = True,
)

cc_library(
    name = "arpc_dep",
    srcs = [],
    hdrs = [],
    deps = [
        "@havenask//aios/network/arpc:arpc",
        "@havenask//aios/network/arpc/arpc/metric:kmonitor_anet_metric",
    ],
    visibility=['//visibility:public'],
)

interface_includes = [
    "CommonDefine.h",
    "CacheStore.h",
    "CacheStoreMetricsCollector.h",
    "MemoryUtil.h",
    "RequestBlockBuffer.h",
    "LoadContext.h",
    "RemoteStoreTask.h",
    "MessagerRequest.h",
    "ErrorCodeUtil.h",
]

interface_sources = [
    "RequestBlockBuffer.cpp",
    "LoadContext.cpp",
]

cc_library(
    name = "cache_store_interface",
    hdrs = interface_includes,
    srcs = interface_sources,
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:env_util",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:time_util",
        # "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:event",
    ],
    visibility=['//visibility:public'],
    alwayslink = True,
)

cc_library(
    name = "cache_store_base",
    srcs = glob(["*.cpp"], exclude = ["Impl.cpp"] + interface_sources),
    hdrs = glob(["*.h"], exclude = interface_includes),
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/proto:cache_store_service_cc_proto",
        "//:rtp_compute_ops",
        ":arpc_dep",
    ],
    visibility=['//visibility:public'],
    alwayslink = True,
)
