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
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
        "@havenask//aios/network/arpc:arpc",
        "@havenask//aios/network/arpc/arpc/metric:kmonitor_anet_metric",
    ],
    visibility=['//visibility:public'],
)

cc_library(
    name = "cache_store_interface",
    hdrs = [
        "CommonDefine.h",
        "CacheStore.h",
        "MemoryUtil.h",
        "RequestBlockBuffer.h",
        "LoadContext.h",
    ],
    srcs = [
        "RequestBlockBuffer.cpp",
        "LoadContext.cpp",
    ],
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:env_util",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:event",
    ],
    visibility=['//visibility:public'],
)

cc_library(
    name = "cache_store_base",
    srcs = [
        "RequestBlockBufferStore.cpp",
        "CacheStoreServiceImpl.cpp",
        "CacheStoreServiceImplContext.cpp",
        "MessagerServer.cpp",
        "CacheLoadServiceClosure.cpp",
        "MessagerClient.cpp",
        "NormalCacheStore.cpp",
        "NoRdmaMemoryUtilImpl.cpp",
        "TimerManager.cpp",
        "Timer.cpp"
    ],
    hdrs = [
        "NoRdmaMemoryUtilImpl.h",
        "Interface.h",
        "RequestBlockBufferStore.h",
        "InitParams.h",
        "CacheStoreServiceImpl.h",
        "CacheStoreServiceImplContext.h",
        "MessagerServer.h",
        "CacheLoadServiceClosure.h",
        "MessagerClient.h",
        "NormalCacheStore.h",
        "TimerManager.h",
        "Timer.h"
    ],
    deps = [
        ":cache_store_interface",
        "//rtp_llm/cpp/disaggregate/cache_store/proto:cache_store_service_cc_proto",
        "//rtp_llm/cpp/disaggregate/cache_store/metrics:metrics",
        "//rtp_llm/cpp/devices:devices_base",
        "@havenask//aios/autil:thread",
        "@havenask//aios/autil:net",
        ":arpc_dep",
    ],
    visibility=['//visibility:public'],
)
