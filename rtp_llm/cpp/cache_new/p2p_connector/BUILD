package(default_visibility = ["//rtp_llm/cpp/cache_new:__subpackages__"])
load("@//bazel:arch_select.bzl", "cache_store_deps")

cache_store_deps()

cc_library(
    name = "arpc_dep",
    srcs = [
        "TcpClient.cc",
        "TcpServer.cc",
    ],
    hdrs = [
        "TcpClient.h",
        "TcpServer.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/network/arpc:arpc",
        "@havenask//aios/network/arpc/arpc/metric:kmonitor_anet_metric",
        "@havenask//aios/autil:net",
    ],
    visibility=['//visibility:public'],
)

cc_library(
    name = "basic_types",
    hdrs = [
        "CommonDefs.h",
        "LayerCacheBuffer.h",
    ],
    srcs = [
        "LayerCacheBuffer.cc",
    ],
    deps = [
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/autil:thread",
    ],
)

cc_library(
    name = "client",
    hdrs = [
        "CacheStoreClientLoadContext.h",
        "CacheStoreClientService.h",
        "CacheStoreClient.h",
    ],
    srcs = [
        "CacheStoreClientLoadContext.cc",
        "CacheStoreClientService.cc",
        "CacheStoreClient.cc",
    ],
    deps = [
        ":basic_types",
        ":arpc_dep",
        "//rtp_llm/cpp/cache_new/p2p_connector/proto:service_cc_proto",
        "//rtp_llm/cpp/cache_new:cache_new",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/metrics:metrics",
    ],
)

cc_library(
    name = "server",
    hdrs = [
        "CacheStoreServerService.h",
        "CacheStoreServer.h",
    ],
    srcs = [
        "CacheStoreServerService.cc",
        "CacheStoreServer.cc",
    ],
    deps = [
        ":basic_types",
        ":arpc_dep",
        "//rtp_llm/cpp/cache_new/p2p_connector/proto:service_cc_proto",
        "//rtp_llm/cpp/cache_new:cache_new",
        "//rtp_llm/cpp/core:event",
        "@havenask//aios/autil:thread",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/metrics:metrics",
    ],
)

cc_library(
    name = "connector",
    hdrs = ["KVCacheP2PConnector.h"],
    srcs = ["KVCacheP2PConnector.cc"],
    deps = [
        ":client",
        ":server",
        "//rtp_llm/cpp/cache_new:cache_new",
    ],
)