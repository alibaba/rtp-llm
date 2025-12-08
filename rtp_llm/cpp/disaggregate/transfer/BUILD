package(default_visibility = ["//visibility:public"])

load("@//bazel:arch_select.bzl", "transfer_rdma_deps")

transfer_rdma_deps()

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
    name = "rdma_header",
    hdrs = ["RdmaInterface.h"],
    srcs = [],
    deps = [
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/disaggregate/transfer/proto:service_cc_proto",
    ],
    visibility = ['//visibility:public']
)

cc_library(
    name = "no_rdma_impl",
    srcs = ["RdmaInterface.cc"],
    hdrs = [],
    deps = [
        ":rdma_header",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "basic_types",
    hdrs = [
        "CommonDefs.h",
        "LayerCacheBuffer.h",
        "LayerBlockConvertor.h",
        "LayerCacheBufferUtil.h",
    ],
    srcs = [
        "LayerCacheBuffer.cc",
        "LayerCacheBufferUtil.cc",
    ],
    deps = [
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/utils:core_utils",
    "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/cache_new:batch_kv_cache_resource",
        "@havenask//aios/autil:thread",
    ],
    visibility=['//visibility:public'],
)

cc_library(
    name = "transfer_client",
    hdrs = ["TransferClient.h"],
    srcs = ["TransferClient.cc"],
    deps = [
        ":arpc_dep",
        ":basic_types",
        ":rdma_header",
        ":transfer_rdma_impl",
        "//rtp_llm/cpp/disaggregate/transfer/proto:service_cc_proto",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:net",
    ],
    visibility=['//visibility:public'],
)

cc_library(
    name = "transfer_server",
    hdrs = ["TransferServer.h", "TransferServerService.h"],
    srcs = ["TransferServer.cc", "TransferServerService.cc"],
    deps = [
        ":arpc_dep",
        ":basic_types",
        ":rdma_header",
        ":transfer_rdma_impl",
        "//rtp_llm/cpp/disaggregate/transfer/proto:service_cc_proto",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility=['//visibility:public'],
)

