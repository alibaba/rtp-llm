package(default_visibility = ["//visibility:public"])

load("@//bazel:arch_select.bzl", "transfer_rdma_deps", "torch_deps")

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
        "//rtp_llm/cpp/cache/connector/p2p/transfer/proto:service_cc_proto",
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
        "LayerCacheBuffer.h",
        "LayerBlockConvertor.h",
        "LayerCacheBufferUtil.h",
        "TransferTask.h",
    ],
    srcs = [
        "LayerCacheBuffer.cc",
        "LayerCacheBufferUtil.cc",
        "TransferTask.cc",
    ],
    deps = [
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "@havenask//aios/autil:thread",
    ],
    visibility=['//visibility:public'],
)

cc_library(
    name = "transfer",
    hdrs = ["TransferClient.h", "TransferServer.h", "CudaCopyUtil.h", "TransferServerService.h", "TransferTaskContext.h", "TransferMetric.h"],
    srcs = ["TransferClient.cc", "TransferServer.cc", "TransferServerService.cc", "TransferTaskContext.cc", "TransferMetric.cc", "CudaCopyUtil.cc"],
    deps = [
        ":arpc_dep",
        ":basic_types",
        ":rdma_header",
        ":transfer_rdma_impl",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/proto:service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/cache:cache_core",
        "@havenask//aios/autil:net",
        "//rtp_llm/cpp/metrics:metrics",
    ],
    visibility=['//visibility:public'],
)