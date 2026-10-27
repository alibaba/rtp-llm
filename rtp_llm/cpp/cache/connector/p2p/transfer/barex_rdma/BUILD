# barex/accl RDMA transfer backend, open-sourced from the internal tree.
#
# accl-barex is a permanently closed library: the @alibaba_rdma repository is
# defined only by the internal deps overlay. Every target below that links
# accl (directly or through //3rdparty/tnet:tnet) gates its srcs/hdrs/deps on
# //:using_rdma, which no public platform config enables. With use_rdma OFF a
# default public `bazel build //...` compiles these targets as empty
# libraries and never resolves @alibaba_rdma. Building with
# --define=use_rdma=true requires the internal overlay; without it the build
# fails with 'no such package @alibaba_rdma//...' — this is intentional (see
# rtp_llm/cpp/rdma_transport/barex/README.md).

load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

# The transfer package's transfer_backend_rdma_impl compiles the RDMA-variant
# factory from this package (the public base file keeps the TransferBackendFactory
# name in the transfer package, so the renamed variant lives here).
exports_files(["TransferBackendFactoryRdma.cc"])

# accl-free interface header: always buildable.
cc_library(
    name = "rdma_interface",
    hdrs = ["RdmaInterface.h"],
    deps = [
        "//rtp_llm/cpp/cache:block_info",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

# Sources and headers include accl-barex interface headers, so they only
# appear when use_rdma is ON.
cc_library(
    name = "transfer_rdma_impl",
    srcs = select({
        "//:using_rdma": [
            "RdmaMemoryManager.cc",
            "RdmaConnection.cc",
            "RdmaClient.cc",
            "RdmaServer.cc",
            "RdmaHostConnectionPool.cc",
            "RdmaMetric.cc",
            "RdmaInterface.cc",
        ],
        "//conditions:default": [],
    }),
    hdrs = select({
        "//:using_rdma": [
            "RdmaMemoryManager.h",
            "RdmaConnection.h",
            "RdmaClient.h",
            "RdmaServer.h",
            "RdmaHostConnectionPool.h",
            "RdmaMetric.h",
        ],
        "//conditions:default": [],
    }),
    # accl-barex links tnet on x86; arm builds disable it.
    copts = select({
        "//:using_arm": ["-DDISABLE_TNET=1"],
        "//:using_cuda12_arm": ["-DDISABLE_TNET=1"],
        "//conditions:default": [],
    }),
    deps = select({
        # arm: accl without tnet (stricter keys than using_rdma alone)
        "//:using_rdma_arm": [
            ":rdma_interface",
            "//rtp_llm/cpp/cache/connector/p2p/transfer:arpc_dep",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto:rdma_service_cc_proto",
            "//rtp_llm/cpp/utils:core_utils",
            "//rtp_llm/cpp/utils:time_util",
            "//rtp_llm/cpp/metrics:metrics",
            "@havenask//aios/autil:log",
            "@havenask//aios/network/arpc:arpc",
            "//aios/network/rdma:accl_rdma",
            "@alibaba_rdma//aios/network/accl-barex:accl-barex",
            "@alibaba_rdma//aios/network/accl-barex/metric:kmonitor_metric_reporter",
        ],
        "//:using_rdma_cuda12_arm": [
            ":rdma_interface",
            "//rtp_llm/cpp/cache/connector/p2p/transfer:arpc_dep",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto:rdma_service_cc_proto",
            "//rtp_llm/cpp/utils:core_utils",
            "//rtp_llm/cpp/utils:time_util",
            "//rtp_llm/cpp/metrics:metrics",
            "@havenask//aios/autil:log",
            "@havenask//aios/network/arpc:arpc",
            "//aios/network/rdma:accl_rdma",
            "@alibaba_rdma//aios/network/accl-barex:accl-barex",
            "@alibaba_rdma//aios/network/accl-barex/metric:kmonitor_metric_reporter",
        ],
        "//:using_rdma": [
            ":rdma_interface",
            "//rtp_llm/cpp/cache/connector/p2p/transfer:arpc_dep",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto:rdma_service_cc_proto",
            "//rtp_llm/cpp/utils:core_utils",
            "//rtp_llm/cpp/utils:time_util",
            "//rtp_llm/cpp/metrics:metrics",
            "@havenask//aios/autil:log",
            "@havenask//aios/network/arpc:arpc",
            "//aios/network/rdma:accl_rdma",
            "@alibaba_rdma//aios/network/accl-barex:accl-barex",
            "@alibaba_rdma//aios/network/accl-barex/metric:kmonitor_metric_reporter",
            "//3rdparty/tnet:tnet",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rdma_task_context",
    hdrs = ["RdmaTaskContext.h"],
    srcs = ["RdmaTaskContext.cc"],
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_metric",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        ":rdma_interface",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto:rdma_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rdma_transfer_service",
    hdrs = ["RdmaTransferService.h"],
    srcs = ["RdmaTransferService.cc"],
    deps = [
        ":rdma_task_context",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        ":rdma_interface",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto:rdma_service_cc_proto",
        "@havenask//aios/autil:thread",
        "@havenask//aios/autil:lock_free",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rdma_backend",
    hdrs = [
        "RdmaKVCacheSender.h",
        "RdmaKVCacheReceiver.h",
    ],
    srcs = select({
        "//:using_rdma": [
            "RdmaKVCacheSender.cc",
            "RdmaKVCacheReceiver.cc",
        ],
        "//conditions:default": [],
    }),
    deps = select({
        "//:using_rdma": [
            "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_metric",
            ":rdma_transfer_service",
            ":transfer_rdma_impl",
            "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
            ":rdma_interface",
            "//rtp_llm/cpp/cache/connector/p2p/transfer:arpc_dep",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto:rdma_service_cc_proto",
            "//rtp_llm/cpp/utils:core_utils",
            "//rtp_llm/cpp/utils:time_util",
            "@havenask//aios/network/arpc:arpc",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
