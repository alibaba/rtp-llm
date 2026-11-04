# barex/accl RDMA cache-store implementation, open-sourced from the internal
# tree.
#
# accl-barex is a permanently closed library: the @alibaba_rdma repository is
# defined only by the internal deps overlay. srcs/hdrs/deps appear only when
# //:using_rdma is ON (no public platform config enables it), so a default
# public `bazel build //...` compiles an empty library and never resolves
# @alibaba_rdma. Building with --define=use_rdma=true requires the internal
# overlay; see rtp_llm/cpp/rdma_transport/barex/README.md.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cache_store_rdma_impl",
    srcs = select({
        "//:using_rdma": [
            "Impl.cpp",
            "RdmaCacheStoreLoadServiceClosure.cpp",
            "RdmaCacheStoreServiceImpl.cpp",
            "RdmaCacheStoreServiceImplContext.cpp",
            "RdmaClient.cpp",
            "RdmaClientConnectionCallback.cpp",
            "RdmaClientHostConnections.cpp",
            "RdmaConnection.cpp",
            "RdmaMemoryUtilImpl.cpp",
            "RdmaMessager.cpp",
            "RdmaServer.cpp",
            "RdmaServerConnectionCallback.cpp",
            "RdmaServerHostConnections.cpp",
            "RdmaTransferConnection.cpp",
        ],
        "//conditions:default": [],
    }),
    hdrs = select({
        "//:using_rdma": [
            "RdmaCacheStoreLoadServiceClosure.h",
            "RdmaCacheStoreServiceImpl.h",
            "RdmaCacheStoreServiceImplContext.h",
            "RdmaClient.h",
            "RdmaClientConnectionCallback.h",
            "RdmaClientHostConnections.h",
            "RdmaConnection.h",
            "RdmaMemoryUtilImpl.h",
            "RdmaMessager.h",
            "RdmaServer.h",
            "RdmaServerConnectionCallback.h",
            "RdmaServerHostConnections.h",
            "RdmaTransferConnection.h",
        ],
        "//conditions:default": [],
    }),
    # accl-barex links tnet on x86; arm builds disable it.
    copts = select({
        "//:using_cuda12_arm": ["-DDISABLE_TNET=1"],
        "//conditions:default": [],
    }),
    deps = select({
        "//:using_rdma": [
            "//rtp_llm/cpp/disaggregate/cache_store:cache_store_base",
            "//aios/network/rdma:accl_rdma",
            "@alibaba_rdma//aios/network/accl-barex:accl-barex",
            "@alibaba_rdma//aios/network/accl-barex/metric:kmonitor_metric_reporter",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    alwayslink = True,
)
