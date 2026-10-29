# barex (accl) implementation of the public Tensor RDMA transport interface,
# open-sourced from the internal tree.
#
# accl-barex is a permanently closed library: the @alibaba_rdma repository is
# defined only by the internal deps overlay. srcs/deps below appear only when
# //:using_rdma is ON (no public platform config enables it), so a default
# public `bazel build //...` compiles an empty library and never resolves
# @alibaba_rdma. See README.md in this directory.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "barex_rdma_transport_impl",
    srcs = select({
        "//:using_rdma": ["BarexRdmaTransport.cc"],
        "//conditions:default": [],
    }),
    hdrs = ["BarexRdmaTransport.h"],
    # accl-barex links tnet on x86; arm builds disable it.
    copts = select({
        "//:using_cuda12_arm": ["-DDISABLE_TNET=1"],
        "//conditions:default": [],
    }),
    deps = select({
        # arm: accl without tnet (stricter key than using_rdma alone)
        "//:using_rdma_cuda12_arm": [
            "//rtp_llm/cpp/rdma_transport:rdma_transport",
            "//rtp_llm/cpp/rdma_transport:deferred_read_queue",
            "//rtp_llm/cpp/utils:core_utils",
            "//rtp_llm/cpp/utils:time_util",
            "@havenask//aios/autil:log",
            "//aios/network/rdma:accl_rdma",
            "@alibaba_rdma//aios/network/accl-barex:accl-barex",
            "@alibaba_rdma//aios/network/accl-barex/metric:kmonitor_metric_reporter",
        ],
        "//:using_rdma": [
            "//rtp_llm/cpp/rdma_transport:rdma_transport",
            "//rtp_llm/cpp/rdma_transport:deferred_read_queue",
            "//rtp_llm/cpp/utils:core_utils",
            "//rtp_llm/cpp/utils:time_util",
            "@havenask//aios/autil:log",
            "//aios/network/rdma:accl_rdma",
            "@alibaba_rdma//aios/network/accl-barex:accl-barex",
            "@alibaba_rdma//aios/network/accl-barex/metric:kmonitor_metric_reporter",
            "//3rdparty/tnet:tnet",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
