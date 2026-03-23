package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_copy_util",
    hdrs = ["CudaCopyUtil.h"],
    srcs = ["CudaCopyUtil.cc"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tcp_task_context",
    hdrs = ["TcpTaskContext.h"],
    srcs = ["TcpTaskContext.cc"],
    deps = [
        ":cuda_copy_util",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_metric",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto:tcp_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/core:buffer",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tcp_transfer_service",
    hdrs = ["TcpTransferService.h"],
    srcs = ["TcpTransferService.cc"],
    deps = [
        ":cuda_copy_util",
        ":tcp_task_context",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto:tcp_service_cc_proto",
        "@havenask//aios/autil:thread",
        "@havenask//aios/autil:lock_free",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tcp_backend",
    hdrs = [
        "TcpKVCacheSender.h",
        "TcpKVCacheReceiver.h",
    ],
    srcs = [
        "TcpKVCacheSender.cc",
        "TcpKVCacheReceiver.cc",
    ],
    deps = [
        ":cuda_copy_util",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_metric",
        ":tcp_transfer_service",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:arpc_dep",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto:tcp_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/network/arpc:arpc",
    ],
    visibility = ["//visibility:public"],
)

