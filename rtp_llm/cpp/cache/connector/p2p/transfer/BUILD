package(default_visibility = ["//visibility:public"])

load("@//bazel:arch_select.bzl", "transfer_backend_deps")

transfer_backend_deps()

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
    visibility = ['//visibility:public'],
)

cc_library(
    name = "transfer_metric",
    hdrs = ["TransferMetric.h"],
    srcs = ["TransferMetric.cc"],
    deps = [
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:time_util",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "transfer_backend_config",
    hdrs = ["TransferBackendConfig.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "basic_types",
    hdrs = [
        "Types.h",
        "TransferErrorCode.h",
        "IKVCacheSender.h",
        "IKVCacheReceiver.h",
        "TransferTask.h",
    ],
    srcs = [
        "TransferTask.cc",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/cache:block_info",
    ],
    visibility = ['//visibility:public'],
)

cc_library(
    name = "transfer_backend_factory_interface",
    hdrs = ["TransferBackendFactory.h"],
    deps = [
        ":basic_types",
        ":transfer_backend_config",
        "//rtp_llm/cpp/metrics:metrics",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "transfer_backend_base_impl",
    srcs = ["TransferBackendFactory.cc"],
    deps = [
        ":transfer_backend_factory_interface",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp:tcp_backend",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "backend_factory",
    hdrs = ["TransferBackendFactory.h"],
    deps = [
        ":transfer_backend_arch_select_impl",
        ":transfer_backend_factory_interface",
    ],
    visibility = ['//visibility:public'],
)

