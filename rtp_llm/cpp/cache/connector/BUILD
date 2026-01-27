load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "connector",
    hdrs = [
        "KVCacheConnectorCoordinator.h",
        "KVCacheConnectorReadWriteContext.h",
    ],
    srcs = [
        "KVCacheConnectorCoordinator.cc",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:thread",
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/cache/connector/p2p:layer_block_convertor",
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "igenerate_stream",
    hdrs = [
        "IGenerateStream.h",
    ],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "connector_header",
    hdrs = [
        "AsyncContext.h",
        "KVCacheConnector.h",
        "KVCacheConnectorReadWriteContext.h",
        "IKVCacheConnectorCoordinator.h",
    ],
    srcs = [
        "AsyncContext.cc",
    ],
    copts = copts(),
    deps = [
        ":igenerate_stream",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

