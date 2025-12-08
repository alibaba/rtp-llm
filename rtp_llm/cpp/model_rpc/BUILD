load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "requirement", "embedding_arpc_deps")

requirement([
    "grpcio"
])

embedding_arpc_deps()

cc_library(
    name = "rpc_error_code",
    hdrs = [
        "RpcErrorCode.h",
    ],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "model_rpc_pool",
    srcs = [],
    hdrs = ["RPCPool.h"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "query_converter",
    srcs = [
        "QueryConverter.cc",
    ],
    hdrs = [
        "QueryConverter.h",
    ],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/engine_base/stream:stream",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "local_rpc_server",
    srcs = [
        "LocalRpcServer.cc",
        "GenerateContext.cc",
    ],
    hdrs = [
        "LocalRpcServer.h",
        "GenerateContext.h",
        "RpcServerRuntimeMeta.h",
    ],
    deps = [
        ":query_converter",
        ":rpc_error_code",
        ":embedding_arpc_deps",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/speculative_engine:speculative_engine",
        "//rtp_llm/cpp/multimodal_processor:multimodal_processor",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/cache_new:tp_broadcast_manager",
        "//rtp_llm/cpp/utils:atomic_util",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "remote_rpc_server",
    hdrs = [
        "RemoteServerResource.h",
        "RemoteRpcServer.h",
    ],
    srcs = [
        "RemoteRpcServer.cc",
    ],
    deps = [
        ":local_rpc_server",
        ":model_rpc_pool",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "prefill_entrance_rpc_server",
    srcs = [
        "PrefillRpcServer.cc",
        "DecodeRpcServer.cc",
        "PrefillGenerateContext.cc",
        "DecodeGenerateContext.cc",
    ],
    hdrs = [
        "PrefillRpcServer.h",
        "DecodeRpcServer.h",
        "PrefillGenerateContext.h",
        "DecodeGenerateContext.h",
    ],
    deps = [
        ":remote_rpc_server",
    ],
)

cc_library(
    name = "decode_entrance_rpc_server",
    srcs = [
        "PrefillRpcServerNew.cc",
        "DecodeRpcServerNew.cc",
        "PrefillGenerateContextNew.cc",
        "DecodeGenerateContextNew.cc",
    ],
    hdrs = [
        "PrefillRpcServerNew.h",
        "DecodeRpcServerNew.h",
        "PrefillGenerateContextNew.h",
        "DecodeGenerateContextNew.h",
    ],
    deps = [
        ":remote_rpc_server",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "decode_rpc_server_p2p",
    srcs = [
        "PrefillServerCaller.cc",
        "DecodeRpcServerNew2.cc",
        "PrefillRpcServerNew2.cc",
    ],
    hdrs = [
        "PrefillServerCaller.h",
        "DecodeRpcServerNew2.h",
        "PrefillRpcServerNew2.h",
    ],
    deps = [
        ":remote_rpc_server",
        "//rtp_llm/cpp/disaggregate/p2p_connector:prefill_connector",
        "//rtp_llm/cpp/disaggregate/p2p_connector:decode_connector",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "model_rpc_server",
    srcs = [
        "RemoteRpcServiceImpl.cc",
    ],
    hdrs = [
        "RemoteRpcServiceImpl.h",
        "LocalRpcServiceImpl.h",
    ],
    deps = [
        ":local_rpc_server",
        ":prefill_entrance_rpc_server",
        ":decode_entrance_rpc_server",
        ":decode_rpc_server_p2p",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "model_rpc_client",
    srcs = ["model_rpc_client.py"],
    deps = [
        ":grpcio",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto"
    ],
    visibility = ["//visibility:public"],
)
