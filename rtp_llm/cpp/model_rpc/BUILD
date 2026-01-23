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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "query_converter",
    srcs = ["QueryConverter.cc"],
    hdrs = ["QueryConverter.h"],
    deps = [
        ":rpc_error_code",
        ":model_rpc_pool",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "model_rpc_server",
    srcs =  glob(["*.cc"], exclude = ["QueryConverter.cc"]),
    hdrs =  glob(["*.h"], exclude = ["RPCPool.h", "QueryConverter.h"]),
    deps = [
        ":embedding_arpc_deps",
        ":model_rpc_pool",
        ":rpc_error_code",
        ":query_converter",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/speculative_engine:speculative_engine",
        "//rtp_llm/cpp/multimodal_processor:multimodal_processor",
        "//rtp_llm/cpp/utils:atomic_util",
        "//:rtp_compute_ops",
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

cc_library(
    name = "tp_broadcast_manager",
    srcs = [],
    hdrs = ["TpBroadcastManager.h"],
    deps = [
        ":model_rpc_pool",
    ],
    visibility = ["//visibility:public"],
)