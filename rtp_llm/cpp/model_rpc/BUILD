load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "embedding_arpc_deps", "requirement")

requirement([
    "grpcio",
])

embedding_arpc_deps()

cc_library(
    name = "rpc_error_code",
    hdrs = [
        "RpcErrorCode.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
)

cc_library(
    name = "model_rpc_pool",
    srcs = [],
    hdrs = ["RPCPool.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "model_rpc_server",
    srcs = glob(["*.cc"]),
    hdrs = glob(
        ["*.h"],
        exclude = ["RPCPool.h"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":embedding_arpc_deps",
        ":model_rpc_pool",
        ":rpc_error_code",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/embedding_engine",
        "//rtp_llm/cpp/engine_base/schedulers",
        "//rtp_llm/cpp/engine_base/stream",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/multimodal_processor",
        "//rtp_llm/cpp/normal_engine",
        "//rtp_llm/cpp/speculative_engine",
        "//rtp_llm/cpp/utils:atomic_util",
    ],
)

py_library(
    name = "model_rpc_client",
    srcs = ["model_rpc_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":grpcio",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
    ],
)

cc_library(
    name = "broadcast_manager",
    srcs = [],
    hdrs = ["BroadcastManager.h"],
    deps = [
        ":model_rpc_pool",
    ],
    visibility = ["//visibility:public"]
)
