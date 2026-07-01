load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "embedding_arpc_deps", "requirement", "torch_deps")

requirement([
    "grpcio",
])

embedding_arpc_deps()

cc_library(
    name = "tensor_pb_convert",
    srcs = ["TensorPbConvert.cc"],
    hdrs = ["TensorPbConvert.h"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rpc_error_code",
    hdrs = [
        "RpcErrorCode.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/utils:error_code",
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
    name = "prefill_server_caller",
    srcs = [
        "PrefillServerCaller.cc",
        "PrefillServerCallerContext.cc",
    ],
    hdrs = [
        "PrefillServerCaller.h",
        "PrefillServerCallerContext.h",
        "RpcTimeoutUtils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":model_rpc_pool",
        ":rpc_error_code",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:deferred_completion_queue_drainer",
        "//rtp_llm/cpp/utils:grpc_address_util",
        "//rtp_llm/cpp/utils:time_util",
    ],
)

cc_library(
    name = "model_rpc_server",
    srcs = glob(
        ["*.cc"],
        exclude = [
            "TensorPbConvert.cc",
            "PrefillServerCaller.cc",
            "PrefillServerCallerContext.cc",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = [
            "RPCPool.h",
            "PrefillServerCaller.h",
            "PrefillServerCallerContext.h",
            "RpcErrorCode.h",
            "RpcTimeoutUtils.h",
            "TensorPbConvert.h",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":embedding_arpc_deps",
        ":model_rpc_pool",
        ":prefill_server_caller",
        ":rpc_error_code",
        ":tensor_pb_convert",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache/connector:connector",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/multimodal_processor",
        "//rtp_llm/cpp/normal_engine",
        "//rtp_llm/cpp/utils:atomic_util",
        "//rtp_llm/cpp/utils:grpc_address_util",
        "//rtp_llm/cpp/utils:profiling_scope",
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
    visibility = ["//visibility:public"],
    deps = [
        ":model_rpc_pool",
    ],
)
