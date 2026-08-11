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
    name = "prefill_peer_selector",
    hdrs = ["PrefillPeerSelector.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "prefill_run_waiter",
    hdrs = ["PrefillRunWaiter.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rpc_fanout_utils",
    hdrs = ["RpcFanoutUtils.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "remote_load_budget",
    hdrs = ["RemoteLoadBudget.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "request_deadline_budget",
    hdrs = ["RequestDeadlineBudget.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "propagated_client_context",
    hdrs = ["PropagatedClientContext.h"],
    visibility = ["//visibility:public"],
    deps = ["@grpc//:grpc++"],
)

cc_library(
    name = "remote_load_target_tracker",
    hdrs = ["RemoteLoadTargetTracker.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache_transfer_block_selector",
    hdrs = ["CacheTransferBlockSelector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:cache_group_type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "cache_transfer_lease",
    hdrs = ["CacheTransferLease.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":remote_load_fence",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "remote_load_fence",
    srcs = ["RemoteLoadFence.cc"],
    hdrs = ["RemoteLoadFence.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":remote_load_budget",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "remote_load_lease_retainer",
    srcs = ["RemoteLoadLeaseRetainer.cc"],
    hdrs = ["RemoteLoadLeaseRetainer.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "sampler_generator_state",
    srcs = ["SamplerGeneratorState.cc"],
    hdrs = ["SamplerGeneratorState.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + torch_deps(),
)

cc_library(
    name = "model_rpc_server",
    srcs = glob(
        ["*.cc"],
        exclude = [
            "RemoteLoadFence.cc",
            "RemoteLoadLeaseRetainer.cc",
            "SamplerGeneratorState.cc",
            "TensorPbConvert.cc",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = [
            "CacheTransferLease.h",
            "PrefillPeerSelector.h",
            "PrefillRunWaiter.h",
            "CacheTransferBlockSelector.h",
            "PropagatedClientContext.h",
            "RPCPool.h",
            "RemoteLoadBudget.h",
            "RemoteLoadFence.h",
            "RemoteLoadLeaseRetainer.h",
            "RemoteLoadTargetTracker.h",
            "RequestDeadlineBudget.h",
            "RpcFanoutUtils.h",
            "SamplerGeneratorState.h",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":embedding_arpc_deps",
        ":cache_transfer_block_selector",
        ":cache_transfer_lease",
        ":model_rpc_pool",
        ":prefill_peer_selector",
        ":prefill_run_waiter",
        ":propagated_client_context",
        ":rpc_error_code",
        ":remote_load_budget",
        ":remote_load_fence",
        ":remote_load_lease_retainer",
        ":remote_load_target_tracker",
        ":request_deadline_budget",
        ":rpc_fanout_utils",
        ":sampler_generator_state",
        ":tensor_pb_convert",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/disaggregate/cache_store",
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/multimodal_processor",
        "//rtp_llm/cpp/normal_engine",
        "//rtp_llm/cpp/utils:atomic_util",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/utils:request_admission_gate",
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
