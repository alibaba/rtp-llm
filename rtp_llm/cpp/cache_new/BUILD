load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cache_new_types",
    hdrs = [
        "types.h",
        "CacheConfig.h",
    ],
    deps = [
        ":batch_kv_cache_resource",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:types",
        "//:rtp_compute_ops",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)


cc_library(
    name = "batch_kv_cache_resource",
    hdrs = ["BatchKVCacheResource.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache_new",
    hdrs = glob([
        "*.h",
    ], exclude = ["types.h", "CacheConfig.h", "KVCacheResource.h", "BatchKVCacheResource.h", "KVCacheConnector.h", "TpBroadcastManager.h"]),
    srcs = glob([
        "*.cc",
    ], exclude = ["TpBroadcastManager.cc"]),
    deps = [
        ":cache_new_types",
        ":batch_kv_cache_resource",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/metrics:metrics",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + torch_deps() + select({
        "//:using_3fs": [
            ":storage_3fs",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)


cc_library(
    name = "tp_broadcast_manager",
    hdrs = ["TpBroadcastManager.h"],
    srcs = ["TpBroadcastManager.cc"],
    deps = [
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:thread",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "connector_interface",
    srcs = [],
    hdrs = [
        "KVCacheConnector.h",
    ],
    deps = [
        "//rtp_llm/cpp/core:event",
        ":batch_kv_cache_resource",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "storage_3fs",
    hdrs = [
    ],
    srcs = [
    ],
    deps = [
        "//3rdparty/3fs:hf3fs",
    ],
    visibility = ["//visibility:public"],
)
